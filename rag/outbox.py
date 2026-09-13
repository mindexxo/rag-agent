"""색인 작업 대기열 — 트랜잭셔널 outbox (#139). 어휘·처리 규약의 **정의점**.

## 요점 하나

**색인에 할 일을 PG 변경과 같은 트랜잭션에 남긴다.** 업로드·삭제·토글·FAQ 변경이 커밋되면
그에 따른 색인 작업이 반드시 기록돼 있고, 롤백되면 그 일도 함께 사라진다. 커밋과 등재가
원자적이라는 것 — 그게 이 패턴의 전부다. 등재는 `session.add` 한 줄이라 네트워크도 await도 없다.

이전 방식(커밋 후 직접 호출, arq 잡 등록)은 그 사이에 프로세스가 죽거나 Redis가 순단하면
색인이 조용히 낡았다. 색인이 서빙의 정본인 구성(rag/opensearch.py)에서 그 누락은 곧
낡은 답변이고, 상담 답변은 고객에게 나간 뒤에야 알게 되므로 되돌릴 수 없다.

## 처리 — 단일 워커, cron 1분, 폴링

워커 cron(rag/worker.py)이 1분마다 `drain_once`를 부른다. pending 행을 **id 순으로** 하나씩
처리하고 성공하면 done. 실패는 attempts에 쌓여 MAX_ATTEMPTS를 넘으면 failed로 확정한다
(INDEX_DOCUMENT면 문서도 failed — FE가 그걸로 실패를 본다). 행은 지우지 않는다: done/failed
행이 이력이자 관측 지점이다. 보존 정리는 나중에 스케줄러로.

**의도적으로 하지 않은 것**과 그 트리거:
- 워커 여러 대 → 지금은 단일 워커라 프로세스 안 `asyncio.Lock` 하나로 cron 겹침(1분 넘게
  도는 회차)을 막는다. 워커를 늘리면 claim 컬럼(+ 고아 회수)이 필요하다.
  **다른 프로세스**의 drain(테스트 `ingest`, eval 스크립트가 워커 cron과 동시에)은 이 잠금이
  못 막는다 — SELECT에 행 잠금이 없어 같은 pending 행을 둘이 뽑을 수 있다. 결과는 멱등으로
  수렴하지만(같은 _id upsert라 두 번 색인해도 같다), 그래서 테스트·eval은
  `row_ids`로 자기 문서의 행만 처리한다. 개발계에서 워커를 띄운 채 eval을 돌리면 이 경합이 실재한다.
- 인라인(커밋 직후 즉시 처리) → 안 한다. 삭제·토글·FAQ 수정도 cron까지 최대 1분(재시도 포함
  1~5분) 뒤 검색에 반영된다. **제품 결정**: "문서 변경은 검색에 최대 1~5분 뒤 반영될 수 있다"로
  가이드한다. 그 사이 삭제·비공개 문서가 인용될 수 있음을 받아들인 것이다(답변 캐시는
  라우터가 즉시 무효화하므로 창은 새 검색에만 열린다). 좁히려면 워커 폴링 루프(10초)나
  "지금 반영" 수동 트리거를 얹으면 되고, 둘 다 이 구조 위에 그대로 붙는다.
- 백오프 → 매 회차 재시도. 엔진이 죽어 있으면 매분 실패가 쌓일 뿐 해롭지 않다.
- 크래시 루프 방어(잡을 때 attempts 증가) → 파서가 프로세스를 죽이는 파일은 사람이 본다.

## 멱등성이 전제다

모든 연산이 멱등이다: 색인은 `_id=chunk_id` upsert(chunk_id가 결정적 — opensearch.chunk_os_id),
삭제는 없는 것을 지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이다. 그래서 겹쳐 두 번 처리해도
결과가 같고, "처리했는데 done 찍기 전에 죽는" 경우도 다음 회차가 한 번 더 할 뿐이다.
**at-least-once**를 택한 것이다 — exactly-once는 분산 트랜잭션이 필요하고 멱등 연산에서는 값이 없다.

## payload에는 id만 싣는다

값을 싣지 않는다. 처리 시점에 PG의 **최신값**을 읽으므로 그 사이 토글이 여러 번 바뀌었어도
마지막 상태가 반영되고, 같은 문서에 META 행이 여럿 쌓여 있어도 전부 같은 최종값을 쓴다.
"""
import asyncio
import logging

from sqlalchemy import select, update

from database import AsyncSessionLocal
from rag import opensearch
from rag.metrics import SEARCH_INDEX_SYNC_TOTAL
from rag.models import Document, SearchIndexOutbox

logger = logging.getLogger(__name__)

# 연산 어휘 — payload 형태가 각각 다르다.
INDEX_DOCUMENT = 'index_document'      # {"document_id": int}   파싱·임베딩·색인·ready 승격 (인제스션 전체)
DROP_DOCUMENTS = 'drop_documents'      # {"document_ids": [int]}
META_DOCUMENTS = 'meta_documents'      # {"document_ids": [int]} 검색가능·폴더 부분 갱신
INDEX_FAQ = 'index_faq'                # {"faq_id": int}
DROP_FAQS = 'drop_faqs'                # {"faq_ids": [int]}
META_FAQS = 'meta_faqs'                # {"faq_ids": [int]}

PENDING, DONE, FAILED = 'pending', 'done', 'failed'
MAX_ATTEMPTS = 5        # 이걸 넘으면 failed 확정 — 사람이 본다. 테스트는 monkeypatch로 1을 쓴다.
BATCH = 50              # 한 회차가 처리할 최대 행. 다음 회차가 이어받으니 크지 않아도 된다.

_drain_lock = asyncio.Lock()   # cron 겹침 방지 — 단일 워커 전제(모듈 docstring)


def enqueue(session, tenant_id: str, op: str, **payload):
    """대기열 등재 — **commit하지 않는다.** 호출부의 트랜잭션에 얹히는 것이 요점이다."""
    row = SearchIndexOutbox(tenant_id=tenant_id, op=op, payload=payload, status=PENDING)
    session.add(row)
    return row


async def mark_done(session, row_id: int) -> None:
    """행을 done으로 — INDEX_DOCUMENT 핸들러가 ready 승격과 **같은 커밋**에 넣기 위해 쓴다."""
    await session.execute(
        update(SearchIndexOutbox).where(SearchIndexOutbox.id == row_id).values(status=DONE))


async def _apply(session, op: str, payload: dict, row_id: int) -> None:
    """행 하나를 실제로 반영한다. 실패는 예외로 올린다 — 횟수·failed 판정은 drain이 한다."""
    if op == INDEX_DOCUMENT:
        from rag.documents import index_pending_document   # 지연 import — documents가 이 모듈을 import
        await index_pending_document(payload['document_id'], outbox_row_id=row_id)
    elif op == DROP_DOCUMENTS:
        await opensearch.drop_documents_now(payload['document_ids'])
    elif op == META_DOCUMENTS:
        await opensearch.sync_meta_documents_now(session, payload['document_ids'])
    elif op == INDEX_FAQ:
        await opensearch.index_faq_chunks(session, payload['faq_id'])
    elif op == DROP_FAQS:
        await opensearch.drop_faqs_now(payload['faq_ids'])
    elif op == META_FAQS:
        await opensearch.sync_meta_faqs_now(session, payload['faq_ids'])
    else:
        raise ValueError(f'알 수 없는 outbox op: {op!r}')


async def drain(session, limit: int = BATCH, *, row_ids: list[int] | None = None) -> dict:
    """pending 행을 id 순으로 처리한다. 반환 {'done': n, 'failed': n}.

    row_ids — 이 행들만 처리한다. 테스트·eval 스크립트용: 개발계 DB를 여러 세션이 공유하므로
    전체 drain은 남의 pending 업로드까지 처리해 버린다. 운영 cron은 넘기지 않는다.

    한 행의 실패가 다음 행을 막지 않는다 — 실패한 행만 attempts를 올리고 계속 간다.
    행 단위로 commit한다: 50건 중 30번째에서 프로세스가 죽어도 29건은 done으로 남는다.

    INDEX_DOCUMENT는 핸들러가 자기 트랜잭션에서 done을 찍는다(ready 승격과 같은 커밋 —
    "ready ≡ 색인됨"을 커밋 단위로 보장). 그래서 여기서는 그 op의 status를 건드리지 않는다.
    """
    stmt = (select(SearchIndexOutbox.id, SearchIndexOutbox.op, SearchIndexOutbox.payload)
            .where(SearchIndexOutbox.status == PENDING)
            .order_by(SearchIndexOutbox.id).limit(limit))
    if row_ids is not None:
        stmt = stmt.where(SearchIndexOutbox.id.in_(list(row_ids)))
    rows = (await session.execute(stmt)).all()

    done = failed = 0
    for row_id, op, payload in rows:           # 값으로 들고 간다 — rollback이 ORM 객체를 만료시킨다
        try:
            await _apply(session, op, payload or {}, row_id)
            if op != INDEX_DOCUMENT:
                await mark_done(session, row_id)
            await session.commit()
            SEARCH_INDEX_SYNC_TOTAL.labels(op=op, result='ok').inc()
            done += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:                     # noqa: BLE001 — 행 단위로 격리한다
            await session.rollback()               # 핸들러가 이 세션을 더럽혔을 수 있다
            err = str(e)[:500]
            attempts = (await session.execute(
                update(SearchIndexOutbox)
                .where(SearchIndexOutbox.id == row_id)
                .values(attempts=SearchIndexOutbox.attempts + 1, last_error=err)
                .returning(SearchIndexOutbox.attempts)
            )).scalar()
            terminal = attempts >= MAX_ATTEMPTS
            if terminal:
                await session.execute(
                    update(SearchIndexOutbox).where(SearchIndexOutbox.id == row_id)
                    .values(status=FAILED))
                if op == INDEX_DOCUMENT:
                    # 문서도 failed — 사용자가 FE에서 실패를 보고 재업로드할 수 있게.
                    # pending일 때만: 그새 삭제됐거나 다른 경로로 끝난 문서는 건드리지 않는다.
                    await session.execute(
                        update(Document)
                        .where(Document.id == (payload or {}).get('document_id'))
                        .where(Document.status == 'pending')
                        .values(status='failed', status_reason=f'색인 {attempts}회 실패: {err}'[:500]))
            await session.commit()
            SEARCH_INDEX_SYNC_TOTAL.labels(op=op, result='error').inc()
            failed += 1
            logger.warning('outbox 반영 실패 (op=%s id=%s attempts=%d%s): %s',
                           op, row_id, attempts, ' → failed 확정' if terminal else '', e)
    return {'done': done, 'failed': failed}


async def drain_once(limit: int = BATCH, *, row_ids: list[int] | None = None) -> dict:
    """자기 세션으로 한 회차 처리. 워커 cron·테스트·eval 스크립트가 부른다.

    이전 회차가 아직 돌고 있으면(대량 인제스션으로 1분을 넘긴 경우) 즉시 빠진다 — 단일 워커라
    프로세스 안 잠금으로 충분하다. 놓친 회차는 다음 분이 잡는다.
    """
    if _drain_lock.locked():
        return {'done': 0, 'failed': 0, 'skipped': True}
    async with _drain_lock:
        async with AsyncSessionLocal() as session:
            return await drain(session, limit, row_ids=row_ids)


async def pending_row_ids(session, *, document_id: int | None = None,
                          faq_id: int | None = None) -> list[int]:
    """한 문서 또는 한 FAQ에 걸린 pending 행 id — 테스트·eval이 row_ids로 넘길 값.

    document_id: 그 문서를 가리키는 INDEX_DOCUMENT(payload.document_id)·DROP/META_DOCUMENTS
    (payload.document_ids 배열) 행 전부. faq_id: INDEX_FAQ(payload.faq_id)·META/DROP_FAQS
    (payload.faq_ids 배열) 행 전부 — 라우터 한 요청이 어느 op를 남겼든 같은 호출로 반영할 수 있게.
    둘 중 하나만 넘긴다.
    """
    from sqlalchemy import or_
    assert (document_id is None) != (faq_id is None), 'document_id 또는 faq_id 하나만'
    stmt = (select(SearchIndexOutbox.id)
            .where(SearchIndexOutbox.status == PENDING)
            .order_by(SearchIndexOutbox.id))
    if document_id is not None:
        stmt = stmt.where(or_(
            SearchIndexOutbox.payload['document_id'].as_integer() == document_id,
            SearchIndexOutbox.payload['document_ids'].contains([document_id]),   # JSONB @> '[id]'
        ))
    else:
        stmt = stmt.where(or_(
            SearchIndexOutbox.payload['faq_id'].as_integer() == faq_id,
            SearchIndexOutbox.payload['faq_ids'].contains([faq_id]),
        ))
    return list((await session.execute(stmt)).scalars().all())
