"""색인 작업 대기열 — 트랜잭셔널 outbox (#139). 어휘·처리 규약의 **정의점**.

## 요점 하나

**색인에 할 일을 PG 변경과 같은 트랜잭션에 남긴다.** 업로드·삭제·토글·FAQ 변경이 커밋되면
그에 따른 색인 작업이 반드시 기록돼 있고, 롤백되면 그 일도 함께 사라진다. 커밋과 등재가
원자적이라는 것 — 그게 이 패턴의 전부다. 등재는 `session.add` 한 줄이라 네트워크도 await도 없다.

이전 방식(커밋 후 직접 호출, arq 잡 등록)은 그 사이에 프로세스가 죽거나 Redis가 순단하면
색인이 조용히 낡았다. 색인이 서빙의 정본인 구성(rag/os_search.py의 fetch_chunk_map)에서 그 누락은 곧
낡은 답변이고, 상담 답변은 고객에게 나간 뒤에야 알게 되므로 되돌릴 수 없다.

## 처리 — 단일 워커, cron 1분, 폴링

워커 cron(rag/worker.py)이 1분마다 `drain_once`를 부른다. pending 행을 **id 순으로** 하나씩
처리하고 성공하면 done. 실패는 두 갈래다(#185) — **결정적** 실패(파싱 오류·빈 파일·지원 안 하는
형식처럼 다시 돌려도 같은 것)는 1회로 failed 확정, **일시** 실패(엔진·임베딩 서버 연결·타임아웃·5xx)는
attempts에 쌓이며 `next_attempt_at`을 지수 백오프로 미룬다(1·2·4…60분 상한, 총 창 약 19시간).
MAX_ATTEMPTS까지 못 끝내면 failed 확정. 어느 쪽이든 failed가 되는 순간 `_on_failed`가 불린다 —
지금은 ERROR 로그와 확정 카운터까지, 알람 연동은 그 함수만 채우면 된다.
INDEX_DOCUMENT의 failed는 문서도 failed로 찍고(FE가 그걸로 실패를 본다) **잔여 청크 DROP을 같은
커밋에 등재한다**(#184 — ② 엔진 쓰기 뒤 ③ 커밋만 실패하면 검색 가능한 청크가 남기 때문).
행은 지우지 않는다: done/failed 행이 이력이자 관측 지점이다. 보존 정리는 나중에 스케줄러로.

일시/결정적 분류의 정의점은 이 파일의 `is_transient` 하나다. 기본은 **결정적**이다 — 목록에 없는
예외를 일시로 봐 하루를 기다리게 하는 것보다, 결정적으로 봐 바로 failed로 보이는 쪽이 사용자에게
싸다(재업로드 한 번). 알려진 오분류: bulk 부분 실패(`RuntimeError`)에 섞인 `rejected_execution`은
일시인데 결정적으로 잡힌다 — 단일 워커라 드물어 감수한다.

**의도적으로 하지 않은 것**과 그 트리거:
- 워커 여러 대 → 지금은 단일 워커라 프로세스 안 `asyncio.Lock` 하나로 cron 겹침(1분 넘게
  도는 회차)을 막는다. 워커를 늘리면 claim 컬럼(+ 고아 회수)이 필요하다.
  **다른 프로세스**의 drain(테스트 `ingest`, eval 스크립트가 워커 cron과 동시에)은 이 잠금이
  못 막는다 — SELECT에 행 잠금이 없어 같은 pending 행을 둘이 뽑을 수 있다. 결과는 멱등으로
  수렴하지만(같은 _id upsert라 두 번 색인해도 같다), 그래서 테스트·eval은
  `row_ids`로 자기 문서의 행만 처리한다. 개발계에서 워커를 띄운 채 eval을 돌리면 이 경합이 실재한다.
- 인라인(커밋 직후 즉시 처리) → 안 한다. 삭제·토글·FAQ 수정도 cron까지 최대 1분 뒤 검색에
  반영된다. **제품 결정**: "문서 변경은 검색에 최대 1분 뒤 반영되고, 검색 엔진 장애 중이면 복구 후
  최대 1시간(백오프 상한) 안에 따라잡는다"로 가이드한다. 그 사이 삭제·비공개 문서가 인용될 수 있음을 받아들인 것이다(답변 캐시는
  라우터가 즉시 무효화하므로 창은 새 검색에만 열린다). 좁히려면 워커 폴링 루프(10초)나
  "지금 반영" 수동 트리거를 얹으면 되고, 둘 다 이 구조 위에 그대로 붙는다.
- 크래시 루프 방어(잡을 때 attempts 증가) → 파서가 프로세스를 죽이는 파일은 사람이 본다.

## 처리 순서에 기대지 마라

id 순은 **성향이지 보장이 아니다**(#185). 백오프로 미뤄진 행은 뒤의 행보다 늦게 돌고, 행 단위 격리라
실패한 행 뒤의 행은 같은 회차에 먼저 끝난다. 그래서 모든 핸들러는 "이 행이 등재 순서대로 왔다"를
전제하지 않고 **처리 시점의 PG 상태**로 판단한다 — META는 최신값을 읽고, DROP은 살아 있는 문서를
건너뛰고(_apply), INDEX_DOCUMENT는 version 조건으로 구버전만 supersede한다(rag/documents.py ①).
새 op를 넣을 때도 같은 물음을 먼저 던져라: "이 행이 뒤에 등재된 행보다 늦게 돌면 무엇이 깨지나."

## 멱등성이 전제다

모든 연산이 멱등이다: 색인은 `_id=chunk_id` upsert(chunk_id가 결정적 — os_index.chunk_os_id),
삭제는 없는 것을 지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이다. 그래서 겹쳐 두 번 처리해도
결과가 같고, "처리했는데 done 찍기 전에 죽는" 경우도 다음 회차가 한 번 더 할 뿐이다.
**at-least-once**를 택한 것이다 — exactly-once는 분산 트랜잭션이 필요하고 멱등 연산에서는 값이 없다.

## payload에는 id만 싣는다

값을 싣지 않는다. 처리 시점에 PG의 **최신값**을 읽으므로 그 사이 토글이 여러 번 바뀌었어도
마지막 상태가 반영되고, 같은 문서에 META 행이 여럿 쌓여 있어도 전부 같은 최종값을 쓴다.
"""
import asyncio
import logging
from datetime import timedelta

import httpx
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import InterfaceError, OperationalError

from database import AsyncSessionLocal
from rag import os_client, os_index
from rag.metrics import INDEX_TOTAL, SEARCH_INDEX_FAILED_TOTAL, SEARCH_INDEX_SYNC_TOTAL, ext_label
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, SearchIndexOutbox

logger = logging.getLogger(__name__)

# 연산 어휘 — payload 형태가 각각 다르다.
INDEX_DOCUMENT = 'index_document'      # {"document_id": int}   파싱·임베딩·색인·ready 승격 (인제스션 전체)
DROP_DOCUMENTS = 'drop_documents'      # {"document_ids": [int]}
META_DOCUMENTS = 'meta_documents'      # {"document_ids": [int]} 검색가능·폴더 부분 갱신
INDEX_FAQ = 'index_faq'                # {"faq_id": int}
DROP_FAQS = 'drop_faqs'                # {"faq_ids": [int]}
META_FAQS = 'meta_faqs'                # {"faq_ids": [int]}

PENDING, DONE, FAILED = 'pending', 'done', 'failed'
# 일시 실패의 재시도 상한(#185). 1·2·4·8·16·32분 뒤 60분 고정이라 24회면 총 약 19시간 —
# "엔진이 반나절 죽었다 살아나도 사람 손 없이 따라잡는다"가 목표 눈금이다. 결정적 실패는 이 수를 안 본다.
MAX_ATTEMPTS = 24
BACKOFF_CAP_MINUTES = 60   # 복구 뒤 늦어도 이 안에 반영된다 — 눈금을 키우면 그만큼 낡은 채로 기다린다
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
        # 살아 있는 문서는 건너뛴다(#184 안전판). failed 확정이 등재한 DROP이 백오프로 미뤄진 사이
        # 사용자가 같은 파일을 재업로드하면 그 failed 행이 되살아나(#161) INDEX가 먼저 끝날 수 있다 —
        # 그 뒤에 도는 DROP이 새 청크를 지우면 ready인데 검색에 없는 문서가 된다. payload엔 id만
        # 싣고 처리 시점의 PG를 읽는다는 규약 그대로: 지금 deleted·failed(또는 행 없음)인 것만 지운다.
        ids = list(payload['document_ids'])
        alive = set((await session.execute(
            select(Document.id).where(Document.id.in_(ids))
            .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES)))).scalars().all())
        if alive:
            logger.info('DROP_DOCUMENTS 건너뜀 — 되살아난 문서 %s', sorted(alive))
        if targets := [i for i in ids if i not in alive]:
            await os_index.drop_documents_now(targets)
    elif op == META_DOCUMENTS:
        await os_index.sync_meta_documents_now(session, payload['document_ids'])
    elif op == INDEX_FAQ:
        await os_index.index_faq_chunks(session, payload['faq_id'])
    elif op == DROP_FAQS:
        await os_index.drop_faqs_now(payload['faq_ids'])
    elif op == META_FAQS:
        await os_index.sync_meta_faqs_now(session, payload['faq_ids'])
    else:
        raise ValueError(f'알 수 없는 outbox op: {op!r}')


def is_transient(exc: BaseException) -> bool:
    """실패가 **일시적**인가 — 다시 시도하면 될 수 있는 종류인가. 분류의 정의점(#185).

    일시 = 상대(검색 엔진·임베딩 서버)에 닿지 못했거나 상대가 바빴다는 신호만:
      - opensearchpy: ConnectionError(ConnectionTimeout·SSLError 포함), TransportError 중 429·502·503·504
        또는 상태코드 없음('N/A' — 연결 계층 실패)
      - httpx(임베딩 TEI 호출): TransportError(ConnectError·Timeout·RemoteProtocolError),
        HTTPStatusError 중 429·5xx
      - builtin ConnectionError(ConnectionRefusedError 등 OSError 하위)·TimeoutError(asyncio.TimeoutError와 동일 클래스)
      - PG 자체(SQLAlchemy가 asyncpg 예외를 감싼 것): OperationalError·InterfaceError — 핸들러가 같은 세션으로
        PG를 읽고 쓰는 도중 커넥션이 끊긴 경우. 분류 뒤의 attempts UPDATE도 같은 DB라 함께 실패할 수
        있는데, 그때는 예외가 drain 밖으로 나가 이 회차가 끊기고 행은 pending 그대로 남는다 — 무해.
    그 밖은 전부 **결정적** — 파싱 ValueError, docling RuntimeError, blob 없음(FileNotFoundError),
    프로그래밍 오류. 백 번 돌려도 같으므로 바로 failed로 보이는 것이 사용자에게 싸다(모듈 docstring).
    """
    if isinstance(exc, (ConnectionError, TimeoutError, OperationalError, InterfaceError)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    # opensearchpy는 함수 안에서 import한다 — rag/ 어느 모듈도 톱레벨에 들이지 않는 규율(os_client.client
    # 참조, AGENTS.md)을 따른다. 순환 회피가 아니라서 AGENTS.md의 지연 import 목록엔 따로 적혀 있다.
    from opensearchpy import ConnectionError as OsConnectionError, TransportError
    if isinstance(exc, OsConnectionError):
        return True
    if isinstance(exc, TransportError):
        return exc.status_code in ('N/A', 429, 502, 503, 504)
    return False


def backoff_delay(attempts: int) -> timedelta:
    """attempts번째 실패 뒤 다음 시도까지 — 1·2·4·8·16·32분, 그 뒤 BACKOFF_CAP_MINUTES 고정."""
    return timedelta(minutes=min(2 ** (attempts - 1), BACKOFF_CAP_MINUTES))


def _on_failed(op: str, row_id: int, err: str) -> None:
    """failed 확정 지점 하나 — 알람을 붙일 때 여기만 채운다(#185). 지금은 ERROR 로그와 확정 카운터.

    확정은 재시도와 다르다: 이 시점부터 그 행의 PG↔엔진 불일치는 사람이 고칠 때까지 영구다
    (META·DROP·FAQ는 Document/Faq에 표시도 남지 않는다 — 이 로그·카운터가 유일한 단서).
    """
    SEARCH_INDEX_FAILED_TOTAL.labels(op=op).inc()
    logger.error('outbox failed 확정 (op=%s id=%s): %s', op, row_id, err)


def due_pending_stmt(limit: int = BATCH, *, row_ids: list[int] | None = None):
    """drain이 이번 회차에 집을 행 — pending이고 **시도 시각이 됐거나 정해지지 않은** 것, id 순.

    row_ids를 주면 그 행들만 고르고 **백오프와 limit을 무시한다** — "지정한 행을 지금 전부 처리한다"가
    그 인자의 뜻이다(테스트 `ingest`·eval 스크립트는 자기 문서의 행을 즉시 반영하려고 부른다).
    """
    stmt = (select(SearchIndexOutbox.id, SearchIndexOutbox.tenant_id, SearchIndexOutbox.op,
                   SearchIndexOutbox.payload)
            .where(SearchIndexOutbox.status == PENDING)
            .order_by(SearchIndexOutbox.id))
    if row_ids is not None:
        return stmt.where(SearchIndexOutbox.id.in_(list(row_ids)))
    return (stmt.where(or_(SearchIndexOutbox.next_attempt_at.is_(None),
                           SearchIndexOutbox.next_attempt_at <= func.now()))
            .limit(limit))


async def drain(session, limit: int = BATCH, *, row_ids: list[int] | None = None) -> dict:
    """pending 행을 id 순으로 처리한다. 반환 {'done': n, 'failed': n} — failed는 "이번 회차에 실패한
    행 수"다(백오프로 물러난 것과 확정된 것을 합친다).

    row_ids — 이 행들만 처리한다(백오프·limit 무시 — due_pending_stmt). 테스트·eval 스크립트용: 개발계 DB를
    여러 세션이 공유하므로 전체 drain은 남의 pending 업로드까지 처리해 버린다. 운영 cron은 넘기지 않는다.

    한 행의 실패가 다음 행을 막지 않는다 — 실패한 행만 attempts를 올리고 계속 간다.
    행 단위로 commit한다: 50건 중 30번째에서 프로세스가 죽어도 29건은 done으로 남는다.

    INDEX_DOCUMENT는 핸들러가 자기 트랜잭션에서 done을 찍는다(ready 승격과 같은 커밋 —
    "ready ≡ 색인됨"을 커밋 단위로 보장). 그래서 여기서는 그 op의 status를 건드리지 않는다.
    """
    rows = (await session.execute(due_pending_stmt(limit, row_ids=row_ids))).all()

    done = failed = 0
    for row_id, tenant_id, op, payload in rows:   # 값으로 들고 간다 — rollback이 ORM 객체를 만료시킨다
        payload = payload or {}
        try:
            await _apply(session, op, payload, row_id)
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
            # 결정적 실패는 횟수를 안 본다 — 다시 돌려도 같은 결과라 기다리는 것이 사용자에게 손해다.
            terminal = not is_transient(e) or attempts >= MAX_ATTEMPTS
            if terminal:
                await session.execute(
                    update(SearchIndexOutbox).where(SearchIndexOutbox.id == row_id)
                    .values(status=FAILED))
                if op == INDEX_DOCUMENT:
                    # 문서도 failed — 사용자가 FE에서 실패를 보고 재업로드할 수 있게.
                    # pending일 때만: 그새 삭제됐거나 다른 경로로 끝난 문서는 건드리지 않는다.
                    await session.execute(
                        update(Document)
                        .where(Document.id == payload.get('document_id'))
                        .where(Document.status == 'pending')
                        .values(status='failed', status_reason=f'색인 {attempts}회 실패: {err}'[:500]))
                    # 잔여 청크 DROP을 같은 커밋에(#184). index_pending_document는 ② 엔진 쓰기 뒤 ③ 커밋이라
                    # ③만 실패해 failed로 굳으면 searchable=True 청크가 남아 실패한 문서가 검색에 잡힌다.
                    # "failed = 엔진에 청크 없음"을 여기서 참으로 만든다 — 생길 때 지운다. 그 failed를
                    # 나중에 내릴 때 지우는 handle_upload의 others DROP(#161)은 이것의 이중 안전판이다.
                    # 청크가 없었으면(②보다 앞서 죽음) 0건 삭제 — 멱등이라 무해.
                    enqueue(session, tenant_id, DROP_DOCUMENTS, document_ids=[payload.get('document_id')])
            else:
                await session.execute(
                    update(SearchIndexOutbox).where(SearchIndexOutbox.id == row_id)
                    .values(next_attempt_at=func.now() + backoff_delay(attempts)))
            await session.commit()
            SEARCH_INDEX_SYNC_TOTAL.labels(op=op, result='error').inc()
            if terminal:
                _on_failed(op, row_id, err)
            if op == INDEX_DOCUMENT:
                # 문서 단위 결과를 따로 센다 (#151). 위 카운터는 재시도 단위라 실패율의 분모가
                # 못 된다. ext는 실패 경로에서만 한 번 더 읽는다 — 드문 경로라 비용이 무의미하고,
                # "PDF만 실패한다" 같은 패턴은 이 라벨이 없으면 보이지 않는다.
                fname = (await session.execute(
                    select(Document.filename)
                    .where(Document.id == payload.get('document_id')))).scalar()
                INDEX_TOTAL.labels(ext=ext_label(fname or ''),
                                   result='failed' if terminal else 'retry').inc()
            failed += 1
            logger.warning('outbox 반영 실패 (op=%s id=%s attempts=%d%s): %s',
                           op, row_id, attempts,
                           ' → failed 확정' if terminal else f' → {backoff_delay(attempts)} 뒤 재시도', e)
    return {'done': done, 'failed': failed}


async def drain_once(limit: int = BATCH, *, row_ids: list[int] | None = None) -> dict:
    """자기 세션으로 한 회차 처리. 워커 cron·테스트·eval 스크립트가 부른다.

    이전 회차가 아직 돌고 있으면(대량 인제스션으로 1분을 넘긴 경우) 즉시 빠진다 — 단일 워커라
    프로세스 안 잠금으로 충분하다. 놓친 회차는 다음 분이 잡는다.
    """
    if _drain_lock.locked():
        return {'done': 0, 'failed': 0, 'skipped': True}
    # 기동 시 엔진에 못 붙었을 수 있다(ensure_index_soft) — 색인 전에 인덱스를 다시 보장한다.
    # 없는 인덱스에 bulk하면 동적 매핑으로 굳으므로, 실패하면 이 회차는 건너뛴다(행은 pending 그대로).
    if not await os_client.ensure_index_soft():
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
