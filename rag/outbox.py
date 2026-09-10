"""외부 검색 색인 반영 대기열 (#139 outbox) — 어휘·처리 규약의 **정의점**.

## 왜 필요한가

색인이 서빙의 정본인 구성에서(rag/opensearch.py 상단) "반영 누락"은 곧 낡은 답변이다.
커밋 **후** 직접 호출하는 방식은 그 사이에 프로세스가 죽거나 엔진이 죽으면 색인이 조용히
낡는다 — 상담 답변은 이미 생성돼 고객에게 나간 뒤에야 알게 되므로 되돌릴 수 없다.

outbox는 그 창을 닫는다: **할 일을 PG 변경과 같은 트랜잭션에 남긴다.** 커밋이 성공하면
일이 반드시 기록돼 있고, 롤백되면 일도 함께 사라진다. 커밋과 대기열 등재가 원자적이라는
것이 이 패턴의 전부다.

## 처리 주체 둘

- **인라인 드레인**: 커밋 직후 호출부가 바로 처리한다(`drain_now`). 반영 지연을 없앤다.
- **워커 cron**: 1분 주기로 남은 것을 처리한다(rag/worker.py). 인라인이 실패하거나
  프로세스가 죽어도 반영을 **보장**하는 쪽이다.

성공하면 행을 지운다 — 남아 있는 행이 곧 '아직 반영 안 된 일'이다. 실패는 attempts에
쌓이고 next_attempt_at으로 지수 백오프한다(1분·2분·4분… 상한 1시간).

## 멱등성이 전제다

모든 연산이 멱등이다: 색인은 `_id=chunk_id` upsert, 삭제는 없는 것을 지워도 무해, 메타
갱신은 같은 값을 덮어쓸 뿐이다. 그래서 인라인과 cron이 겹쳐 두 번 처리해도 결과가 같고,
"처리했는데 행 삭제 전에 죽는" 경우도 다음 드레인이 한 번 더 할 뿐 문제가 없다.
**at-least-once**를 택한 것이다 — exactly-once를 만들려면 분산 트랜잭션이 필요하고,
멱등 연산에서는 값이 없다.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, select

from rag import opensearch
from rag.models import SearchIndexOutbox

logger = logging.getLogger(__name__)

# 연산 어휘 — payload 형태가 각각 다르다.
INDEX_DOCUMENT = 'index_document'      # {"document_id": int}  청크 재색인(본문·벡터 포함)
DROP_DOCUMENTS = 'drop_documents'      # {"document_ids": [int]}
META_DOCUMENTS = 'meta_documents'      # {"document_ids": [int]}  검색가능·폴더 부분 갱신
INDEX_FAQ = 'index_faq'                # {"faq_id": int}
DROP_FAQS = 'drop_faqs'                # {"faq_ids": [int]}
META_FAQS = 'meta_faqs'                # {"faq_ids": [int]}

BATCH = 50              # 한 번의 드레인이 처리할 최대 행
MAX_BACKOFF_MIN = 60


def enqueue(session, tenant_id: str, op: str, **payload) -> None:
    """대기열 등재 — **commit하지 않는다.** 호출부의 트랜잭션에 얹히는 것이 요점이다.

    `session.add`만 하므로 네트워크도 await도 없다. 라우터·워커가 자기 커밋에 함께 실어
    보내면 되고, 그 커밋이 실패하면 이 행도 함께 사라진다.
    """
    if not opensearch.enabled():
        return
    session.add(SearchIndexOutbox(tenant_id=tenant_id, op=op, payload=payload))


async def _apply(session, row) -> None:
    """행 하나를 실제로 반영한다. 실패는 예외로 올린다(호출부가 백오프를 건다)."""
    p = row.payload or {}
    if row.op == INDEX_DOCUMENT:
        await opensearch.index_document_chunks(session, p['document_id'])
    elif row.op == DROP_DOCUMENTS:
        await opensearch.drop_documents_now(p['document_ids'])
    elif row.op == META_DOCUMENTS:
        await opensearch.sync_meta_documents_now(session, p['document_ids'])
    elif row.op == INDEX_FAQ:
        await opensearch.index_faq_chunks(session, p['faq_id'])
    elif row.op == DROP_FAQS:
        await opensearch.drop_faqs_now(p['faq_ids'])
    elif row.op == META_FAQS:
        await opensearch.sync_meta_faqs_now(session, p['faq_ids'])
    else:
        raise ValueError(f'알 수 없는 outbox op: {row.op!r}')


async def drain(session, limit: int = BATCH) -> dict:
    """처리할 때가 된 행을 오래된 순으로 처리한다. 반환 {'done': n, 'failed': n}.

    워커 cron과 인라인이 같이 부른다. `SKIP LOCKED`로 잠긴 행을 건너뛰어 둘이 동시에
    돌아도 같은 행을 두 번 잡지 않는다(겹쳐도 멱등이라 안전하지만, 헛일을 줄인다).

    한 행의 실패가 다음 행을 막지 않는다 — 실패는 그 행에만 백오프를 걸고 계속 간다.
    """
    from rag.metrics import SEARCH_INDEX_SYNC_TOTAL

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    rows = (await session.execute(
        select(SearchIndexOutbox)
        .where(SearchIndexOutbox.next_attempt_at <= now)
        .order_by(SearchIndexOutbox.id)
        .limit(limit)
        .with_for_update(skip_locked=True)
    )).scalars().all()

    done = failed = 0
    for row in rows:
        try:
            await _apply(session, row)
            await session.execute(
                delete(SearchIndexOutbox).where(SearchIndexOutbox.id == row.id))
            SEARCH_INDEX_SYNC_TOTAL.labels(op=row.op, result='ok').inc()
            done += 1
        except Exception as e:                     # noqa: BLE001 — 행 단위로 격리한다
            row.attempts += 1
            row.last_error = str(e)[:500]
            backoff = min(2 ** (row.attempts - 1), MAX_BACKOFF_MIN)
            row.next_attempt_at = now + timedelta(minutes=backoff)
            SEARCH_INDEX_SYNC_TOTAL.labels(op=row.op, result='error').inc()
            failed += 1
            logger.warning('outbox 반영 실패 (op=%s id=%s attempts=%d, %d분 뒤 재시도): %s',
                           row.op, row.id, row.attempts, backoff, e)
    await session.commit()
    return {'done': done, 'failed': failed}


async def flush_document_index(document_id: int, superseded_ids, embeddings) -> None:
    """인제스션 전용 빠른 길 — 방금 계산한 임베딩으로 바로 색인하고 그 대기열 행을 지운다.

    느린 길(워커 cron)은 PG에서 벡터를 읽는데, PG가 벡터를 안 드는 구성에선 그게
    **재임베딩**이다. 인제스션은 벡터를 손에 들고 있으니 그 낭비를 피한다.

    **예외를 올리지 않는다.** 실패하면 대기열 행이 그대로 남아 워커가 (필요하면 재임베딩으로)
    반영을 보장한다 — 빠른 길이 막혀도 정합은 유지된다. 호출부가 인제스션의 try 안이라,
    여기서 예외가 새면 이미 커밋된 인제스션이 failed로 뒤집힌다.
    """
    if not opensearch.enabled():
        return
    from database import AsyncSessionLocal
    try:
        await opensearch.index_document_with_vectors(
            document_id, superseded_ids, embeddings)
        async with AsyncSessionLocal() as session:
            await session.execute(delete(SearchIndexOutbox).where(
                SearchIndexOutbox.op.in_([INDEX_DOCUMENT, DROP_DOCUMENTS]),
                SearchIndexOutbox.payload['document_id'].astext == str(document_id)))
            await session.commit()
    except asyncio.CancelledError:
        raise
    except Exception as e:                          # noqa: BLE001 — 의도된 삼킴(위 docstring)
        logger.warning('인제스션 빠른 색인 실패 — 워커가 재시도한다 (document_id=%s): %s',
                       document_id, e)


async def drain_now(session) -> None:
    """커밋 직후의 인라인 드레인 — 반영 지연을 없애는 빠른 길.

    **이 함수는 어떤 경우에도 예외를 올리지 않는다.** 호출부는 이미 커밋을 마친 라우터·
    워커이고, 여기서 터뜨리면 색인 장애가 곧 업로드·수정 장애가 된다. 실패해도 행이
    대기열에 남아 워커 cron이 반영을 보장한다 — 그것이 outbox를 둔 이유다.
    """
    if not opensearch.enabled():
        return
    try:
        await drain(session)
    except asyncio.CancelledError:
        raise                                       # 취소는 삼키지 않는다
    except Exception as e:                          # noqa: BLE001 — 의도된 삼킴(위 docstring)
        logger.warning('인라인 드레인 실패 — 워커 cron이 재시도한다: %s', e)
