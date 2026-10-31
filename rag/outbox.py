"""색인 작업 대기열 — 트랜잭셔널 outbox. 어휘·처리 규약의 **정의점**.

**색인에 할 일을 PG 변경과 같은 트랜잭션에 남긴다** — 커밋되면 색인 작업이 반드시 기록되고, 롤백되면
함께 사라진다. 등재는 `session.add` 한 줄이라 commit하지 않는다.

처리: 워커 cron이 1분마다 `drain_once`를 부르고 pending 행을 id 순으로 처리한다. **결정적** 실패는
1회로 failed 확정, **일시** 실패는 지수 백오프로 MAX_ATTEMPTS까지 재시도한다(분류는 `is_transient`,
목록에 없는 예외는 결정적). INDEX_DOCUMENT가 failed로 확정되면 문서도 failed로 찍고 잔여 청크 DROP을
같은 커밋에 등재한다. 행은 지우지 않는다 — done·failed 행이 이력이자 관측 지점이다.

규약:
- 처리 순서에 기대지 마라 — id 순은 성향일 뿐이다(백오프·행 단위 격리). 핸들러는 **처리 시점의 PG
  상태**로 판단한다. 새 op도 "이 행이 뒤 행보다 늦게 돌면 무엇이 깨지나"를 먼저 따져라.
- 모든 연산이 멱등이다(at-least-once) — 두 번 처리해도 결과가 같다.
- payload에는 id만 싣는다 — 처리 시점에 최신값을 읽는다.
- 단일 워커 전제 — 프로세스 안 `asyncio.Lock`으로 cron 겹침만 막는다. 다른 프로세스의 drain과는 같은
  행을 둘이 뽑을 수 있어(멱등으로 수렴), 테스트·eval은 `row_ids`로 자기 행만 처리한다.
- 반영 지연은 최대 1분(엔진 장애 시 복구 후 백오프 상한까지) — 그 사이 삭제 문서가 새 검색에 인용될 수
  있음은 제품이 받아들인 범위다.
"""

import asyncio
import logging
import time
from datetime import timedelta
from typing import NamedTuple

import httpx
import openai
from sqlalchemy import func, or_, select, update
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import AsyncSessionLocal
from rag import os_client, os_index
from rag.metrics import INDEX_TOTAL, SEARCH_INDEX_FAILED_TOTAL, SEARCH_INDEX_SYNC_TOTAL, ext_label
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, SearchIndexOutbox

logger = logging.getLogger(__name__)

# 연산 어휘 — payload 형태가 각각 다르다.
INDEX_DOCUMENT = 'index_document'  # {"document_id": int}   파싱·임베딩·색인·ready 승격 (인제스션 전체)
DROP_DOCUMENTS = 'drop_documents'  # {"document_ids": [int]}
META_DOCUMENTS = 'meta_documents'  # {"document_ids": [int]} 검색가능·폴더 부분 갱신
INDEX_FAQ = 'index_faq'  # {"faq_id": int}
DROP_FAQS = 'drop_faqs'  # {"faq_ids": [int]}
META_FAQS = 'meta_faqs'  # {"faq_ids": [int]}

PENDING, DONE, FAILED = 'pending', 'done', 'failed'
# 일시 실패의 재시도 상한 — 백오프와 합쳐 총 약 19시간, 엔진이 반나절 죽어도 사람 손 없이 따라잡는다.
MAX_ATTEMPTS = 24
BACKOFF_CAP_MINUTES = 60  # 복구 뒤 늦어도 이 안에 반영된다 — 눈금을 키우면 그만큼 낡은 채로 기다린다
BATCH = 50  # 한 회차가 처리할 최대 행. 다음 회차가 이어받으니 크지 않아도 된다.
# 회차 예산 — arq job_timeout(rag/worker.py)이 쓰는 **회차 전체**의 상한. 넘기면 arq가 회차를 취소해 처리 중
# 문서가 헛돌므로, 행을 시작하기 전에 최악(변환 마감 + 꼬리)이 예산 안인지 본다(_fits_round).
INDEX_TAIL_SECONDS = 300  # 변환 뒤 임베딩·엔진 쓰기·커밋의 상한(타임아웃 설정에서 계산)
ROUND_SLACK_SECONDS = 180  # 첫 행 뒤에도 작은 문서가 이어 흐를 여유
# 마감에서 계산한다 — 고정값이면 마감을 올렸을 때 첫 행조차 예산에 못 들어가 대기열이 조용히 멈춘다.
DRAIN_ROUND_SECONDS = int(settings.docling_document_timeout_seconds) + INDEX_TAIL_SECONDS + ROUND_SLACK_SECONDS

_drain_lock = asyncio.Lock()  # cron 겹침 방지 — 단일 워커 전제


def enqueue(session, tenant_id: str, op: str, **payload):
    """대기열 등재 — **commit하지 않는다.** 호출부의 트랜잭션에 얹히는 것이 요점이다."""
    row = SearchIndexOutbox(tenant_id=tenant_id, op=op, payload=payload, status=PENDING)
    session.add(row)
    return row


async def _set_row(session: AsyncSession, row_id: int, **values: object) -> None:
    await session.execute(update(SearchIndexOutbox).where(SearchIndexOutbox.id == row_id).values(**values))


async def mark_done(session, row_id: int) -> None:
    """행을 done으로 — INDEX_DOCUMENT 핸들러가 ready 승격과 **같은 커밋**에 넣기 위해 쓴다."""
    await _set_row(session, row_id, status=DONE)


async def _apply(session, op: str, payload: dict, row_id: int) -> None:
    """행 하나를 실제로 반영한다. 실패는 예외로 올린다 — 횟수·failed 판정은 drain이 한다."""
    if op == INDEX_DOCUMENT:
        from rag.ingestion import index_pending_document  # 지연 import — ingestion이 이 모듈을 import

        await index_pending_document(payload['document_id'], outbox_row_id=row_id)
    elif op == DROP_DOCUMENTS:
        # 처리 시점에 살아 있는 문서는 건너뛴다 — 미뤄진 DROP이 그새 되살아난 문서의 새 청크를 지우지 않게.
        ids = list(payload['document_ids'])
        alive = set(
            (
                await session.execute(
                    select(Document.id).where(Document.id.in_(ids)).where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
                )
            )
            .scalars()
            .all()
        )
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
    """실패가 **일시적**인가(다시 시도하면 될 수 있는가) — 분류의 정의점.

    일시 = 상대(검색 엔진·임베딩·LLM·PG)에 닿지 못했거나 상대가 바빴다는 신호만: 연결·타임아웃 예외,
    429·5xx 응답. 401·400 같은 응답과 그 밖의 예외(파싱 ValueError, blob 없음, 프로그래밍 오류)는 전부
    **결정적** — 백 번 돌려도 같으므로 바로 failed로 보이는 것이 사용자에게 싸다.
    rag/chunking의 변환 예외도 이 기준에 맞춰 종류를 고른다(마감 초과·캡션/OCR 실패는 일시, 그림 장수 초과는 결정적).
    알려진 오분류: bulk 부분 실패(RuntimeError)에 섞인 rejected_execution은 일시인데 결정적으로 잡힌다(단일 워커라 드물다).
    routers/kms.py도 질의 준비 단계 실패를 503으로 볼지 같은 기준으로 판정한다 — 여기를 바꾸면 응답 코드도 바뀐다.
    """
    if isinstance(exc, (ConnectionError, TimeoutError, OperationalError, InterfaceError)):
        return True
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    # opensearchpy는 rag/ 어느 모듈도 톱레벨에 들이지 않는다(그 패키지 없이 import할 수 있게).
    from opensearchpy import ConnectionError as OsConnectionError
    from opensearchpy import TransportError

    if isinstance(exc, OsConnectionError):
        return True
    if isinstance(exc, TransportError):
        return exc.status_code in ('N/A', 429, 502, 503, 504)
    return False


def backoff_delay(attempts: int) -> timedelta:
    """attempts번째 실패 뒤 다음 시도까지 — 1·2·4·8·16·32분, 그 뒤 BACKOFF_CAP_MINUTES 고정."""
    return timedelta(minutes=min(2 ** (attempts - 1), BACKOFF_CAP_MINUTES))


def _on_failed(op: str, row_id: int, err: str) -> None:
    """failed 확정 지점 하나 — 알람을 붙일 때 여기만 채운다.

    이 시점부터 그 행의 PG↔엔진 불일치는 사람이 고칠 때까지 영구다(META·DROP·FAQ는 이 로그·카운터가 유일한 단서).
    """
    SEARCH_INDEX_FAILED_TOTAL.labels(op=op).inc()
    logger.error('outbox failed 확정 (op=%s id=%s): %s', op, row_id, err)


def due_pending_stmt(limit: int = BATCH, *, row_ids: list[int] | None = None):
    """drain이 이번 회차에 집을 행 — pending이고 시도 시각이 됐거나 정해지지 않은 것, id 순.

    row_ids를 주면 그 행들만 고르고 **백오프와 limit을 무시한다**("지정한 행을 지금 전부 처리").
    """
    stmt = (
        select(SearchIndexOutbox.id, SearchIndexOutbox.tenant_id, SearchIndexOutbox.op, SearchIndexOutbox.payload)
        .where(SearchIndexOutbox.status == PENDING)
        .order_by(SearchIndexOutbox.id)
    )
    if row_ids is not None:
        return stmt.where(SearchIndexOutbox.id.in_(list(row_ids)))
    return stmt.where(
        or_(SearchIndexOutbox.next_attempt_at.is_(None), SearchIndexOutbox.next_attempt_at <= func.now())
    ).limit(limit)


class _Row(NamedTuple):
    """due_pending_stmt가 고른 행 — 값으로 들고 간다(rollback이 ORM 객체를 만료시킨다)."""

    id: int
    tenant_id: str
    op: str
    payload: dict


async def drain(session, limit: int = BATCH, *, row_ids: list[int] | None = None) -> dict:
    """pending 행을 id 순으로 행 단위 격리·행 단위 commit으로 처리한다. 반환의 failed는 재시도·확정을 합친 수.

    row_ids는 테스트·eval용(공유 DB에서 남의 행을 처리하지 않게) — 운영 cron은 넘기지 않는다.
    INDEX_DOCUMENT의 done은 핸들러가 ready 승격과 같은 커밋에 찍으므로 여기서 건드리지 않는다.
    """
    result = await session.execute(due_pending_stmt(limit, row_ids=row_ids))
    rows = [_Row(row_id, tenant_id, op, payload or {}) for row_id, tenant_id, op, payload in result.all()]

    done = failed = 0
    started = time.monotonic()
    for row in rows:
        if row_ids is None and not _fits_round(time.monotonic() - started):
            break  # 남은 행은 pending 그대로 — 다음 회차(최대 1분 뒤)가 잇는다
        try:
            await _apply(session, row.op, row.payload, row.id)
            if row.op != INDEX_DOCUMENT:
                await mark_done(session, row.id)
            await session.commit()
            SEARCH_INDEX_SYNC_TOTAL.labels(op=row.op, result='ok').inc()
            done += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — 행 단위로 격리한다
            await _record_failure(session, row, e)
            failed += 1
    return {'done': done, 'failed': failed}


async def _record_failure(session: AsyncSession, row: _Row, exc: Exception) -> None:
    """실패를 행에 기록하고 재시도·확정을 갈라 한 커밋으로 끝낸다. 결정적 실패는 횟수를 안 본다 — 다시 돌려도 같다."""
    await session.rollback()  # 핸들러가 이 세션을 더럽혔을 수 있다
    err = str(exc)[:500]
    attempts = (
        await session.execute(
            update(SearchIndexOutbox)
            .where(SearchIndexOutbox.id == row.id)
            .values(attempts=SearchIndexOutbox.attempts + 1, last_error=err)
            .returning(SearchIndexOutbox.attempts)
        )
    ).scalar()
    terminal = not is_transient(exc) or attempts >= MAX_ATTEMPTS
    if terminal:
        await _mark_terminal(session, row, attempts, err)
    else:
        await _set_row(session, row.id, next_attempt_at=func.now() + backoff_delay(attempts))
    await session.commit()
    SEARCH_INDEX_SYNC_TOTAL.labels(op=row.op, result='error').inc()
    if terminal:
        _on_failed(row.op, row.id, err)
    if row.op == INDEX_DOCUMENT:
        # 문서 단위 결과를 따로 센다 — 위 카운터는 재시도 단위라 실패율의 분모가 못 된다.
        fname = (
            await session.execute(select(Document.filename).where(Document.id == row.payload.get('document_id')))
        ).scalar()
        INDEX_TOTAL.labels(ext=ext_label(fname or ''), result='failed' if terminal else 'retry').inc()
    logger.warning(
        'outbox 반영 실패 (op=%s id=%s attempts=%d%s): %s',
        row.op,
        row.id,
        attempts,
        ' → failed 확정' if terminal else f' → {backoff_delay(attempts)} 뒤 재시도',
        exc,
    )


async def _mark_terminal(session: AsyncSession, row: _Row, attempts: int, err: str) -> None:
    """failed 확정 — commit하지 않는다(호출부의 커밋에 얹힌다). INDEX_DOCUMENT면 문서도 failed로 찍고 잔여 청크 DROP을 등재한다."""
    await _set_row(session, row.id, status=FAILED)
    if row.op != INDEX_DOCUMENT:
        return
    doc_id = row.payload.get('document_id')
    # 문서도 failed — 사용자가 FE에서 실패를 보고 재업로드할 수 있게.
    # pending일 때만: 그새 삭제됐거나 다른 경로로 끝난 문서는 건드리지 않는다.
    await session.execute(
        update(Document)
        .where(Document.id == doc_id)
        .where(Document.status == 'pending')
        .values(status='failed', status_reason=f'색인 {attempts}회 실패: {err}'[:500])
    )
    # 잔여 청크 DROP을 같은 커밋에 — 엔진 쓰기 뒤 커밋만 실패했으면 검색 가능한 청크가 남아 있다.
    enqueue(session, row.tenant_id, DROP_DOCUMENTS, document_ids=[doc_id])


def _fits_round(elapsed: float) -> bool:
    """회차를 elapsed초 쓴 지금 행 하나를 더 시작해도 되는가 — 최악(변환 마감 + 꼬리)으로 걸려도 예산 안일 때만.

    첫 행은 항상 시작한다. 꼬리가 비정상으로 길면 arq가 회차를 취소하고 그 행은 attempts 없이 다음 회차에 다시 잡힌다.
    """
    return elapsed + settings.docling_document_timeout_seconds + INDEX_TAIL_SECONDS <= DRAIN_ROUND_SECONDS


async def drain_once(limit: int = BATCH, *, row_ids: list[int] | None = None) -> dict:
    """자기 세션으로 한 회차 처리. 이전 회차가 아직 돌고 있으면 즉시 빠진다(놓친 회차는 다음 분이 잡는다)."""
    if _drain_lock.locked():
        return {'done': 0, 'failed': 0, 'skipped': True}
    # 없는 인덱스에 bulk하면 동적 매핑으로 굳으므로, 인덱스를 보장하지 못하면 이 회차는 건너뛴다.
    if not await os_client.ensure_index_soft():
        return {'done': 0, 'failed': 0, 'skipped': True}
    async with _drain_lock:
        async with AsyncSessionLocal() as session:
            return await drain(session, limit, row_ids=row_ids)


async def pending_row_ids(session, *, document_id: int | None = None, faq_id: int | None = None) -> list[int]:
    """한 문서 또는 한 FAQ를 가리키는 pending 행 id(op 무관) — 테스트·eval이 row_ids로 넘길 값. 둘 중 하나만 넘긴다."""
    assert (document_id is None) != (faq_id is None), 'document_id 또는 faq_id 하나만'
    stmt = select(SearchIndexOutbox.id).where(SearchIndexOutbox.status == PENDING).order_by(SearchIndexOutbox.id)
    if document_id is not None:
        stmt = stmt.where(
            or_(
                SearchIndexOutbox.payload['document_id'].as_integer() == document_id,
                SearchIndexOutbox.payload['document_ids'].contains([document_id]),  # JSONB @> '[id]'
            )
        )
    else:
        stmt = stmt.where(
            or_(
                SearchIndexOutbox.payload['faq_id'].as_integer() == faq_id,
                SearchIndexOutbox.payload['faq_ids'].contains([faq_id]),
            )
        )
    return list((await session.execute(stmt)).scalars().all())
