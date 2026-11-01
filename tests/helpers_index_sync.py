"""색인 동기화 통합 테스트 공용 헬퍼 — outbox 행 조회·검색 후보 문서 집합·커밋 단계 고장 주입."""

from sqlalchemy import select

from database import AsyncSessionLocal
from rag import outbox
from rag.models import SearchIndexOutbox
from rag.search.retriever import retrieve_candidates


async def outbox_rows(doc_id):
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox).order_by(SearchIndexOutbox.id))).scalars().all()
    return [
        (r.op, r.status, r.attempts, r.last_error)
        for r in rows
        if r.payload.get('document_id') == doc_id or doc_id in (r.payload.get('document_ids') or [])
    ]


async def cited_ids(tenant_id, query='반품 기간') -> set[int]:
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, query, top_n=20)
    return {c.document_id for c in cands.chunks}


async def latest_index_row(doc_id) -> SearchIndexOutbox:
    """문서의 **최신** INDEX_DOCUMENT 행 — attempts·next_attempt_at·status를 본다.
    failed 행을 재사용해 재업로드하면 같은 document_id로 행이 둘이라 최신을 고른다."""
    async with AsyncSessionLocal() as s:
        return (
            (
                await s.execute(
                    select(SearchIndexOutbox)
                    .where(SearchIndexOutbox.op == outbox.INDEX_DOCUMENT)
                    .where(SearchIndexOutbox.payload['document_id'].as_integer() == doc_id)
                    .order_by(SearchIndexOutbox.id.desc())
                    .limit(1)
                )
            )
            .scalars()
            .one()
        )


async def drop_rows(doc_id):
    async with AsyncSessionLocal() as s:
        return (
            (
                await s.execute(
                    select(SearchIndexOutbox)
                    .where(SearchIndexOutbox.op == outbox.DROP_DOCUMENTS)
                    .where(SearchIndexOutbox.payload['document_ids'].contains([doc_id]))
                    .order_by(SearchIndexOutbox.id)
                )
            )
            .scalars()
            .all()
        )


async def boom_mark_done(session, row_id):
    """③ 커밋 단계만 죽인다 — ②는 끝나 엔진에 청크가 있고, mark_done(③이 ready 승격과 함께 부른다)이 폭발.
    결정적 예외라 1회로 failed 확정. 픽스처 monkeypatch가 아니라 자기 context로 건다 — 픽스처를 `undo()`하면
    fake_embed까지 걷혀 실제 임베딩 서버로 나간다."""
    raise RuntimeError('③ 커밋 직전 폭발')
