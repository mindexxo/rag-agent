"""PG↔OpenSearch 문서·FAQ 단위 재동기화 — outbox가 놓친 것을 줍는 안전판(평상시 정합은 `rag/outbox.py`).

PG에 청크가 없어 청크 일부 누락은 못 잡는다. 두 가지가 계약이다:
- 읽는 순서: OpenSearch 먼저, PG 나중(반대면 두 스냅샷 사이에 색인된 문서를 '잉여'로 보고 지운다).
- 잉여의 기준은 ready가 아니라 **살아 있는 문서**(ALIVE_DOCUMENT_STATUSES): 색인 중인 pending 문서는 청크가 엔진에
  먼저 들어가고 ready 커밋은 뒤에 온다. ready만 기준으로 삼으면 그 사이에 돈 --apply가 청크를 지워 청크 없는 ready가 된다.
  누락의 기준은 ready뿐이다 — pending은 어차피 대기열이 색인한다.

    python -m rag.os_reconcile                     # 대조만 (dry-run)
    python -m rag.os_reconcile --apply             # 누락 → 대기열 등재(문서는 pending으로), 잉여 → 색인에서 삭제
    python -m rag.os_reconcile --tenant <id> ...   # 한 테넌트만

운영 복구 명령이라 도커 이미지가 제외하는 `eval/`이 아니라 `rag/`에 둔다. 첫 배포 후 색인 채우기에도 쓴다.
"""

import argparse
import asyncio

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from database import AsyncSessionLocal
from rag import outbox
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Faq
from rag.search import os_client, os_index
from rag.search.os_client import client


def _tenant_term(tenant_id):
    return {"term": {"tenant_id": tenant_id}} if tenant_id else {"match_all": {}}


async def _indexed_parent_ids(field: str, tenant_id) -> set[int]:
    """색인에 있는 부모 id 집합(document_id 또는 faq_id) — terms 집계. 반드시 PG보다 **먼저** 읽는다.

    집계 size가 상한이다 — 한 테넌트(전체 실행이면 전체)의 부모 id가 65,536개를 넘으면 그 너머는 "색인에 없음"으로
    오판해 재등재한다. 그 규모가 되면 composite 집계로 바꿔야 한다.
    """
    agg = await client().search(
        index=settings.opensearch_index,
        body={"size": 0, "query": _tenant_term(tenant_id), "aggs": {"p": {"terms": {"field": field, "size": 65536}}}},
    )
    return {int(b["key"]) for b in agg["aggregations"]["p"]["buckets"]}


async def reconcile(session, tenant_id: str | None = None) -> dict:
    """PG↔OS 문서 단위 재동기화 — 누락은 ready 기준, 잉여는 살아 있는 문서 기준(모듈 docstring). OS를 PG보다 먼저 읽는다.

    tenant_id가 없으면 전체. 반환: {'pg': ready 문서 수, 'os': 색인 문서 수, 'indexed': 재등재 수, 'deleted': 삭제 청크 수}.
    """

    os_docs = await _indexed_parent_ids("document_id", tenant_id)
    pg_docs, alive, tenant_of = await _pg_documents(session, tenant_id)

    missing, extra = pg_docs - os_docs, os_docs - alive
    # pending으로 되돌려 재등재한다 — 재색인·ready 승격은 인제스션 핸들러(index_pending_document) 한 곳이 맡는다.
    if missing:
        await session.execute(
            update(Document)
            .where(Document.id.in_(list(missing)), Document.status == 'ready')  # 읽은 뒤 삭제된 문서를 되살리지 않는다
            .values(status='pending')
        )
        for did in sorted(missing):
            outbox.enqueue(session, tenant_of[did], outbox.INDEX_DOCUMENT, document_id=did)
        await session.commit()
    deleted = await os_index.drop_documents_now(sorted(extra)) if extra else 0
    return {'pg': len(pg_docs), 'os': len(os_docs), 'indexed': len(missing), 'deleted': deleted, 'unit': 'document'}


async def reconcile_faqs(session, tenant_id: str | None = None) -> dict:
    """reconcile()의 FAQ 짝(OS 먼저). 비활성 FAQ도 searchable=False로 색인되므로 PG의 모든 FAQ 행과 대조한다."""

    os_faqs = await _indexed_parent_ids("faq_id", tenant_id)
    stmt = select(Faq.id, Faq.tenant_id)
    if tenant_id:
        stmt = stmt.where(Faq.tenant_id == tenant_id)
    pg_rows = (await session.execute(stmt)).all()
    pg_faqs = {fid for fid, _ in pg_rows}
    tenant_of = dict(pg_rows)

    missing, extra = pg_faqs - os_faqs, os_faqs - pg_faqs
    if missing:
        for fid in sorted(missing):
            outbox.enqueue(session, tenant_of[fid], outbox.INDEX_FAQ, faq_id=fid)
        await session.commit()
    deleted = await os_index.drop_faqs_now(sorted(extra)) if extra else 0
    return {'pg': len(pg_faqs), 'os': len(os_faqs), 'indexed': len(missing), 'deleted': deleted, 'unit': 'faq'}


async def _pg_documents(session: AsyncSession, tenant_id: str | None) -> tuple[set[int], set[int], dict[int, str]]:
    """(ready 문서 id, 살아 있는 문서 id, id→tenant) — reconcile과 _diff가 같은 기준을 쓰게 한 곳에서 읽는다."""
    stmt = select(Document.id, Document.tenant_id, Document.status).where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
    if tenant_id:
        stmt = stmt.where(Document.tenant_id == tenant_id)
    rows = (await session.execute(stmt)).all()
    ready = {did for did, _, status in rows if status == 'ready'}
    return ready, {did for did, _, _ in rows}, {did: tid for did, tid, _ in rows}


async def _diff(session, tenant_id):
    """reconcile과 같은 순서(OS 먼저)·같은 기준으로 문서·FAQ 두 집합의 차이를 읽기만 한다."""
    os_docs = await _indexed_parent_ids("document_id", tenant_id)
    os_faqs = await _indexed_parent_ids("faq_id", tenant_id)
    pg_docs, alive, _ = await _pg_documents(session, tenant_id)
    f = select(Faq.id)
    if tenant_id:
        f = f.where(Faq.tenant_id == tenant_id)
    pg_faqs = set((await session.execute(f)).scalars().all())
    return (pg_docs - os_docs, os_docs - alive), (pg_faqs - os_faqs, os_faqs - pg_faqs)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='실제로 재등재·삭제한다 (기본은 대조만)')
    ap.add_argument('--tenant', help='이 테넌트만 (기본 전체)')
    args = ap.parse_args()

    async with AsyncSessionLocal() as session:
        if not args.apply:
            (dm, dx), (fm, fx) = await _diff(session, args.tenant)
            print(f"문서 누락(ready인데 OS에 없음) {len(dm)}건: {sorted(dm)[:20]}")
            print(f"문서 잉여(OS에만 있음)          {len(dx)}건: {sorted(dx)[:20]}")
            print(f"FAQ  누락                       {len(fm)}건: {sorted(fm)[:20]}")
            print(f"FAQ  잉여                       {len(fx)}건: {sorted(fx)[:20]}")
            print("변경 없음 — 반영하려면 --apply")
            return
        r = await reconcile(session, args.tenant)
        print(f"문서: PG ready {r['pg']} · OS {r['os']} · 재등재 {r['indexed']} · 삭제 청크 {r['deleted']}")
        r = await reconcile_faqs(session, args.tenant)
        print(f"FAQ : PG {r['pg']} · OS {r['os']} · 재등재 {r['indexed']} · 삭제 청크 {r['deleted']}")
        print("재등재분은 워커 cron(1분, 회차당 50행)이 처리한다 — 즉시 반영이 필요하면 drain_once를 부를 것")


async def _run() -> None:
    try:
        await main()
    finally:
        await os_client.close_client()  # aiohttp 세션 미종료 경고 방지


if __name__ == '__main__':
    asyncio.run(_run())
