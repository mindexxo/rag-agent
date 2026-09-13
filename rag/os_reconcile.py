"""PG↔OpenSearch 재동기화 — 안전판 (#146, 구 rag/opensearch.py에서 분리).

PG에 청크가 없으므로 문서·FAQ **단위**로만 대조한다 — 청크 일부 누락은 못 잡고 outbox의
문서 단위 원자성(문서 하나를 색인한 뒤 한 커밋)에 의존한다.

**읽는 순서가 계약이다: OpenSearch 먼저, PG 나중.** 두 스냅샷 사이에 인제스션이 끼면 새 문서는
PG에만 있어 'missing'으로 판정돼 재등재된다 — 이미 있는 것을 다시 넣는 무해한 방향이다.
반대 순서면 색인에만 있어 'extra'로 판정돼 **방금 색인된 청크를 지운다.**

평상시 정합은 이 파일이 아니라 트랜잭셔널 outbox가 지킨다(`rag/outbox.py`가 정의점) — 여기는
그게 놓친 것을 줍는 안전판이다.

## 실행

    python -m rag.os_reconcile                     # 대조만 (dry-run) — 누락·잉여 수 출력, 변경 없음
    python -m rag.os_reconcile --apply             # 누락 → 대기열 등재(문서는 pending으로 되돌림), 잉여 → 색인에서 삭제
    python -m rag.os_reconcile --tenant <id> ...   # 한 테넌트만

**`rag/` 아래에 둔 이유**: 도커 이미지가 `eval/`을 제외하므로(`.dockerignore`) 구 eval 진입점은
컨테이너 안에서 실행할 수 없었다(#139 배포에서 실측 — ssh 터널로 개발자 PC에서 돌려야 했다).
운영 복구 명령이 운영 이미지 안에 있어야 한다.

언제 쓰나: outbox가 failed로 확정한 문서를 사람이 고친 뒤, 엔진 스냅샷 복구 뒤, **첫 배포 후 색인
채우기**(PG에 있는 문서·FAQ 전부가 누락으로 잡혀 등재된다 — 워커가 분당 50건씩 처리), 또는
"색인이 낡았다"는 의심이 들 때.
"""
import argparse
import asyncio

from sqlalchemy import select, update

from config import settings
from database import AsyncSessionLocal
from rag import os_client, outbox
from rag.models import Document, Faq
from rag.os_client import client
from rag import os_index


def _tenant_term(tenant_id):
    return {"term": {"tenant_id": tenant_id}} if tenant_id else {"match_all": {}}


async def _indexed_parent_ids(field: str, tenant_id) -> set[int]:
    """색인에 있는 부모 id 집합(document_id 또는 faq_id) — terms 집계. 반드시 PG보다 **먼저** 읽는다."""
    agg = await client().search(index=settings.opensearch_index, body={
        "size": 0, "query": _tenant_term(tenant_id),
        "aggs": {"p": {"terms": {"field": field, "size": 65536}}}})
    return {int(b["key"]) for b in agg["aggregations"]["p"]["buckets"]}


async def reconcile(session, tenant_id: str | None = None) -> dict:
    """PG↔OS **문서** 단위 재동기화 — 안전판. `python -m rag.os_reconcile`이 부른다.

    tenant_id를 주면 그 테넌트만 본다 — 공유 DB에서 한 테넌트를 손볼 때, 그리고 테스트가
    남의 문서를 pending으로 되돌리지 않게. 없으면 전체.
    반환: {'pg': ready 문서 수, 'os': 색인 문서 수, 'indexed': 재등재 수, 'deleted': 삭제 청크 수}.
    PG에 청크가 없으므로 "ready 문서가 색인에도 있는가"만 본다 — 청크 수준 드리프트(일부 누락)는
    못 잡고 outbox의 문서 단위 원자성에 의존한다.

    **읽는 순서가 계약이다: OpenSearch 먼저, PG 나중.** 두 스냅샷 사이에 인제스션이 끼면
    새 문서는 pg_docs에만 있어 'missing'으로 판정돼 재등재된다 — 이미 있는 것을 다시 넣는
    무해한 방향이다. 반대 순서면 os_docs에만 있어 'extra'로 판정돼 **방금 색인된 청크를 지운다.**
    """

    os_docs = await _indexed_parent_ids("document_id", tenant_id)
    stmt = select(Document.id, Document.tenant_id).where(Document.status == 'ready')
    if tenant_id:
        stmt = stmt.where(Document.tenant_id == tenant_id)
    pg_rows = (await session.execute(stmt)).all()
    pg_docs = {did for did, _ in pg_rows}
    tenant_of = dict(pg_rows)

    missing, extra = pg_docs - os_docs, os_docs - pg_docs
    # 색인에 없는 ready 문서는 pending으로 되돌려 대기열에 넣는다 — 재파싱·재임베딩·ready 승격은
    # 인제스션 핸들러 한 곳(rag/documents.index_pending_document)이 맡는다. 두 벌을 두지 않는다.
    if missing:
        await session.execute(update(Document).where(Document.id.in_(list(missing))).values(status='pending'))
        for did in sorted(missing):
            outbox.enqueue(session, tenant_of[did], outbox.INDEX_DOCUMENT, document_id=did)
        await session.commit()
    deleted = await os_index.drop_documents_now(sorted(extra)) if extra else 0
    return {'pg': len(pg_docs), 'os': len(os_docs), 'indexed': len(missing), 'deleted': deleted,
            'unit': 'document'}


async def reconcile_faqs(session, tenant_id: str | None = None) -> dict:
    """PG↔OS **FAQ** 단위 재동기화 — reconcile()의 FAQ 짝. 순서 계약 동일(OS 먼저).

    FAQ는 활성 여부와 무관하게 전부 색인 대상이다(비활성은 searchable=False로 들어간다 —
    effective_searchable). 그래서 PG의 모든 FAQ 행과 대조한다. 누락은 INDEX_FAQ 행으로 등재
    (워커가 재임베딩·색인), 잉여(PG에 없는 faq_id)는 색인에서 지운다.
    반환: {'pg': FAQ 수, 'os': 색인 FAQ 수, 'indexed': 재등재 수, 'deleted': 삭제 청크 수}.
    """

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
    return {'pg': len(pg_faqs), 'os': len(os_faqs), 'indexed': len(missing), 'deleted': deleted,
            'unit': 'faq'}


async def _diff(session, tenant_id):
    """reconcile과 같은 순서(OS 먼저)로 문서·FAQ 두 집합의 차이를 읽기만 한다."""
    os_docs = await _indexed_parent_ids("document_id", tenant_id)
    os_faqs = await _indexed_parent_ids("faq_id", tenant_id)
    d = select(Document.id).where(Document.status == 'ready')
    f = select(Faq.id)
    if tenant_id:
        d, f = d.where(Document.tenant_id == tenant_id), f.where(Faq.tenant_id == tenant_id)
    pg_docs = set((await session.execute(d)).scalars().all())
    pg_faqs = set((await session.execute(f)).scalars().all())
    return (pg_docs - os_docs, os_docs - pg_docs), (pg_faqs - os_faqs, os_faqs - pg_faqs)


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
        await os_client.close_client()      # aiohttp 세션 미종료 경고 방지


if __name__ == '__main__':
    asyncio.run(_run())
