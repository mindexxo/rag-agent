"""PG↔OpenSearch 재동기화 — `rag.os_reconcile`의 reconcile(문서)·reconcile_faqs(FAQ) 실행 진입점.

    python -m eval.os_reconcile                     # 대조만 (dry-run) — 누락·잉여 수 출력, 변경 없음
    python -m eval.os_reconcile --apply             # 누락 → 대기열 등재(문서는 pending으로 되돌림), 잉여 → 색인에서 삭제
    python -m eval.os_reconcile --tenant <id> ...   # 한 테넌트만

언제 쓰나: outbox가 failed로 확정한 문서를 사람이 고친 뒤, 엔진 스냅샷 복구 뒤, **첫 배포 후 색인 채우기**
(PG에 있는 문서·FAQ 전부가 누락으로 잡혀 등재된다 — 워커가 분당 50건씩 처리), 또는 "색인이 낡았다"는 의심이
들 때. 청크 수준 드리프트(일부 청크 누락)는 잡지 못한다 — 문서 단위 색인은 한 커밋으로 끝나므로 그 경우는
outbox 원자성이 막는다(rag/outbox.py가 정의점).
"""
import argparse
import asyncio

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from rag import os_client, os_reconcile
from rag.models import Document, Faq


async def _diff(session, tenant_id):
    """reconcile과 같은 순서(OS 먼저)로 문서·FAQ 두 집합의 차이를 읽기만 한다."""
    os_docs = await os_reconcile._indexed_parent_ids("document_id", tenant_id)
    os_faqs = await os_reconcile._indexed_parent_ids("faq_id", tenant_id)
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
        r = await os_reconcile.reconcile(session, args.tenant)
        print(f"문서: PG ready {r['pg']} · OS {r['os']} · 재등재 {r['indexed']} · 삭제 청크 {r['deleted']}")
        r = await os_reconcile.reconcile_faqs(session, args.tenant)
        print(f"FAQ : PG {r['pg']} · OS {r['os']} · 재등재 {r['indexed']} · 삭제 청크 {r['deleted']}")
        print("재등재분은 워커 cron(1분, 회차당 50행)이 처리한다 — 즉시 반영이 필요하면 drain_once를 부를 것")


async def _run() -> None:
    try:
        await main()
    finally:
        await os_client.close_client()      # aiohttp 세션 미종료 경고 방지


if __name__ == '__main__':
    asyncio.run(_run())
