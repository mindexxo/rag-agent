"""PG↔OpenSearch 문서 단위 재동기화 — `rag.opensearch.reconcile`의 실행 진입점.

    python -m eval.os_reconcile            # 대조만 (dry-run) — 누락·잉여 문서 수 출력
    python -m eval.os_reconcile --apply    # 누락 ready 문서를 pending으로 되돌려 대기열 등재,
                                           # PG에 없는 문서의 청크를 색인에서 삭제

언제 쓰나: outbox가 failed로 확정한 문서를 사람이 고친 뒤, 엔진 스냅샷 복구 뒤, 또는 "색인이
낡았다"는 의심이 들 때. 청크 수준 드리프트(일부 청크 누락)는 잡지 못한다 — 문서 단위 색인은
한 커밋으로 끝나므로 그 경우는 outbox 원자성이 막는다(rag/opensearch.py "정합 관리").

`--apply` 없이 돌리면 아무것도 바꾸지 않는다: 두 집합의 차이만 센다.
"""
import argparse
import asyncio

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from rag import opensearch
from rag.models import Document


async def _diff(session) -> tuple[set[int], set[int]]:
    """reconcile과 같은 순서(OS 먼저)로 두 집합을 읽어 (missing, extra)를 돌려준다."""
    agg = await opensearch.client().search(index=settings.opensearch_index, body={
        "size": 0, "aggs": {"d": {"terms": {"field": "document_id", "size": 65536}}}})
    os_docs = {int(b["key"]) for b in agg["aggregations"]["d"]["buckets"]}
    pg_docs = set((await session.execute(
        select(Document.id).where(Document.status == 'ready'))).scalars().all())
    return pg_docs - os_docs, os_docs - pg_docs


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='실제로 재등재·삭제한다 (기본은 대조만)')
    args = ap.parse_args()

    async with AsyncSessionLocal() as session:
        if not args.apply:
            missing, extra = await _diff(session)
            print(f"색인 누락(ready인데 OS에 없음) {len(missing)}건: {sorted(missing)[:20]}")
            print(f"색인 잉여(OS에만 있음)          {len(extra)}건: {sorted(extra)[:20]}")
            print("변경 없음 — 반영하려면 --apply")
            return
        r = await opensearch.reconcile(session)
        print(f"PG ready {r['pg']} · OS 문서 {r['os']} · 재등재 {r['indexed']} · 삭제 청크 {r['deleted']}")
        if r['indexed']:
            print("재등재분은 워커 cron(1분)이 다시 색인한다 — 즉시 반영이 필요하면 drain_once를 부를 것")


async def _run() -> None:
    try:
        await main()
    finally:
        await opensearch.close_client()      # aiohttp 세션 미종료 경고 방지


if __name__ == '__main__':
    asyncio.run(_run())
