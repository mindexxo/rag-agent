"""스캔판이 검색에 주는 영향(#224) — 대상 문서의 골드 문항으로 '정답 문서가 처음 나오는 순위'를 잰다.

스캔판은 띄어쓰기·글자가 달라 골드 스니펫으로 정답 청크를 풀 수 없다 — 그래서 청크가 아니라 **문서** 단위로 본다.
같은 파일명으로 디지털판 / 스캔판(OCR 변형별)을 번갈아 색인해 놓고 이 스크립트를 돌린다(교체는 swap 명령).

사용:
  python -m eval.scan_quality.retrieval swap <digital|remote|local_min736|local_max|local_min400>   개발계 5건 교체
  python -m eval.scan_quality.retrieval measure <라벨>     → eval/results/scan224/retrieval_<라벨>.jsonl
"""

import asyncio
import json
import sys
from collections import defaultdict

import httpx
from sqlalchemy import select

from eval.scan_quality.scans import OUT, SPECS

_flat = lambda xs: [y for x in xs for y in (x if isinstance(x, list) else [x])]  # noqa: E731


def _gold():
    from eval.generation import load_gold

    names = {s.filename for s in SPECS}
    return [
        g
        for g in load_gold()
        if g.get('has_evidence') and not g.get('conversation') and set(_flat(g.get('expected_docs') or [])) & names
    ]


async def swap(kind: str) -> None:
    """5건을 디지털판 또는 스캔판으로 바꿔 올린다 — 옛 문서는 지우고(DROP drain) 새로 올린다(같은 파일명 재업로드 함정, #222)."""
    from database import AsyncSessionLocal
    from main import app
    from rag import outbox
    from rag.models import Document

    if kind != 'digital':
        from eval.scan_quality.ocr_variant import install

        install(kind)
    for s in SPECS:
        async with AsyncSessionLocal() as db:
            old = (
                (
                    await db.execute(
                        select(Document.id).where(
                            Document.tenant_id == s.tenant,
                            Document.filename == s.filename,
                            Document.status != 'deleted',
                        )
                    )
                )
                .scalars()
                .all()
            )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url='http://local',
            headers={'X-Tenant-Id': s.tenant},
            timeout=1200,
        ) as c:
            if old:
                await c.delete('/kms/documents?' + '&'.join(f'ids={i}' for i in old))
                # 지운 문서의 행만 — 테넌트 전체 pending을 집으면 공유 개발계에서 다른 세션 업로드까지 이 프로세스가 처리해
                # 그쪽 blob이 없으면 failed로 굳는다(#179와 같은 함정, #224 리뷰)
                async with AsyncSessionLocal() as db:
                    rows = [r for i in old for r in await outbox.pending_row_ids(db, document_id=i)]
                _check(await outbox.drain_once(row_ids=rows), f'{s.filename} 옛 판 삭제')
            src = s.src if kind == 'digital' else s.out
            r = await c.post('/kms/documents', files={'file': (s.filename, src.read_bytes(), 'application/pdf')})
            doc_id = r.json()['document_id']
            async with AsyncSessionLocal() as db:
                ids = await outbox.pending_row_ids(db, document_id=doc_id)
            res = _check(await outbox.drain_once(row_ids=ids), f'{s.filename} 색인')
            async with AsyncSessionLocal() as db:
                d = await db.get(Document, doc_id)
            if d.status != 'ready':
                raise RuntimeError(f'{s.filename} 색인 실패: {d.status} {d.status_reason}')
            print(kind, s.filename, d.status, res, flush=True)


def _check(res: dict, what: str) -> dict:
    """drain이 건너뛰었거나(다른 drain이 잠금 중·색인 확인 실패) 실패한 행이 있으면 멈춘다 — 그대로 두면 옛 청크가 남은 채
    measure가 라벨과 다른 상태를 잰다(#224 리뷰)."""
    if res.get('skipped') or res.get('failed'):
        raise RuntimeError(f'{what}: drain 결과 {res}')
    return res


async def measure(label: str) -> None:
    from database import AsyncSessionLocal
    from eval.generation import row_tenant
    from rag.search.retriever import retrieve_candidates

    out, by = [], defaultdict(list)
    async with AsyncSessionLocal() as db:
        for g in _gold():
            want = set(_flat(g['expected_docs'])) & {s.filename for s in SPECS}
            cands = (await retrieve_candidates(db, row_tenant(g), g['query'], top_n=20)).chunks
            rank = next((i + 1 for i, c in enumerate(cands) if c.filename in want), None)
            out.append({'id': g['id'], 'doc': sorted(want)[0], 'rank': rank})
            by[sorted(want)[0]].append(rank)
    (OUT / f'retrieval_{label}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in out) + '\n')
    for doc, ranks in by.items():
        n = len(ranks)
        print(
            f"{label:<14} {doc[:28]:<30} n={n:>2} top1 {sum(r == 1 for r in ranks) / n:.2f} top5 {sum(bool(r) and r <= 5 for r in ranks) / n:.2f} "
            f"miss20 {sum(r is None for r in ranks)}"
        )


if __name__ == '__main__':
    cmd, arg = sys.argv[1], sys.argv[2]

    async def _run():
        try:
            await (swap(arg) if cmd == 'swap' else measure(arg))
        finally:
            from rag.search import os_client

            await os_client.close_client()

    asyncio.run(_run())
