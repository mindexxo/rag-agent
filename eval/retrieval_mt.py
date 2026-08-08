"""멀티턴 검색축 (#5 신설) — gold multi_turn 90문항으로 condense→검색 파이프라인 채점.

retrieval_v2가 "condense는 LLM 의존"이라 제외해 온 축. 멀티쿼리 통합(#5)의 핵심 가설
("이전 답변의 문서 어휘를 아는 상태에서 만든 변형이 멀티턴 검색을 개선한다")을 재려면
condense를 실제로 태워야 하므로 별도 스크립트로 둔다 (기본 축은 LLM-free 원칙 유지).

off: condense_query(현행) → 단일 쿼리 검색
on : condense_to_queries(#5) → 첫 줄 + 변형 RRF 융합 검색 (--multi)

condense가 확률적이므로 off/on 모두 LLM 경유 — 같은 조건에서 비교된다.

사용: python3 -m eval.retrieval_mt           # off (condense 단일 쿼리)
     python3 -m eval.retrieval_mt --multi   # on  (멀티쿼리 융합)
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

from database import AsyncSessionLocal
from eval.generation import row_tenant
from eval.retrieval import resolve_gold, score_one
from eval.retrieval_v2 import GOLD, METRICS
from rag.conversation import condense_query, condense_to_queries
from rag.llm import LlmClient
from rag.retriever import retrieve_candidates


async def compute(multi: bool = False) -> dict:
    """multi_turn 축 채점 → 요약 반환. 반환 형식은 retrieval_v2.compute와 동일 계열."""
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g['type'] == 'multi_turn']

    llm = LlmClient()
    rows, skipped, stale = [], 0, {}
    async with AsyncSessionLocal() as session:
        by_tenant = {}
        for g in target:
            by_tenant.setdefault(row_tenant(g), []).append(g)
        for tenant, items in by_tenant.items():
            resolved = await resolve_gold(session, tenant, items)
            if resolved.stale:
                stale[tenant] = len(resolved.stale)
            for g in items:
                gold_ids = resolved.chunk_ids.get(g['id']) or []
                if not gold_ids:
                    skipped += 1
                    continue
                # gold의 conversation을 condense가 받는 Message 형태로 (eval.condense와 동일 방식)
                msgs = [SimpleNamespace(role=m['role'], content=m['content'])
                        for m in (g.get('conversation') or [])]
                if multi:
                    queries = await condense_to_queries(llm, g['query'], msgs)
                else:
                    queries = [await condense_query(llm, g['query'], msgs)]
                cands = await retrieve_candidates(session, tenant, queries[0], top_n=20,
                                                  expanded_queries=queries[1:])
                scores = score_one([c.chunk_id for c in cands.chunks], gold_ids)
                rows.append({'id': g['id'], 'type': g['type'], 'tenant': tenant,
                             'scores': scores, 'queries': queries})

    overall = {m: (sum(r['scores'][m] for r in rows) / len(rows) if rows else 0.0) for m in METRICS}
    return {'rows': rows, 'skipped': skipped, 'overall': overall, 'stale': stale}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--multi', action='store_true',
                        help='멀티쿼리 통합(#5) on으로 측정 — 결과는 retrieval_mt_multi.jsonl에 별도 저장')
    args = parser.parse_args()

    result = await compute(multi=args.multi)
    rows = result['rows']
    for tenant, n in result['stale'].items():
        print(f'⚠ {tenant} 라벨 노후 {n}건')

    name = 'retrieval_mt_multi.jsonl' if args.multi else 'retrieval_mt.jsonl'
    out = Path(__file__).resolve().parent / 'results' / name
    out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    print(f'\n[multi_turn 검색축 — {"멀티쿼리 on" if args.multi else "condense 단일(off)"}]')
    print(f'채점 {len(rows)}문항 (resolve 불가 스킵 {result["skipped"]})  →  {out}')
    print(''.join(f'{m:>12}' for m in ['R@5', 'R@20', 'Hit@1', 'MRR']))
    print(''.join(f'{result["overall"][m]:>12.3f}' for m in METRICS))


if __name__ == '__main__':
    asyncio.run(main())
