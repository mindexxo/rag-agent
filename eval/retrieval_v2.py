"""검색 품질 측정 v2 — 멀티테넌트 gold로 Recall@5/20, Hit@1, MRR.

생성(LLM)과 분리된 검색 축 단독 평가. LLM 불필요 — TEI 임베딩 + DB만.
대상: single_fact/paraphrase/rare_lexical/multi_doc (multi_turn은 condense가
LLM 의존이라 검색 단독 축에서 제외 — v1과 동일 기준).

사용: python3 -m eval.retrieval_v2
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from database import AsyncSessionLocal
from eval.generation import row_tenant
from eval.retrieval import resolve_gold, score_one
from rag.retriever import retrieve_candidates

GOLD = Path(__file__).resolve().parent / 'gold_set_v2.jsonl'
TYPES = {'single_fact', 'paraphrase', 'rare_lexical', 'multi_doc'}
METRICS = ['recall_at_5', 'recall_at_20', 'hit_at_1', 'mrr']


def _table(rows, group_fn, label):
    groups = defaultdict(list)
    for r in rows:
        groups[group_fn(r)].append(r['scores'])
    print(f'{label:<14}{"n":>5}' + ''.join(f'{m:>10}' for m in ['R@5', 'R@20', 'Hit@1', 'MRR']))
    for g in sorted(groups):
        ss = groups[g]
        avgs = [sum(s[m] for s in ss) / len(ss) for m in METRICS]
        print(f'{g:<14}{len(ss):>5}' + ''.join(f'{a:>10.3f}' for a in avgs))
    all_ss = [s for ss in groups.values() for s in ss]
    avgs = [sum(s[m] for s in all_ss) / len(all_ss) for m in METRICS]
    print(f'{"(전체)":<14}{len(all_ss):>5}' + ''.join(f'{a:>10.3f}' for a in avgs))


def _overall(rows) -> dict:
    """전체 평균 지표 dict — run_all 요약·직전 대비 비교용."""
    all_ss = [r['scores'] for r in rows]
    if not all_ss:
        return {m: 0.0 for m in METRICS}
    return {m: sum(s[m] for s in all_ss) / len(all_ss) for m in METRICS}


async def compute() -> dict:
    """검색축 채점 실행 → 요약 반환 (출력·파일저장은 main이 담당).

    반환: {'rows': [...], 'skipped': int, 'overall': {recall_at_5, ...}, 'stale': {tenant: n}}
    """
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g['type'] in TYPES]

    by_tenant = defaultdict(list)
    for g in target:
        by_tenant[row_tenant(g)].append(g)

    rows, skipped, stale = [], 0, {}
    async with AsyncSessionLocal() as session:
        for tenant, items in by_tenant.items():
            resolved = await resolve_gold(session, tenant, items)
            if resolved.stale:
                stale[tenant] = len(resolved.stale)
            for g in items:
                gold_ids = resolved.chunk_ids.get(g['id']) or []
                if not gold_ids:
                    skipped += 1
                    continue
                cands = await retrieve_candidates(session, tenant, g['query'], top_n=20)
                scores = score_one([c.chunk_id for c in cands.chunks], gold_ids)
                rows.append({'id': g['id'], 'type': g['type'], 'tenant': tenant, 'scores': scores})

    return {'rows': rows, 'skipped': skipped, 'overall': _overall(rows), 'stale': stale}


async def main() -> None:
    result = await compute()
    rows, skipped = result['rows'], result['skipped']
    for tenant, n in result['stale'].items():
        print(f'⚠ {tenant} 라벨 노후 {n}건')

    out = Path(__file__).resolve().parent / 'results' / 'retrieval_v2.jsonl'
    out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    print(f'\n채점 {len(rows)}문항 (resolve 불가 스킵 {skipped})  →  {out}\n')
    print('[타입별]')
    _table(rows, lambda r: r['type'], 'type')
    print('\n[테넌트별]')
    _table(rows, lambda r: r['tenant'], 'tenant')


if __name__ == '__main__':
    asyncio.run(main())
