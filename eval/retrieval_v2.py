"""검색 품질 측정 v2 — 멀티테넌트 gold로 Recall@5/20, Hit@1, MRR.

생성(LLM)과 분리된 검색 축 단독 평가. 기본 실행은 LLM 불필요 — TEI 임베딩 + DB만.
대상: single_fact/paraphrase/rare_lexical/multi_doc (multi_turn은 condense가
LLM 의존이라 검색 단독 축에서 제외 — v1과 동일 기준).

사용: python3 -m eval.retrieval_v2
     python3 -m eval.retrieval_v2 --expand   # 쿼리 확장(#3) on A/B — 이때만 LLM 사용
"""
import argparse
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


# 검색 난이도 층화 — type을 난이도 그룹으로 묶는다 (별도 태깅 없이 type 재활용).
# 쉬움: 직접 사실·고유용어(매칭 쉬움) / 어려움: 다문서 종합·멀티턴 맥락 의존.
DIFFICULTY = {
    'single_fact': 'easy', 'rare_lexical': 'easy',
    'paraphrase': 'medium',
    'multi_doc': 'hard', 'multi_turn': 'hard',
}


def _by_difficulty(rows) -> dict:
    """난이도 그룹별 R@5 — '어려운 질의에서만 약한지'를 평균에 가려지지 않게 본다."""
    groups = defaultdict(list)
    for r in rows:
        groups[DIFFICULTY.get(r['type'], 'medium')].append(r['scores'])
    out = {}
    for g, ss in groups.items():
        out[g] = sum(s['recall_at_5'] for s in ss) / len(ss) if ss else 0.0
    return out


async def compute(expand: bool = False) -> dict:
    """검색축 채점 실행 → 요약 반환 (출력·파일저장은 main이 담당).

    expand=True면 케이스별로 운영과 동일한 expand_query() 변형을 생성해
    retrieve_candidates에 전달 (#3 A/B). 변형은 rows에 같이 기록해 사후 검수 가능.

    반환: {'rows': [...], 'skipped': int, 'overall': {recall_at_5, ...}, 'stale': {tenant: n}}
    """
    llm = None
    if expand:
        from rag.conversation import expand_query
        from rag.llm import LlmClient
        llm = LlmClient()

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
                expanded = await expand_query(llm, g['query']) if llm else []
                cands = await retrieve_candidates(session, tenant, g['query'], top_n=20,
                                                  expanded_queries=expanded)
                scores = score_one([c.chunk_id for c in cands.chunks], gold_ids)
                row = {'id': g['id'], 'type': g['type'], 'tenant': tenant, 'scores': scores}
                if expand:
                    row['expanded'] = expanded
                rows.append(row)

    return {'rows': rows, 'skipped': skipped, 'overall': _overall(rows),
            'by_difficulty': _by_difficulty(rows), 'stale': stale}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--expand', action='store_true',
                        help='쿼리 확장(#3) on으로 측정 — 결과는 retrieval_v2_expand.jsonl에 별도 저장 (A/B)')
    args = parser.parse_args()

    result = await compute(expand=args.expand)
    rows, skipped = result['rows'], result['skipped']
    for tenant, n in result['stale'].items():
        print(f'⚠ {tenant} 라벨 노후 {n}건')

    # off 기본 결과를 덮어쓰지 않게 expand는 별도 파일 — 두 파일을 나란히 비교
    name = 'retrieval_v2_expand.jsonl' if args.expand else 'retrieval_v2.jsonl'
    out = Path(__file__).resolve().parent / 'results' / name
    out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    print(f'\n채점 {len(rows)}문항 (resolve 불가 스킵 {skipped})  →  {out}\n')
    print('[타입별]')
    _table(rows, lambda r: r['type'], 'type')
    print('\n[테넌트별]')
    _table(rows, lambda r: r['tenant'], 'tenant')


if __name__ == '__main__':
    asyncio.run(main())
