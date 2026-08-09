"""검색 품질 측정 v2 — 멀티테넌트 gold로 Recall@5/20, Hit@1, MRR.

생성(LLM)과 분리된 검색 축 단독 평가. 기본 실행은 LLM 불필요 — TEI 임베딩 + DB만.
대상: single_fact/paraphrase/rare_lexical/multi_doc (multi_turn은 condense가
LLM 의존이라 이 축에서 제외 — 멀티턴 검색축은 eval.retrieval_mt가 담당, #5).

사용: python3 -m eval.retrieval_v2
     python3 -m eval.retrieval_v2 --expand   # 멀티쿼리 통합(#5) on A/B — 이때만 LLM 사용
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

    expand=True면 운영 플래그 on과 동일하게 condense_to_queries(#5, 빈 히스토리)로
    멀티쿼리를 만들어 첫 줄=주 쿼리, 나머지=변형으로 검색 (A/B). 변형은 rows에 기록.

    반환: {'rows': [...], 'skipped': int, 'overall': {recall_at_5, ...}, 'stale': {tenant: n}}
    """
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g['type'] in TYPES]

    # 1단계(expand만): LLM 선병렬 — vLLM 연속 배칭 활용 (#18). DB 접근 없어 gather 안전
    queries_map: dict[str, list[str]] = {}
    if expand:
        from rag.conversation import condense_to_queries
        from rag.llm import LlmClient
        llm = LlmClient()
        sem = asyncio.Semaphore(6)   # worker15 공유 장비 — RAGAS와 동일 안전선

        async def _expand(g):
            async with sem:
                return g['id'], await condense_to_queries(llm, g['query'], [])
        queries_map = dict(await asyncio.gather(*(_expand(g) for g in target)))

    by_tenant = defaultdict(list)
    for g in target:
        by_tenant[row_tenant(g)].append(g)

    # 2단계: 검색·채점 — 세션 직렬 (AsyncSession은 동시 실행 불가)
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
                if expand:
                    queries = queries_map[g['id']]
                    cands = await retrieve_candidates(session, tenant, queries[0], top_n=20,
                                                      expanded_queries=queries[1:])
                else:
                    queries = [g['query']]
                    cands = await retrieve_candidates(session, tenant, g['query'], top_n=20)
                scores = score_one([c.chunk_id for c in cands.chunks], gold_ids)
                row = {'id': g['id'], 'type': g['type'], 'tenant': tenant, 'scores': scores}
                if expand:
                    row['queries'] = queries
                rows.append(row)

    return {'rows': rows, 'skipped': skipped, 'overall': _overall(rows),
            'by_difficulty': _by_difficulty(rows), 'stale': stale}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--expand', action='store_true',
                        help='멀티쿼리 통합(#5) on으로 측정 — 결과는 retrieval_v2_expand.jsonl에 별도 저장 (A/B)')
    args = parser.parse_args()

    result = await compute(expand=args.expand)
    rows, skipped = result['rows'], result['skipped']
    if not rows:   # 채점 0 = gold resolve 전멸 (빈/잘못된 DB) — 결과 덮어쓰기 전에 중단 (#18 실사고 가드)
        raise SystemExit(f"채점 0문항 (스킵 {skipped}) — DATABASE_URL이 코퍼스 있는 DB인지 확인. 결과 파일 미변경.")
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
