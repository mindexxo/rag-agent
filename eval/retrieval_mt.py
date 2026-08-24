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

from config import settings
from database import AsyncSessionLocal
from eval._gold_history import messages_from_conversation
from eval.generation import row_tenant
from eval.retrieval import resolve_gold, score_one
from eval.retrieval_v2 import GOLD, METRICS
from rag.conversation import (condense_query, condense_to_queries,
                              trim_messages_for_condense)
from rag.llm import LlmClient
from rag.retriever import retrieve_candidates


TYPES = {'multi_turn', 'multi_turn_long'}
# multi_turn      : 1턴 히스토리 (기존 90문항 — 수치 비교 연속성 유지)
# multi_turn_long : 6~16메시지 긴 히스토리 (#5 E2E에서 발굴한 유형: 주제 전환·상대 참조·
#                   장거리 참조·히스토리 예산 트리밍. 실서버 대화를 재료로 라벨링)


CONCURRENCY = 6   # worker15 공유 장비 — RAGAS max_workers와 동일 안전선 (#18)


async def compute(multi: bool = False) -> dict:
    """multi_turn 축 채점 → 요약 반환. 반환 형식은 retrieval_v2.compute와 동일 계열.

    LLM(condense)은 선병렬(vLLM 연속 배칭 활용, #18), 검색·채점은 세션 직렬 —
    AsyncSession은 동시 실행 불가라 gather에 태우지 않는다.
    """
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g['type'] in TYPES]

    llm = LlmClient()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _condense(g: dict) -> tuple[str, list[str]]:
        # 운영과 같은 예산으로 자른다 — #81. 이 축의 multi_turn_long 8문항 중 5문항이 실제로
        # 걸린다(600 예산, 원래 12~16메시지 → 8~12). 그 5문항이 이 축의 존재 이유(긴 이력)라
        # 트리밍을 안 하면 docstring이 말하는 "히스토리 예산 트리밍"을 한 번도 안 태우게 된다.
        msgs = trim_messages_for_condense(messages_from_conversation(g.get('conversation')),
                                          settings.condense_history_budget_tokens)
        async with sem:
            if multi:
                return g['id'], await condense_to_queries(llm, g['query'], msgs)
            return g['id'], [await condense_query(llm, g['query'], msgs)]

    # 1단계: LLM 선병렬 — DB 접근 없음
    queries_map = dict(await asyncio.gather(*(_condense(g) for g in target)))

    # 2단계: 검색·채점 — 세션 직렬
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
                queries = queries_map[g['id']]
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
    if not rows:   # 채점 0 = gold resolve 전멸 (빈/잘못된 DB) — 결과 덮어쓰기 전에 중단 (#18 실사고 가드)
        raise SystemExit(f"채점 0문항 (스킵 {result['skipped']}) — DATABASE_URL이 코퍼스 있는 DB인지 확인. 결과 파일 미변경.")
    for tenant, n in result['stale'].items():
        print(f'⚠ {tenant} 라벨 노후 {n}건')

    name = 'retrieval_mt_multi.jsonl' if args.multi else 'retrieval_mt.jsonl'
    out = Path(__file__).resolve().parent / 'results' / name
    out.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n')
    print(f'\n[multi_turn 검색축 — {"멀티쿼리 on" if args.multi else "condense 단일(off)"}]')
    print(f'채점 {len(rows)}문항 (resolve 불가 스킵 {result["skipped"]})  →  {out}')
    # 타입별 분리 — multi_turn(기존 90, 비교 연속성)과 multi_turn_long(신설)을 섞지 않는다
    print(f'{"type":<18}' + ''.join(f'{m:>12}' for m in ['R@5', 'R@20', 'Hit@1', 'MRR']))
    for t in sorted({r['type'] for r in rows}):
        ss = [r['scores'] for r in rows if r['type'] == t]
        print(f'{t:<18}' + ''.join(f'{sum(s[m] for s in ss) / len(ss):>12.3f}' for m in METRICS))
    print(f'{"(전체)":<18}' + ''.join(f'{result["overall"][m]:>12.3f}' for m in METRICS))


if __name__ == '__main__':
    asyncio.run(main())
