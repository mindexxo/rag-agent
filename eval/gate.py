"""1.5.E 게이트 거절 평가 (LLM 불필요).

근거 게이트(apply_gate)가 LLM 앞단에서 "근거 없는 질문"을 거르는지 측정.
- 거절이 정답: no_evidence, trap (has_evidence=False) → 거절정확도 ↑가 좋음
- 통과가 정답: single_fact/paraphrase/rare_lexical/multi_doc (has_evidence=True)
  → 거절하면 오거부(false reject) ↓가 좋음

리랭커·LLM 무관 (게이트는 retrieve_candidates의 top_dense_distance만 봄).
threshold(max_dense_distance)별 트레이드오프를 출력해 최적값을 정한다.

실행: python -m eval.gate
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from database import AsyncSessionLocal
from rag.retriever import retrieve_candidates, apply_gate

TENANT = "demo"
GOLD = Path("eval/gold_set_v1.jsonl")
SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

SHOULD_PASS = {"single_fact", "paraphrase", "rare_lexical", "multi_doc"}  # 통과가 정답
SHOULD_REJECT = {"no_evidence", "trap"}                                  # 거절이 정답
# 그 외(smalltalk·prompt_injection·pii·tenant_leak·multi_turn)는 참고용으로만 표시


async def main():
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]

    rows = []
    async with AsyncSessionLocal() as session:
        for g in gold:
            cands = await retrieve_candidates(session, TENANT, g["query"], top_n=20)
            # threshold별 게이트 판정: True=거절(no_evidence)
            rejects = {th: apply_gate(cands, th)[0] for th in SWEEP}
            rows.append({"type": g["type"], "rejects": rejects})

    n_pass = sum(1 for r in rows if r["type"] in SHOULD_PASS)
    n_rej = sum(1 for r in rows if r["type"] in SHOULD_REJECT)

    # ----- 핵심 트레이드오프 -----
    print(f"[게이트 threshold 트레이드오프]  (통과대상 {n_pass}문항 / 거절대상 {n_rej}문항)")
    print(f"{'th':>5}{'오거부(근거O 거부)':>22}{'거절정확도(근거X 거부)':>24}")
    for th in SWEEP:
        false_rej = sum(1 for r in rows if r["type"] in SHOULD_PASS and r["rejects"][th])
        correct_rej = sum(1 for r in rows if r["type"] in SHOULD_REJECT and r["rejects"][th])
        print(f"{th:>5}{f'{false_rej}/{n_pass} ({false_rej/n_pass:.0%})':>22}"
              f"{f'{correct_rej}/{n_rej} ({correct_rej/n_rej:.0%})':>24}")

    # ----- type별 거절률 (참고) -----
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["rejects"])
    order = ["no_evidence", "trap", "smalltalk", "prompt_injection", "pii", "tenant_leak",
             "single_fact", "paraphrase", "rare_lexical", "multi_doc", "multi_turn"]
    print("\n[type별 거절률 (각 threshold)]")
    print(f"{'type':<16}" + "".join(f"{th:>7}" for th in SWEEP))
    for t in order:
        if t not in by_type:
            continue
        rs = by_type[t]
        cells = [sum(1 for x in rs if x[th]) / len(rs) for th in SWEEP]
        print(f"{t:<16}" + "".join(f"{c:>7.0%}" for c in cells))


if __name__ == "__main__":
    asyncio.run(main())
