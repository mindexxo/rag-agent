"""질의 재작성(condense) 평가 — 후속 질문을 독립 질문으로 바꿀 때
현재 질문의 조건·수치를 보존하고, 대명사를 해소하며, 이전 답변 내용을 주입하지 않는지.

condense는 LLM 호출인데 세 축(인텐트·검색·생성) 어디에도 안 잡히던 사각지대였다.
2026-07-20 실사고: "7일 맞죠?"→"14일인가요?"(전제 주입), "하자 교환"→"단순변심"(용어 치환).
정답 문장을 못 박기 어려워 '행동 기반' 규칙 채점을 쓴다:
  - must_keep       : 현재 질문의 조건·수치가 재작성 결과에 남아야 (보존)
  - must_not_contain: 이전 답변의 값이 재작성에 끼면 안 됨 (오염 방지)
  - (must_resolve는 must_keep로 통합 — 대명사가 가리키는 구체어가 채워졌나)

condense는 확률적이라 케이스마다 RUNS회 반복 → 일관성(flaky)도 함께 본다.
공백은 무시하고 매칭한다("한 달"="한달").

실행: python -m eval.condense
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from rag.llm import LlmClient
from rag.conversation import condense_query

GOLD = Path(__file__).resolve().parent / "condense_set_v1.jsonl"
RUNS = 3            # 케이스당 반복 (일관성 측정)
CONCURRENCY = 4


def _norm(s: str) -> str:
    return s.replace(" ", "")


def _passes(case: dict, rewrite: str) -> bool:
    """한 번의 재작성 결과가 케이스의 모든 단언을 만족하나."""
    r = _norm(rewrite)
    if not all(_norm(k) in r for k in case.get("must_keep", [])):
        return False
    if any(_norm(b) in r for b in case.get("must_not_contain", [])):
        return False
    return True


async def compute() -> dict:
    """condense 채점 → 요약. 반환: {accuracy, n, runs, flaky, by_category, misses}."""
    cases = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    llm = LlmClient()
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(case: dict, _i: int):
        msgs = [SimpleNamespace(role=m["role"], content=m["content"]) for m in case["conversation"]]
        async with sem:
            out = await condense_query(llm, case["query"], msgs)
        return case["id"], _passes(case, out), out

    # 케이스 × RUNS 회 전부 실행
    results = await asyncio.gather(*(_one(c, i) for c in cases for i in range(RUNS)))

    per_case = defaultdict(list)       # id -> [pass 여부...]
    sample = {}                        # id -> 마지막 재작성(오류 표시용)
    for cid, ok, out in results:
        per_case[cid].append(ok)
        if not ok:
            sample[cid] = out

    by_case = {c["id"]: c for c in cases}
    total_runs = len(cases) * RUNS
    passing_runs = sum(sum(v) for v in per_case.values())
    flaky = [cid for cid, oks in per_case.items() if 0 < sum(oks) < len(oks)]

    by_cat = defaultdict(lambda: [0, 0])   # category -> [pass_runs, total_runs]
    for cid, oks in per_case.items():
        cat = by_case[cid]["category"]
        by_cat[cat][0] += sum(oks)
        by_cat[cat][1] += len(oks)

    misses = [{"id": cid, "query": by_case[cid]["query"], "got": sample[cid],
               "pass": f"{sum(oks)}/{len(oks)}"}
              for cid, oks in per_case.items() if sum(oks) < len(oks)]

    return {
        "accuracy": passing_runs / total_runs if total_runs else 0.0,
        "n": len(cases),
        "runs": RUNS,
        "flaky": len(flaky),
        "by_category": {k: v for k, v in by_cat.items()},
        "misses": misses,
    }


async def main() -> None:
    r = await compute()
    print(f"[질의재작성(condense) 정확도]  케이스 {r['n']} × {r['runs']}회 = {r['n'] * r['runs']}런\n")
    print(f"{'category':<24}{'정확도':>14}")
    for cat, (ok, tot) in sorted(r["by_category"].items()):
        print(f"{cat:<24}{f'{ok}/{tot} ({ok / tot:.0%})':>14}")
    print(f"\n전체: {r['accuracy']:.0%}  |  불안정(flaky) 케이스: {r['flaky']}건")

    if r["misses"]:
        print("\n[미달 케이스]")
        for m in r["misses"]:
            print(f"  [{m['pass']}] {m['query']!r}\n      재작성: {m['got']}")
    else:
        print("\n전 케이스 통과.")


if __name__ == "__main__":
    asyncio.run(main())
