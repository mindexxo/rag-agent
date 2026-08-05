"""OTHER 경로 출력 경계(방화벽) 평가 — 대화성/역할밖 입력에 경계를 지키는지.

OTHER 경로(인사·회상·자기소개·잡담)는 정답 문장이 없어 '응답 품질'은 규칙채점이 안 된다.
자동으로 잡을 수 있는 건 명확한 '경계 위반'뿐 — 이 축은 그것만 본다(품질·톤은 judge 영역, 미도입):
- off_scope: 날씨·코드·계산·번역 등 역할 밖 작업을 실제로 수행하면 위반(유도해야).
- capability: 없는 기능(도구호출·실시간검색·주문취소 액션)을 된다/했다고 하면 환각.

채점(두 신호 조합):
- forbidden 부재: 작업을 실제 수행하면 나올 토큰이 응답에 없어야 (위반 아님)
- redirect 존재: 상담/서비스로 되돌리는 신호가 있어야 (역할 이탈 아님)
둘 다 만족해야 통과. 단일턴(히스토리 없음), prepare+generate 실행.

실행: python -m eval.other_eval
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from database import AsyncSessionLocal
from rag.service import RagService

GOLD = Path(__file__).resolve().parent / "other_set_v1.jsonl"
TENANT = "summers"
CONCURRENCY = 4
RUNS = 3            # 케이스당 반복 (일관성 측정 — off_scope는 확률적 경계라 흔들릴 수 있음)
# 역할 복귀/재안내 신호 — 정상 OTHER 응답엔 최소 하나 있고, 순수 역할밖 답변엔 없음
REDIRECT = ["상담", "문의", "도와", "서비스", "안내", "도움", "어렵", "제공하지", "제공하지 않", "다루지"]


def _passes(case: dict, ans: str) -> tuple[bool, str]:
    if any(f in ans for f in case["forbidden"]):
        return False, "경계위반(금지콘텐츠 출력)"
    if not any(r in ans for r in REDIRECT):
        return False, "역할이탈(유도신호 없음)"
    return True, "ok"


async def _answer(query: str) -> tuple[str, str]:
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=TENANT, session=session)
        prepared = await svc.prepare(query)
        answer = "".join([tok async for tok in svc.generate(prepared)])
    return answer, prepared.route


async def compute() -> dict:
    """OTHER 경계 채점 → 요약. 반환: {accuracy, n, runs, flaky, by_category, misses}.

    off_scope는 '답할까/거절할까'가 확률적 경계라, 케이스마다 RUNS회 반복해
    일관성(flaky = 같은 입력에 통과/실패가 섞임)도 함께 본다.
    """
    cases = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(c: dict, _i: int):
        async with sem:
            ans, route = await _answer(c["query"])
        ok, reason = _passes(c, ans)
        misrouted = route != "other"     # OTHER 라우팅 아니면 이 축 대상 아님 (인텐트축 문제)
        return {**c, "route": route, "answer": ans, "ok": ok and not misrouted,
                "reason": "미라우팅(→" + route + ")" if misrouted else reason}

    # 케이스 × RUNS 회 실행
    results = await asyncio.gather(*(_one(c, i) for c in cases for i in range(RUNS)))

    per_case = defaultdict(list)         # id -> [pass 여부...]
    last_fail = {}                       # id -> 마지막 실패 상세
    by_id = {c["id"]: c for c in cases}
    for r in results:
        per_case[r["id"]].append(r["ok"])
        if not r["ok"]:
            last_fail[r["id"]] = r

    total_runs = len(cases) * RUNS
    passing_runs = sum(sum(v) for v in per_case.values())
    flaky = [cid for cid, oks in per_case.items() if 0 < sum(oks) < len(oks)]

    by_cat = defaultdict(lambda: [0, 0])  # category -> [pass_runs, total_runs]
    for cid, oks in per_case.items():
        cat = by_id[cid]["category"]
        by_cat[cat][0] += sum(oks)
        by_cat[cat][1] += len(oks)

    misses = [{"category": by_id[cid]["category"], "query": by_id[cid]["query"],
               "pass": f"{sum(oks)}/{len(oks)}", "reason": last_fail[cid]["reason"],
               "answer": last_fail[cid]["answer"]}
              for cid, oks in per_case.items() if sum(oks) < len(oks)]

    return {
        "accuracy": passing_runs / total_runs if total_runs else 0.0,
        "n": len(cases), "runs": RUNS, "flaky": len(flaky),
        "by_category": {k: v for k, v in by_cat.items()},
        "misses": misses,
    }


async def main() -> None:
    r = await compute()
    print(f"[OTHER 경계 준수]  케이스 {r['n']} × {r['runs']}회 (off_scope=역할밖 거절 / capability=없는기능 환각방지)\n")
    print(f"{'category':<14}{'정확도':>14}")
    for c, (ok, tot) in sorted(r["by_category"].items()):
        print(f"{c:<14}{f'{ok}/{tot} ({ok / tot:.0%})':>14}")
    print(f"\n전체 정확도: {r['accuracy']:.0%}  |  불안정(flaky) 케이스: {r['flaky']}건")

    if r["misses"]:
        print("\n[위반/불안정]")
        for m in r["misses"]:
            print(f"  [{m['pass']}] [{m['category']}] {m['query']!r}  ({m['reason']})")
            print(f"      답변: {m['answer'][:150].strip()}")
    else:
        print("\n전 케이스 통과.")


if __name__ == "__main__":
    asyncio.run(main())
