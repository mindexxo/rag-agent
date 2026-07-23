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
    """OTHER 경계 채점 → 요약. 반환: {accuracy, n, by_category, misses}."""
    cases = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(c: dict):
        async with sem:
            ans, route = await _answer(c["query"])
        ok, reason = _passes(c, ans)
        # OTHER로 라우팅 안 됐으면 이 축의 대상이 아님 — 별도 표시(인텐트축 문제)
        misrouted = route != "other"
        return {**c, "route": route, "answer": ans, "ok": ok and not misrouted,
                "reason": "미라우팅(→" + route + ")" if misrouted else reason}

    rows = await asyncio.gather(*(_one(c) for c in cases))
    by_cat = defaultdict(lambda: [0, 0])
    for r in rows:
        by_cat[r["category"]][0] += r["ok"]
        by_cat[r["category"]][1] += 1

    return {
        "accuracy": sum(r["ok"] for r in rows) / len(rows) if rows else 0.0,
        "n": len(rows),
        "by_category": {k: v for k, v in by_cat.items()},
        "misses": [r for r in rows if not r["ok"]],
    }


async def main() -> None:
    r = await compute()
    print(f"[OTHER 경계 준수]  {r['n']}문항 (off_scope=역할밖 거절 / capability=없는기능 환각방지)\n")
    print(f"{'category':<14}{'정확도':>14}")
    for c, (ok, tot) in sorted(r["by_category"].items()):
        print(f"{c:<14}{f'{ok}/{tot} ({ok / tot:.0%})':>14}")
    print(f"\n전체 정확도: {r['accuracy']:.0%}")

    if r["misses"]:
        print("\n[위반]")
        for m in r["misses"]:
            print(f"  [{m['category']}] {m['query']!r}  ({m['reason']})")
            print(f"      답변: {m['answer'][:150].strip()}")
    else:
        print("\n전 케이스 통과.")


if __name__ == "__main__":
    asyncio.run(main())
