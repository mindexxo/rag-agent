"""거절 정확성 평가 — 시스템이 '맞게 거절하고 맞게 답하는지'.

거절은 두 경로에서 난다: 근거 게이트(apply_gate) + LLM 자체 거절(프롬프트 규칙3).
운영은 게이트를 거의 꺼두고(max_dense_distance=inf) LLM 자체 거절에 의존하므로,
게이트만 보는 옛 eval/gate.py로는 부족 — prepare()+generate() 전체를 태워
'최종 거절 여부'를 본다. 프롬프트에 민감하니 즉석 실행(항상 현재 프롬프트 기준).

대상 (gold_set_v2, 모두 단일턴):
- no_evidence(60): 근거 없음 → **거절해야** 정답. 답하면 미거부(환각 위험).
- trap(48): has_evidence=True(근거 있는 유도질문) → **답해야** 정답. 거절하면 오거부.
  (trap이 '낚여서 틀리게 답했나'는 생성축 faithfulness 몫 — 여기선 거절 여부만.)

지점3은 거절/답변 판정만 채점한다. 답변 내용 품질은 보지 않는다.

실행: python -m eval.refusal
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from database import AsyncSessionLocal
from eval.generation import row_tenant
from rag.service import RagService
from rag.prompts import is_refusal

GOLD = Path(__file__).resolve().parent / "gold_set_v2.jsonl"
CONCURRENCY = 4
SHOULD_REFUSE = {"no_evidence"}          # 거절이 정답
SHOULD_ANSWER = {"trap"}                 # 답변이 정답 (근거 있는 유도)


async def _refused(tenant: str, query: str) -> bool:
    """query를 파이프라인에 태워 최종 답변이 거절인지 반환 (단일턴, 히스토리 없음)."""
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant, session=session)
        prepared = await svc.prepare(query)          # 게이트·인텐트·검색 반영
        answer = "".join([tok async for tok in svc.generate(prepared)])
    return is_refusal(answer)


async def compute() -> dict:
    """거절 정확성 채점 → 요약. 반환: {accuracy, n, false_answer, false_refusal, by_type, misses}."""
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g["type"] in (SHOULD_REFUSE | SHOULD_ANSWER)]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(g: dict):
        async with sem:
            refused = await _refused(row_tenant(g), g["query"])
        should_refuse = g["type"] in SHOULD_REFUSE
        return {"id": g["id"], "type": g["type"], "query": g["query"],
                "refused": refused, "ok": refused == should_refuse}

    rows = await asyncio.gather(*(_one(g) for g in target))

    by_type = defaultdict(lambda: [0, 0])   # type -> [ok, total]
    for r in rows:
        by_type[r["type"]][0] += r["ok"]
        by_type[r["type"]][1] += 1

    # 방향별 오류율
    ne = [r for r in rows if r["type"] in SHOULD_REFUSE]
    tr = [r for r in rows if r["type"] in SHOULD_ANSWER]
    false_answer = sum(1 for r in ne if not r["refused"])     # 거절해야 하는데 답함
    false_refusal = sum(1 for r in tr if r["refused"])        # 답해야 하는데 거절함

    return {
        "accuracy": sum(r["ok"] for r in rows) / len(rows) if rows else 0.0,
        "n": len(rows),
        "false_answer": false_answer, "false_answer_n": len(ne),
        "false_refusal": false_refusal, "false_refusal_n": len(tr),
        "by_type": {k: v for k, v in by_type.items()},
        "misses": [r for r in rows if not r["ok"]],
    }


async def main() -> None:
    r = await compute()
    print(f"[거절 정확성]  대상 {r['n']}문항 (no_evidence=거절정답 / trap=답변정답)\n")
    print(f"{'type':<16}{'정확도':>14}")
    for t, (ok, tot) in sorted(r["by_type"].items()):
        print(f"{t:<16}{f'{ok}/{tot} ({ok / tot:.0%})':>14}")
    print(f"\n전체 정확도: {r['accuracy']:.0%}")
    print(f"  미거부(근거X인데 답함) : {r['false_answer']}/{r['false_answer_n']}  ← 환각 위험")
    print(f"  오거부(근거O인데 거절) : {r['false_refusal']}/{r['false_refusal_n']}  ← 유용성 손해")

    if r["misses"]:
        print("\n[오판정]")
        for m in r["misses"][:20]:
            verdict = "거절함" if m["refused"] else "답변함"
            print(f"  [{m['type']}] {m['query'][:50]!r} → {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
