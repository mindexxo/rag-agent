"""거절 정확성 평가 — 시스템이 '맞게 거절하고 맞게 답하는지'.

거절은 두 경로에서 난다: 근거 게이트(apply_gate) + LLM 자체 거절(프롬프트 규칙3).
운영은 게이트를 거의 꺼두고(max_dense_distance=inf) LLM 자체 거절에 의존하므로,
게이트만 보는 옛 eval/gate.py로는 부족 — prepare()+generate() 전체를 태워
'최종 거절 여부'를 본다. 프롬프트에 민감하니 즉석 실행(항상 현재 프롬프트 기준).

**측정 대상이 #61에서 넓어졌다 — 이전 결과와 비교할 때 유의.** 판정이 거절 문구
부분일치에서 "실인용 0건"으로 바뀌면서, 위 두 경로 말고 **제3의 경우까지 여기 들어온다**:
"근거를 못 댔는데 거절 문구도 없이 답한" 답변(부재를 단정하거나 근거 없이 확신하는 유형).
그래서 이 축의 이름은 '거절'이지만 실제로 재는 것은 '근거를 못 댄 비율'이다 —
SHOULD_REFUSE/SHOULD_ANSWER·_refused의 '거절'도 그 뜻으로 읽을 것.
규약의 정의점은 rag/citation_tail.py 모듈 docstring.

대상 (gold_set_v2, 모두 단일턴):
- no_evidence(60): 근거 없음 → **거절(=근거없음)해야** 정답. 답하면 미거부(환각 위험).
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
from eval._turn_cleanup import discard_turn
from eval.generation import row_tenant
from rag.conversation import ensure_conversation
from rag.citation_tail import TailSplitter, resolve_citations
from rag.service import PreparedRag, RagService

GOLD = Path(__file__).resolve().parent / "gold_set_v2.jsonl"
CONCURRENCY = 4
SHOULD_REFUSE = {"no_evidence"}          # 거절이 정답
SHOULD_ANSWER = {"trap"}                 # 답변이 정답 (근거 있는 유도)


def _citations(prepared: PreparedRag, answer: str) -> list:
    """답변의 실인용 — 운영 두 경로와 같은 방식으로 구한다 (#61).

    옛 판정은 `is_refusal(answer)`(거절 문구 부분일치) 한 줄이었다. 폐기 사유·실측은
    rag/citation_tail.py 모듈 docstring(단일 정의점) — 여기서는 그 규약을 적용만 한다.

    경로가 둘인 이유는 운영과 같다. 즉시 경로(캐시 히트·근거없음·차단)와 OTHER는 출처
    꼬리 메커니즘 자체가 없어 prepared.sources가 그대로 답이다 —
    rag/streaming.immediate_stream과 같은 근거다. knowledge 실생성만 꼬리를 걷어낸다.

    완성된 문자열을 한 번에 feed하는 것은 eval/generation.citation_accuracy(v4)와 같은
    배관이다 — 여기서 직접 rsplit하면 운영의 오탐 복구를 잃는다.
    """
    if prepared.resolved_answer is not None or prepared.route != "knowledge":
        return list(prepared.sources)
    splitter = TailSplitter()
    splitter.feed(answer)
    splitter.finish()
    # 후보 파생은 PreparedRag.citation_candidates 한 곳 (#65) — 운영 두 호출부와 같은 통로다.
    # 여기서 직접 조립하면 이 PR이 없애려던 '우연히 같은' 상태가 eval에만 남는다.
    return resolve_citations(splitter.tail_raw, *prepared.citation_candidates)


async def _refused(tenant: str, query: str) -> bool:
    """query를 파이프라인에 태워 최종 답변이 근거없음(=거절)인지 반환 (단일턴, 히스토리 없음)."""
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant, session=session)
        # 대화를 **미리** 만들고 그 id를 넘긴다 (#72). prepare()에 맡기면 그 안에서 실패했을 때
        # (판단 실패·검색 오류 — 폴백을 걷어낸 뒤로 잦다) 방금 커밋된 대화의 id를 알 길이 없어
        # 정리가 통째로 새고, 이 헬퍼가 막으려던 고아 누적이 그대로 재현된다.
        conversation = await ensure_conversation(session, tenant, None)
        await session.commit()
        try:
            prepared = await svc.prepare(query, conversation_id=conversation.id)
            answer = "".join([tok async for tok in svc.generate(prepared)])
        finally:
            await discard_turn(session, tenant, conversation.id)
    return not _citations(prepared, answer)


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
