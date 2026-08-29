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
- no_evidence(58): 근거 없음 → **거절(=근거없음)해야** 정답. 답하면 미거부(환각 위험).
- trap(50): has_evidence=True(근거 있는 유도질문) → **답해야** 정답. 거절하면 오거부.
  (trap이 '낚여서 틀리게 답했나'는 생성축 faithfulness 몫 — 여기선 거절 여부만.)

지점3은 거절/답변 판정만 채점한다. 답변 내용 품질은 보지 않는다.

**부재단정 축이 따라 붙는다 (#76).** 위 판정이 본문을 안 보기 때문에 "제공되지 않습니다
««[]»»"가 정상 거절과 같은 점수를 받는다 — no_evidence에서 둘 다 인용 0건이라 "거절 성공"이다.
그 사각지대를 메우려고, **인용 0건인 답변만** LLM judge에 넘겨 부재를 사실로 단정했는지
따로 센다. 판정 정의·기각한 대안(정규식·RAGAS faithfulness)·self-judge 편향의 한계는
eval/absence_judge.py 모듈 docstring이 정의점이다 — 여기서는 "인용 0건이면 판정을
호출한다"만 안다. absence_rate는 accuracy와 독립이며 **회귀 감시 전용**이다.

실행: python -m eval.refusal
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from database import AsyncSessionLocal
from eval._turn_cleanup import discard_turn
from eval.generation import must_not_contain_violations, row_tenant
from rag.conversation import ensure_conversation
from eval.absence_judge import (JUDGE_PROMPT_VERSION, judge_absence, judge_llm,
                                save_audit)
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


async def _refused(tenant: str, query: str) -> tuple[bool, str, str]:
    """query를 파이프라인에 태워 (근거없음 여부, 답변 전문, route)를 반환 (단일턴, 히스토리 없음).

    answer를 함께 돌려주는 이유: 인용 0건인 답변은 부재단정 판정(2단)의 입력이 된다.
    여기서 버리면 호출부가 파이프라인을 한 번 더 태워야 한다.

    route를 함께 돌려주는 이유: 부재단정 판정은 KNOWLEDGE 경로에만 유효하다. OTHER로
    라우팅된 문항은 시스템 프롬프트 규칙 3을 아예 안 타고 _OTHER_SYSTEM_PROMPT_TEMPLATE로
    생성되므로, 그 화법("역할 밖 요청이라 안내가 어렵다")을 규칙 3 기준으로 재면 오염이다.
    게다가 OTHER는 prepared.sources가 항상 빈 목록(rag/service.py)이라 **무조건 근거없음으로
    집계**된다 — 판정 대상에서 빼지 않으면 분모가 라우팅 확률에 좌우된다.
    """
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
    return not _citations(prepared, answer), answer, prepared.route


async def compute() -> dict:
    """거절 정확성 채점 → 요약. 반환: {accuracy, n, false_answer, false_refusal, by_type, misses}."""
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g["type"] in (SHOULD_REFUSE | SHOULD_ANSWER)]
    sem = asyncio.Semaphore(CONCURRENCY)
    # judge 호출은 기존 세마포어 슬롯 안에서 이어붙인다 — GPU 부하를 새 채널로 늘리지 않는다.
    llm = judge_llm()

    async def _one(g: dict):
        async with sem:
            refused, answer, route = await _refused(row_tenant(g), g["query"])
            # 판정 게이트 2조건(route=='knowledge' · 인용 0건) — 사유는
            # eval/absence_judge.py 모듈 docstring이 정의점이다.
            absence, absence_reason, absence_error = None, None, None
            if refused and route == "knowledge":
                try:
                    v = await judge_absence(llm, g["query"], answer)
                    absence = v.label == "absence_assertion"
                    absence_reason = v.reason
                except Exception as exc:
                    # 판정 불가를 refusal_ok로 추측하지 않는다 — 지표가 실패를 숨긴다.
                    absence_error = f"{type(exc).__name__}: {exc}"
        should_refuse = g["type"] in SHOULD_REFUSE
        # 오답단정 (#95) — trap이 "답했는가"만 재던 공백을 메운다: 낚인 값(must_not_contain)이
        # 답변에 실렸는지. route 게이트는 부재단정과 같은 이유(OTHER는 규칙 3 밖) —
        # 그래서 라우팅 정확도가 이 지표의 분모를 조용히 줄일 수 있다(의도된 설계).
        misinfo = (must_not_contain_violations(answer, g.get("must_not_contain", []))
                   if route == "knowledge" else None)
        return {"id": g["id"], "type": g["type"], "query": g["query"], "answer": answer,
                "route": route, "refused": refused, "ok": refused == should_refuse,
                "misinfo": misinfo,
                "absence": absence, "absence_reason": absence_reason,
                "absence_error": absence_error}

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

    # 부재단정 (#76) — 인용 0건 답변 중 부재를 사실로 단정한 비율.
    # 이 축의 accuracy와 독립이다: 부재단정은 no_evidence에서 "거절 성공"으로 집계되므로
    # accuracy를 아무리 봐도 안 보인다. 그 사각지대가 이 지표의 존재 이유다.
    judged = [r for r in rows if r["absence"] is not None]
    errors = sum(1 for r in rows if r["absence_error"] is not None)
    assertions = [r for r in judged if r["absence"]]

    # 흔들림 관측 (#76 리뷰) — 판정이 항목당 1콜이라 비결정성이 조용히 지나간다.
    # eval/other_eval.py는 전 케이스를 RUNS=3으로 돌려 flaky를 지표로 내는데, 여기서 그걸
    # 그대로 하면 judge 호출이 3배가 된다. 대신 **부재단정으로 판정된 행만** 1회 재판정한다 —
    # 이 축에서 값을 움직이는 것은 분자뿐이고, 그 행이 재판정에서 뒤집히면 그 회차의 수치를
    # 믿을 수 없다는 뜻이다. 실측 근거: 같은 스위트에서 1건 → 0건으로 뒤집힌 회차가 있었다.
    flaky = []
    for r in assertions:
        try:
            again = await judge_absence(llm, r["query"], r["answer"])
            if again.label != "absence_assertion":
                flaky.append(r["id"])
        except Exception:
            flaky.append(r["id"])       # 재판정 실패도 신뢰할 수 없다는 신호다

    audit = save_audit([
        {"id": r["id"], "type": r["type"], "route": r["route"], "query": r["query"],
         "answer": r["answer"], "absence": r["absence"], "reason": r["absence_reason"],
         "error": r["absence_error"], "flaky": r["id"] in flaky,
         "judge_prompt_version": JUDGE_PROMPT_VERSION}
        for r in rows if r["absence"] is not None or r["absence_error"] is not None
    ])

    # 오답단정 (#95) — misinfo가 None이 아닌 행(=must_not_contain 보유 + knowledge 라우팅)만 분모.
    misinfo_target = [r for r in rows if r["misinfo"] is not None]
    misinfo_violated = [r for r in misinfo_target if r["misinfo"]]

    return {
        "accuracy": sum(r["ok"] for r in rows) / len(rows) if rows else 0.0,
        "n": len(rows),
        "misinfo_rate": len(misinfo_violated) / len(misinfo_target) if misinfo_target else None,
        "misinfo_n": len(misinfo_target),
        "misinfo_violated_n": len(misinfo_violated),
        "misinfo_misses": [{"id": r["id"], "query": r["query"], "violated": r["misinfo"]}
                           for r in misinfo_violated],
        "false_answer": false_answer, "false_answer_n": len(ne),
        "false_refusal": false_refusal, "false_refusal_n": len(tr),
        "by_type": {k: v for k, v in by_type.items()},
        "misses": [r for r in rows if not r["ok"]],
        "absence_rate": len(assertions) / len(judged) if judged else None,
        "absence_judged_n": len(judged),
        "absence_assertion_n": len(assertions),
        "absence_judge_errors": errors,
        "absence_flaky": len(flaky),
        "judge_prompt_version": JUDGE_PROMPT_VERSION,   # 감사 로그와 같은 키 이름 — 조인 가능하게
        "absence_audit": str(audit),
        "absence_misses": [{"id": r["id"], "type": r["type"], "query": r["query"],
                            "reason": r["absence_reason"], "flaky": r["id"] in flaky}
                           for r in assertions],
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

    rate, n = r["absence_rate"], r["absence_judged_n"]
    print(f"\n[부재단정 {r['judge_prompt_version']}]  KNOWLEDGE·인용 0건 {n}건 중 "
          f"{r['absence_assertion_n']}건 = {'―' if rate is None else f'{rate:.1%}'}"
          f"  ← 위 정확도에는 안 잡히는 축")
    # 분모를 같이 찍는 이유: n이 작아 1건이 곧 1.7%다(회귀 임계 0.01보다 크다).
    # 비율만 보면 한 건 흔들림을 추세로 오독한다.
    if n:
        print(f"  1건 = {1 / n:.1%}  (회귀 임계 {'0.010'})")
    if r["absence_judge_errors"]:
        print(f"  판정 불가 {r['absence_judge_errors']}건 (분모 제외)")
    if r["absence_flaky"]:
        print(f"  재판정에서 뒤집힘 {r['absence_flaky']}건 ← 이 회차 수치를 믿지 말 것")
    for m in r["absence_misses"][:20]:
        flag = " [flaky]" if m["flaky"] else ""
        print(f"  [{m['type']}]{flag} {m['query'][:44]!r} — {(m['reason'] or '')[:60]}")
    print(f"  감사 로그: {r['absence_audit']}")

    if r["misinfo_n"]:
        print(f"\n[오답단정 (#95)]  must_not_contain 보유·knowledge 라우팅 {r['misinfo_n']}건 중 "
              f"{r['misinfo_violated_n']}건 = {r['misinfo_rate']:.1%}  ← 낚인 값이 답변에 실림")
        for m in r["misinfo_misses"][:20]:
            print(f"  {m['query'][:44]!r} — 검출: {m['violated']}")

    if r["misses"]:
        print("\n[오판정]")
        for m in r["misses"][:20]:
            verdict = "거절함" if m["refused"] else "답변함"
            print(f"  [{m['type']}] {m['query'][:50]!r} → {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
