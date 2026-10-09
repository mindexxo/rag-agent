"""오정보(must_not_contain) 2단계 판정 (#214) — 문자열 후보 → LLM이 '단정'인지 '부정·정정'인지 확인.

1단계 `eval.generation.must_not_contain_violations`는 금지 문구가 답변에 **글자로 들어 있는지**만 본다.
그래서 정답인 답을 위반으로 찍는다 — 실측(거절축 trap 60행, 2026-10-09): 위반 4건 중 3건이 오탐.
  · "등급 쿠폰은 4만원 이상 주문에만 사용할 수 있는 것은 **아닙니다**"            (부정)
  · "결연 후원은 만원부터 시작하는 일반 정기후원과 **달리** 45,000원"            (대조·정정)
  · "60℃ 이상이어야 정상 판정이 됩니다. **다만** … ST-350은 55℃"              (단정 뒤 자기정정)
진짜 오정보(연차 "인정되지 **않**으며")는 금지 문구 자체에 부정어가 있어, 부정 표지 휴리스틱은 이것까지 지운다
→ 기각. 골드를 단언문으로 다시 쓰는 규약(must_not_contain_violations docstring)도 세 번째 사례(단정 뒤 정정)는
못 가른다. 뜻을 봐야 하므로 **걸린 행에만** LLM 판정을 얹는다 — 안 걸린 행은 지금처럼 결정적이다.

판정 실패(LlmJudgmentFailed·예외)는 **위반 유지**(fail-closed)다 — 판정 못 한 것을 '정상'으로 추측하면 지표가
실패를 숨긴다(absence_judge와 같은 원칙). 사유·오류는 호출부가 결과 행에 그대로 남긴다(감사용).
"""
from typing import Literal

from pydantic import BaseModel

from rag.llm import LlmClient
from rag.llm_schemas import acomplete_validated

JUDGE_PROMPT_VERSION = "v1"


class MisinfoJudgment(BaseModel):
    """label에 기본값을 두지 않는다 — absence_judge.AbsenceJudgment와 같은 이유(스키마 required 보장)."""
    label: Literal["asserted", "negated_or_corrected"]
    reason: str | None = None


_JUDGE_SYSTEM = """당신은 한국어 상담 답변을 검사하는 판정기입니다. 답변 하나와 '금지 주장' 하나를 받아 두 갈래로 분류하십시오.
[asserted] 답변이 금지 주장을 **사실로 단정**해 안내한 경우. 그 주장이 답의 결론으로 남아 있습니다.
  예: 금지 주장 "연차 사용일은 근로시간으로 인정되지 않는다" / 답변 "예, 빠집니다. 연차 사용일은 근로시간으로 인정되지 않으며…"
[negated_or_corrected] 답변이 금지 주장을 **부정·정정**하거나, 대조로 끌어와 반대 결론을 내린 경우.
  예: 답변 "…4만원 이상 주문에만 사용할 수 있는 것은 아닙니다. 2만원 이상이면 됩니다."
  예: 답변 "결연 후원은 만원부터 시작하는 일반 후원과 달리 45,000원 정액입니다."
  예: 답변 "60℃ 이상이어야 정상입니다. 다만 ST-350은 350ml 이하라 55℃ 기준이 적용됩니다."  ← 단정 뒤 스스로 고침
판정 규칙:
- 금지 주장의 글자가 들어 있는지가 아니라, **답의 최종 결론**이 그 주장과 같은지로 가르십시오.
- 단정했다가 "다만/하지만/그러나" 뒤에서 고쳤으면 최종 결론을 따릅니다 → negated_or_corrected.
- 금지 주장이 결론에 그대로 남아 있고 정정이 없으면 asserted입니다.
JSON 한 줄로만 답하십시오: {"label": "...", "reason": "판정 근거 한 줄"}"""


def _build_messages(answer: str, phrase: str) -> list[dict]:
    return [{"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": f"금지 주장: {phrase}\n\n답변:\n{answer}"}]


async def judge_misinfo(llm: LlmClient, answer: str, phrase: str) -> MisinfoJudgment:
    """후보 하나를 판정. 실패는 LlmJudgmentFailed로 전파 — 호출부(confirm_violations)가 위반 유지로 집계한다."""
    return await acomplete_validated(llm, _build_messages(answer, phrase), MisinfoJudgment)


async def confirm_violations(llm: LlmClient, answer: str, candidates: list[str]) -> tuple[list[str], list[dict]]:
    """1단계 후보 → (확정 위반 목록, 감사 행). 빈 후보면 ([], []) — LLM을 부르지 않는다.

    감사 행: {"phrase", "label", "reason", "error"}. error가 있으면 label은 None이고 위반으로 **유지**된다(fail-closed).
    """
    confirmed: list[str] = []
    audit: list[dict] = []
    for phrase in candidates:
        try:
            v = await judge_misinfo(llm, answer, phrase)
            audit.append({"phrase": phrase, "label": v.label, "reason": v.reason, "error": None})
            if v.label == "asserted":
                confirmed.append(phrase)
        except Exception as exc:   # LlmJudgmentFailed·네트워크 — 판정 불가는 위반 유지
            audit.append({"phrase": phrase, "label": None, "reason": None, "error": f"{type(exc).__name__}: {exc}"})
            confirmed.append(phrase)
    return confirmed, audit


# 판정기 회귀 고정 — 거절축 실측에서 1단계가 틀린 케이스 전수(2026-10-09) + 진짜 오정보 1건.
FIXTURES: list[dict] = [
    {"id": "negated", "expected": "negated_or_corrected", "phrase": "4만원 이상 주문에만 사용",
     "answer": "등급 쿠폰은 주문금액 2만원 이상인 주문에만 적용할 수 있으며, 무료배송 기준(4만원)과는 별도로 적용됩니다. "
               "따라서 등급 쿠폰은 4만원 이상 주문에만 사용할 수 있는 것은 아닙니다."},
    {"id": "contrast", "expected": "negated_or_corrected", "phrase": "결연 후원은 만원부터",
     "answer": "해외아동결연 후원은 월 45,000원 정액으로 하며, 금액 변경이 불가능합니다. "
               "따라서 결연 후원은 만원부터 시작하는 일반 정기후원과 달리, 최소 금액이 월 45,000원으로 고정되어 있습니다."},
    {"id": "assert_then_correct", "expected": "negated_or_corrected", "phrase": "60℃ 이상이어야 정상",
     "answer": "예, ST-350 데일리 모델의 경우 6시간 후 온도가 60℃ 이상이어야 정상 판정이 됩니다. "
               "다만, 이는 500ml 기준이며, ST-350 데일리는 350ml 이하 제품에 해당하므로 55℃ 이상이면 정상으로 판정됩니다."},
    {"id": "asserted_true_positive", "expected": "asserted", "phrase": "근로시간으로 인정되지 않",
     "answer": "예, 연차를 사용한 날은 근로시간에서 빠집니다. - 연차 사용일은 근로시간으로 인정되지 않으며, "
               "주/월 단위 근로시간 합산 시 제외됩니다."},
]


async def selftest(runs: int = 3) -> dict:
    """FIXTURES를 runs회 판정해 기대와 비교 — 판정기 흔들림 관측(absence_judge.selftest와 같은 idiom)."""
    from eval.absence_judge import judge_llm
    llm = judge_llm()
    out = []
    for f in FIXTURES:
        labels = []
        for _ in range(runs):
            try:
                labels.append((await judge_misinfo(llm, f["answer"], f["phrase"])).label)
            except Exception as exc:
                labels.append(f"error:{type(exc).__name__}")
        out.append({"id": f["id"], "expected": f["expected"], "labels": labels,
                    "ok": all(l == f["expected"] for l in labels)})
    return {"runs": runs, "fixtures": out, "all_ok": all(o["ok"] for o in out)}


if __name__ == "__main__":
    import asyncio, json
    print(json.dumps(asyncio.run(selftest()), ensure_ascii=False, indent=1))
