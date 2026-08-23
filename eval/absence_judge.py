"""부재단정 판정 — LLM judge (#76).

**정의(v1).** 답변이 문서에 근거가 없는 사안을 두고 실제 세계의 사실을 확정적으로
단정하면 부재단정이다 — "제공되지 않습니다"·"불가합니다"·"지원하지 않습니다"처럼.
반대로 **문서·시스템의 상태만 서술**하는 것은 정상 거절이다 — "확인할 수 없습니다",
"문서에 언급이 없습니다", "명시된 바가 없습니다". 이 경계가 곧 시스템 프롬프트
규칙 3의 원리이고("제공되지 않는다" ≠ "확인할 수 없다"), 그래서 판정 기준을 새로
창작하지 않고 생성 프롬프트와 같은 개념을 쓴다 — 정상형 대조 예시로
rag.prompt_texts.NO_EVIDENCE_ANSWER를 import해 박는다(문구가 바뀌면 예시가 따라간다).

**왜 필요한가.** #61이 거절 판정을 "실인용 0건"으로 바꿨다(정의점은
rag/citation_tail.py). 그 규약은 본문을 안 보기 때문에 아래 둘을 구별하지 못한다:

    해당 내용은 제공된 문서에서 확인할 수 없습니다.   ««[]»»   ← 정상 거절
    정책 범위에 포함되지 않습니다.                    ««[]»»   ← 부재단정

둘 다 인용 0건이라 no_evidence 문항에서 똑같이 "거절 성공"으로 집계된다. #62에서
프롬프트 축약 후보 8개를 측정하는 동안 이 사각지대가 매번 판단을 흐렸다 — 거절
정확도가 1.5pp 오른 후보(H13)가 실제로는 부재단정이 2배였다. 그래서 기각 근거를
전부 사람이 원문을 읽어 만들었다. 이 판정기가 그 수동 판정을 대신한다.

**판정 범위는 인용 0건 부분집합뿐이다.** 인용이 있으면 호출하지 않는다 — 그 결정은
eval/refusal.py의 몫이다. 이 2단 구조가 정규식으로는 원리적으로 못 가르던 케이스를
구조적으로 해결한다: "개봉 상품은 반품 불가"는 근거를 인용했으면 정상 답변이고
안 했으면 부재단정인데, 표면 패턴은 둘이 똑같다. 인용 개수가 그걸 가른다.

**정규식을 쓰지 않는 이유.** 이 프로젝트가 이미 실패했다(실측·사유는
rag/citation_tail.py docstring). #62 측정 중 만든 정규식이 밟은 결함은 아래 FIXTURES가 전수다.

**RAGAS faithfulness를 쓰지 않는 이유.** 답변을 사실 주장으로 분해한 뒤 각각의
근거 여부를 재는데, 주장이 추출되지 않으면(statements == []) nan을 반환하고
eval/ragas_eval.py의 _mean이 그걸 걸러낸다. 짧은 거절형 답변이 정확히 그 경우다 —
벌점이 아니라 미측정이다. 우리 판정 대상이 바로 그 짧은 거절형이라 같은 구조를
쓰면 같은 실패가 재발한다. 그래서 문장 분해 없이 턴 단위 단일 판정으로 간다.

**judge에 검색 컨텍스트를 주지 않는다.** "이 답변이 근거를 반영했는가"는 1단(인용
개수)이 이미 판정했다. 2단이 묻는 것은 "이 화법이 부재를 단정하는 형태인가"라는
순수 수사적 분류다. 컨텍스트를 주면 judge가 사실관계까지 섞어 판단해 측정 대상이
흐려진다.

**self-judge 편향 — 해소가 아니라 문서화다.** 기본 judge는 답변을 생성한 그 모델
(사내 vLLM)이다. 그래서 이 지표는 **회귀 감시·상대 비교 전용**이고, "우리 부재단정률은
X%"라는 절대값 주장에는 쓸 수 없다. eval/ragas_eval.py가 같은 이유로 같은 단서를 달았다.

⚠ 그 편향이 **이 판정기의 주 용도에서 가장 강하다.** 규칙 3의 문구를 바꾸면 모델의
단정·완곡 화법 자체가 변하는데, judge가 같은 모델이라 "부재단정과 정상 거절의 경계"를
읽는 기준도 같이 흔들린다 — 편향이 일정한 오프셋이라 전후 비교에서 상쇄된다는 가정이
여기서 깨진다. **규칙 3에 인접한 프롬프트 변경의 결과는 이 수치만으로 판단하지 말고
외부 judge나 사람 판정으로 교차 확인해야 한다.** 코드로 막을 수 없는 지점이다.
외부 judge 경로(ABSENCE_JUDGE=openai)는 자리만 잡아뒀고 아직 미구현이다.

**버저닝.** JUDGE_PROMPT_VERSION을 판정 결과·감사 로그·요약에 모두 스탬프한다.
판정 기준이나 프롬프트가 바뀌면 올리고, 이전 버전의 absence_rate와 비교하지 말 것 —
eval/generation.py::citation_accuracy의 v2~v5 규약과 같다.

**낡을 지점(코드가 막을 수 없다).** 규칙 3이 다시 개정되면 "부재단정"의 경계 자체가
바뀔 수 있는데 이 프롬프트가 자동으로 따라가지 않는다. NO_EVIDENCE_ANSWER import는
문구 변경만 흡수하고 기준 재편은 못 잡는다. 규칙 3을 건드리는 작업은 이 판정 정의를
같이 재검토해야 한다 — rag/prompt_texts.py의 ⚠ 주석에 그 전제를 적어뒀다.

실행: python -m eval.absence_judge        # FIXTURES 자체검증 (판정기를 고칠 때마다)
"""
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from rag.llm import LlmClient
from rag.llm_schemas import acomplete_validated
from rag.prompt_texts import NO_EVIDENCE_ANSWER

JUDGE_PROMPT_VERSION = "v1"
RESULTS = Path(__file__).resolve().parent / "results"
SELFTEST_RUNS = 3          # 판정 흔들림 관측 — eval/other_eval.py·condense.py와 같은 idiom


class AbsenceJudgment(BaseModel):
    """label에 기본값을 주지 않는다 — 구조를 결정하는 필드에 기본값이 있으면 json 스키마에서
    required가 빠져 모델이 생략해도 합법이 된다(사유는 rag/llm_schemas.py 모듈 docstring).
    reason은 원래 없을 수 있는 값이라 옵셔널이 맞다."""
    label: Literal["absence_assertion", "refusal_ok"]
    reason: str | None = None


# 판정 프롬프트. 위 모듈 docstring의 정의를 그대로 지시문으로 옮긴 것 — 정의가 두 곳에서
# 갈리지 않게 여기서는 새 기준을 만들지 않는다. 정상형 예시는 상수를 보간한다.
_JUDGE_SYSTEM = f"""당신은 한국어 상담 답변을 검사하는 판정기입니다. 답변 하나를 두 갈래로 분류하십시오.

[absence_assertion] 실제 세계의 사실을 확정적으로 단정한 경우.
  예: "해외 배송은 제공되지 않습니다." / "오프라인 반품은 불가합니다."
      "정책 범위에 포함되지 않습니다." / "그 기능은 지원하지 않습니다."
      "반품은 어렵습니다."

[refusal_ok] 문서·자료의 상태만 서술한 경우. 사실을 단정하지 않았습니다.
  예: "{NO_EVIDENCE_ANSWER}"
      "문서에 해외 배송에 대한 언급은 없습니다."
      "명시된 바가 없습니다." / "관련 규정은 제공된 문서에 명시되어 있지 않습니다."

판정 규칙:
- **핵심 문장**을 보십시오. 답변이 인사·사과("죄송합니다", "안녕하세요")로 시작하면 그 다음 문장이 핵심입니다.
- 핵심 문장이 이미 단정이면, 뒤에 "확인이 필요합니다" 같은 완화가 붙어도 absence_assertion입니다.
- 반대로 핵심 문장이 문서 상태 서술이면, 뒤에서 가능성을 언급해도 refusal_ok입니다.
- "없습니다"는 무엇이 없다고 했는지로 가르십시오 — "규정이 없습니다"(자료 상태)는 refusal_ok,
  "그런 서비스는 없습니다"(사실)는 absence_assertion입니다.

JSON 한 줄로만 답하십시오: {{"label": "...", "reason": "판정 근거 한 줄"}}"""


def _build_messages(query: str, answer: str) -> list[dict]:
    """검색 컨텍스트를 넣지 않는다 — 사유는 모듈 docstring(순수 화법 분류)."""
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": f"질문: {query}\n\n답변:\n{answer}"},
    ]


def judge_llm() -> LlmClient:
    """판정용 클라이언트. 기본은 생성과 같은 사내 vLLM(self-judge) —
    ABSENCE_JUDGE=openai면 외부 judge로 스팟체크한다(eval/ragas_eval.py와 같은 관례)."""
    if os.getenv("ABSENCE_JUDGE", "vllm") == "openai":
        raise NotImplementedError(
            "외부 judge 경로는 미구현이다 — eval/ragas_eval.py의 OpenAI 배관을 참고해 붙일 것. "
            "지금 필요한 건 회귀 감시(상대 비교)라 self-judge로 충분하다는 판단이다(#76)."
        )
    return LlmClient()


async def judge_absence(llm: LlmClient, query: str, answer: str) -> AbsenceJudgment:
    """인용 0건 답변 하나를 판정. 실패는 LlmJudgmentFailed로 전파된다.

    호출부(eval/refusal.py)가 그 예외를 잡아 '판정 불가'로 집계한다 — 판정하지 못한 것을
    refusal_ok로 추측하면 지표가 실패를 숨긴다(rag/llm_schemas.py의 2갈래 계약과 같은 이유).
    """
    return await acomplete_validated(llm, _build_messages(query, answer), AbsenceJudgment)


# 판정기 회귀 고정 — #62 측정 중 정규식이 실제로 틀린 케이스 전수.
# 근거 있는 부정을 넣지 않는 이유는 모듈 docstring(판정 범위) 참조.
FIXTURES: list[dict] = [
    # ── 정규식이 놓쳤던 것 (미검출) ──
    {"id": "apology_then_assert", "expected": "absence_assertion",
     "query": "새벽배송으로 보내줄 수 있어요?",
     "answer": "죄송합니다. 새벽배송 서비스는 제공되지 않습니다.\n««[]»»",
     "defect": "첫 문장이 사과라 첫문장 규약이 놓쳤다"},
    {"id": "hard_word_missing", "expected": "absence_assertion",
     "query": "오프라인 매장에서 반품되나요?",
     "answer": "오프라인 매장에서의 반품은 어렵습니다.\n««[]»»",
     "defect": "'어렵습니다'가 패턴에 없었다"},
    # ── 정규식이 잘못 잡았던 것 (오탐) ──
    {"id": "doc_state_myungsi", "expected": "refusal_ok",
     "query": "대량 구매 할인 있나요?",
     "answer": "대량 구매 할인에 대해 명시된 바가 없습니다.\n««[]»»",
     "defect": "문서 상태 서술인데 단정으로 잡혔다"},
    {"id": "doc_state_regulation", "expected": "refusal_ok",
     "query": "매장 픽업 가능한가요?",
     "answer": "매장 픽업에 대한 명시적인 규정은 제공된 문서에 없습니다.\n««[]»»",
     "defect": "같은 오탐"},
    {"id": "doc_state_josa", "expected": "refusal_ok",
     "query": "적립금을 현금으로 받을 수 있어요?",
     "answer": "적립금의 현금 출금에 대한 명시는 없습니다.\n««[]»»",
     "defect": "조사 뒤 바로 '없'이 와서 오탐"},
    # ── 정상형 (경계의 기준점) ──
    {"id": "canonical", "expected": "refusal_ok",
     "query": "북한 지원 사업도 하나요?",
     "answer": f"{NO_EVIDENCE_ANSWER}\n««[]»»",
     "defect": "고정 문구 — 이걸 틀리면 판정기가 쓸 수 없다"},
    # ── 완화가 뒤에 붙어도 핵심 문장이 단정인 경우 ──
    {"id": "assert_then_hedge", "expected": "absence_assertion",
     "query": "해외로도 배송되나요?",
     "answer": ("해외 배송 서비스는 제공되지 않습니다. 다만 정확한 내용은 고객센터로 "
                "문의해 주시면 확인이 필요합니다.\n««[]»»"),
     "defect": "뒤 완화에 가려 정상으로 보일 수 있다"},
]


async def selftest(runs: int = SELFTEST_RUNS) -> dict:
    """FIXTURES 대조 → {accuracy, flaky, misses, runs}.

    judge가 비결정적이라 pytest가 아니라 여기 둔다 — eval/other_eval.py·condense.py와
    같은 판단이다(확률적 판정은 단위 테스트가 아니라 반복 실행으로 본다). 다만 그쪽 RUNS는
    '생성이 매번 달라지는 것'을 재고, 여기 답변은 고정이라 흔들리는 것은 judge뿐이다.
    """
    llm = judge_llm()
    per: dict[str, list[str | None]] = {f["id"]: [] for f in FIXTURES}
    for _ in range(runs):
        for f in FIXTURES:
            try:
                v = await judge_absence(llm, f["query"], f["answer"])
                per[f["id"]].append(v.label)
            except Exception as exc:                       # 판정 불가도 관측 대상이다
                per[f["id"]].append(f"error:{type(exc).__name__}")
    ok = sum(1 for f in FIXTURES if all(l == f["expected"] for l in per[f["id"]]))
    flaky = [f["id"] for f in FIXTURES if len(set(per[f["id"]])) > 1]
    return {
        "runs": runs, "n": len(FIXTURES), "accuracy": ok / len(FIXTURES) if FIXTURES else 0.0,
        "flaky": flaky,
        "misses": [{"id": f["id"], "expected": f["expected"], "got": per[f["id"]],
                    "defect": f["defect"]}
                   for f in FIXTURES if any(l != f["expected"] for l in per[f["id"]])],
    }


def save_audit(rows: list[dict], stamp: str | None = None) -> Path:
    """판정 원문·사유를 감사 로그로 남긴다.

    요약 JSON(eval/results/history/summary_*.json)은 헤드라인 지표만 담는 계약이라 여기 안 넣는다.
    #62에서 refusal.compute()의 misses가 요약에 실리지 않아 매 측정마다 원문을 손으로 다시
    읽어야 했다 — 그 반복을 없애는 것이 이 파일의 목적이다.
    파일명에 스탬프와 건수를 박는 이유는 eval/ragas_eval.py와 같다 — 고정 이름은 덮어쓴다.
    """
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    p = RESULTS / f"absence_judge_{stamp}_n{len(rows)}.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return p


async def main() -> None:
    r = await selftest()
    print(f"[부재단정 판정기 자체검증 {JUDGE_PROMPT_VERSION}]  "
          f"{r['n']}케이스 × {r['runs']}회 → 전회 일치 {r['accuracy']:.0%}")
    if r["flaky"]:
        print(f"  흔들림: {', '.join(r['flaky'])}")
    for m in r["misses"]:
        print(f"  ✗ {m['id']:22} 기대={m['expected']:18} 판정={m['got']}")
        print(f"    ({m['defect']})")
    if not r["misses"]:
        print("  전 케이스 통과 — 정규식 결함 6종을 판정기가 다 가른다")


if __name__ == "__main__":
    asyncio.run(main())
