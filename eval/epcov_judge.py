"""EPCov(기대 포인트 커버리지) 2단계 판정 (#221) — 규칙 미적중 포인트만 LLM이 '담겼는지' 다시 본다.

1단계 `eval.generation.expected_points_hits`(부분문자열 → 금액 게이트 → 임베딩 유사도 0.75)는 **뜻이 같아도 표현이
나뉘거나 바뀌면** 경계에서 놓친다. 실측(기준선 A 10/10 retrieved 658행·1,147포인트, 2026-10-11):
  · 규칙 미적중 202포인트 중 **최소 64(32%)가 실제로 담긴 포인트**(위음성) — 갈린 행을 원문 대조로 판정.
    예: adererror_pp001 포인트1은 "데님 특성…" / "따라서 이염은 보상 대상이 아닙니다"로 두 문장에 나뉘어 유사도 0.747.
  · 반대 방향(유사도 적중인데 조건·수치 누락)은 표본 100 중 7 — 이번 범위 밖(전량 심판은 호출 4배).
그래서 **규칙이 놓친 포인트만** 심판에 넘긴다. 적중 포인트는 지금처럼 결정적이다.

판정자 비교(같은 프롬프트, 사람 판정이 끝난 갈린 89행): Claude Sonnet 4.5 76 / 사내 vLLM 71.
틀리는 **방향**이 달랐다 — vLLM은 관대(9,600원 기대에 9,000원 답을 '담김'), Claude는 엄격. 위음성 구제 단계에서
관대한 오판은 점수를 부풀리므로 **기본 판정자는 Claude**다(사용자 결정 2026-10-11). `EPCOV_JUDGE=vllm`이면 사내 vLLM —
토큰 없이 반복 실험할 때. 판정자가 다른 수치는 서로 비교하지 않는다: 결과 행의 `epcov_judge`(판정자·모델·프롬프트 버전)로 가른다.

**금액 게이트를 심판에도 건다** — 포인트의 원 단위 금액이 답변에 없으면 심판에 넘기지 않고 미적중을 유지한다.
위 vLLM의 9,600/9,000 오판이 정확히 이 경우다(호출부 `eval.generation._epcov_scores`가 판정 대상을 거른다).

판정 실패(LlmJudgmentFailed·한도·타임아웃)는 **미적중 유지**(fail-closed) — misinfo_judge·absence_judge와 같은 원칙.
사유·오류는 감사 행으로 결과에 남는다.
"""
import asyncio
import os
from typing import Literal

from pydantic import BaseModel

from rag.llm_schemas import acomplete_validated

JUDGE_PROMPT_VERSION = "v1"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"   # 버전 고정 — 'sonnet' 별칭은 시점마다 모델이 바뀐다


class CoverageJudgment(BaseModel):
    """label에 기본값을 두지 않는다 — absence_judge.AbsenceJudgment와 같은 이유(스키마 required 보장)."""
    label: Literal["covered", "not_covered"]
    reason: str | None = None


# #221 1단계 실측에 쓴 문구 그대로다 — 바꾸면 위 실측치가 이 프롬프트의 근거가 아니게 된다(JUDGE_PROMPT_VERSION 올릴 것).
_JUDGE_SYSTEM = """당신은 상담 답변 채점자입니다. '기대 포인트'의 내용이 '답변'에 담겨 있는지 판정합니다.
- covered: 포인트의 핵심(결론·조건·수치·대상)이 답변에 의미상 모두 있다. 표현이 다르거나, 순서가 바뀌거나, 여러 문장에 나뉘어 있어도 된다.
- not_covered: 포인트의 핵심 중 하나라도 빠졌거나(조건·수치·대상 누락), 답변의 결론이 포인트와 다르다(부정·반대 포함).
답변에 없는 내용을 추측해 채우지 마십시오. JSON 한 줄로만 답하십시오: {"label": "...", "reason": "판정 근거 한 줄"}"""


def judge_spec() -> tuple[str, str]:
    """(판정자, 모델) — 환경변수 EPCOV_JUDGE(기본 claude)·EPCOV_JUDGE_MODEL. 결과 행 기록과 judge_llm이 같은 값을 쓴다."""
    backend = os.getenv("EPCOV_JUDGE", "claude")
    if backend == "claude":
        return backend, os.getenv("EPCOV_JUDGE_MODEL", DEFAULT_CLAUDE_MODEL)
    if backend == "vllm":
        from config import settings
        return backend, settings.vllm_model
    raise SystemExit(f"EPCOV_JUDGE={backend} — claude|vllm 만 지원")


def judge_label() -> str:
    """결과 행에 남기는 판정자 식별자 — 이것이 다르면 EPCov 수치를 비교하지 않는다."""
    backend, model = judge_spec()
    return f"{backend}:{model}:{JUDGE_PROMPT_VERSION}"


def judge_llm():
    """판정 클라이언트. 전용 스위치라 부재 단정·오정보 심판(ABSENCE_JUDGE)과 따로 켜고 끈다 — 축별 토큰 조절."""
    backend, model = judge_spec()
    if backend == "claude":
        import shutil
        if not shutil.which("claude"):
            # 조용히 규칙만으로 진행하지 않는다 — 다른 눈금의 수치가 '심판 반영'으로 착각되는 것을 막는다(#221 리뷰 결정)
            raise SystemExit("EPCov 판정자가 claude(기본)인데 claude CLI가 PATH에 없다 — "
                             "설치·로그인하거나 EPCOV_JUDGE=vllm으로 실행하라")
        from eval.claude_client import ClaudeCliClient
        return ClaudeCliClient(model=model)
    from rag.llm import LlmClient
    return LlmClient()


def _build_messages(answer: str, point: str) -> list[dict]:
    return [{"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": f"기대 포인트: {point}\n\n답변:\n{answer}"}]


async def judge_point(llm, answer: str, point: str) -> CoverageJudgment:
    """포인트 하나를 판정. 실패는 예외로 전파 — confirm_points가 미적중 유지로 집계한다."""
    return await acomplete_validated(llm, _build_messages(answer, point), CoverageJudgment)


async def confirm_points(llm, answer: str, points: list[str], rule_hits: list[bool],
                         targets: list[int], sem: asyncio.Semaphore) -> tuple[list[bool], list[dict]]:
    """규칙 판정 → (최종 히트, 감사 행). targets(규칙 미적중 중 판정 대상 인덱스)만 LLM에 묻는다.

    감사 행: {"k", "label", "reason", "error"} — 판정한 포인트마다 하나. error가 있으면 label은 None이고 미적중 유지.
    targets가 비면 LLM을 부르지 않는다. 포인트들은 sem 안에서 병렬로 묻는다.
    """
    final = list(rule_hits)

    async def _one(k: int) -> dict:
        async with sem:
            try:
                v = await judge_point(llm, answer, points[k])
            except Exception as exc:   # LlmJudgmentFailed·한도·타임아웃 — 판정 불가는 미적중 유지
                return {"k": k, "label": None, "reason": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
        return {"k": k, "label": v.label, "reason": v.reason, "error": None}

    audit = list(await asyncio.gather(*(_one(k) for k in targets)))
    for a in audit:
        if a["label"] == "covered":
            final[a["k"]] = True
    return final, audit


# 판정기 회귀 고정 — #221 1단계에서 사람 판정이 끝난 행(모의 테넌트만 — 저장소가 공개다).
FIXTURES: list[dict] = [
    {"id": "split_sentences", "expected": "covered",            # 규칙 위음성: 유사도 0.747
     "point": "청바지로 인한 이염은 데님 소재의 특성으로 보고, 보상 대상이 아닙니다",
     "answer": "해당 상황은 데님 상품의 특성으로 인한 이염으로 판단됩니다. 데님은 인디고 염료 특성상 젖거나 마찰될 때 "
               "밝은 색 물품에 이염될 수 있습니다. 따라서 이염은 보상 대상이 아닙니다."},
    {"id": "list_items", "expected": "covered",
     "point": "예열 미실시, 음료량 부족, 뚜껑 불완전 체결, 패킹 마모",
     "answer": "반품 접수 전에 확인할 사항입니다.\n- 예열 여부: 끓는 물로 1분간 예열했는지\n- 음료량: 병목 아래 1cm까지 "
               "채웠는지\n- 뚜껑 체결 상태: 완전히 잠갔는지\n- 패킹 사용 기간: 6개월 이상 사용했는지"},
    {"id": "condition_changed", "expected": "not_covered",       # 조건이 다른 진술
     "point": "한 번 녹인 뒤 다시 얼리는 재냉동 금지입니다",
     "answer": "치킨너겟은 냉동실(-18℃ 이하)에서 제조일로부터 18개월까지 보관할 수 있습니다. 개봉 후에는 재냉동이 금지되어 있습니다."},
    {"id": "contradicted", "expected": "not_covered",            # 결론이 반대
     "point": "한정판(래플) 상품은 재고 특성상 사이즈 교환 불가",
     "answer": "불량이 아닌 단순 변심으로 인한 사이즈 교환은 고객 부담의 왕복 배송비 5,000원을 지불하시면 가능합니다."},
    {"id": "condition_missing", "expected": "not_covered",       # 조건 누락 — vLLM이 관대하게 '담김'으로 본 유형
     "point": "제휴카드: 5만원 이상 결제 시 2~6개월 무이자 할부",
     "answer": "제휴카드 무이자 할부는 2~6개월까지 가능합니다. 일반 신용카드의 경우 2~3개월까지 가능합니다."},
]


async def selftest(runs: int = 3) -> dict:
    """FIXTURES를 runs회 판정해 기대와 비교 — 실판정자 흔들림 관측(misinfo_judge.selftest와 같은 idiom)."""
    llm = judge_llm()
    out = []
    for f in FIXTURES:
        labels = []
        for _ in range(runs):
            try:
                labels.append((await judge_point(llm, f["answer"], f["point"])).label)
            except Exception as exc:
                labels.append(f"error:{type(exc).__name__}")
        out.append({"id": f["id"], "expected": f["expected"], "labels": labels,
                    "ok": all(l == f["expected"] for l in labels)})
    return {"judge": judge_label(), "runs": runs, "fixtures": out, "all_ok": all(o["ok"] for o in out)}


if __name__ == "__main__":
    import json
    print(json.dumps(asyncio.run(selftest()), ensure_ascii=False, indent=1))
