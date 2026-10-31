"""생성 판정 2단계 — 기계 판정(문자열 포함)이 실패한 답변만 LLM이 다시 본다(#253, #221 EPCov 심판과 같은 설계).

기계 판정은 "1년에 30일"(정답 '연 30일')·"퇴직일 3일 전"(정답 'D-3일')처럼 말만 바꾼 정답을 못 잡는다(2차 실행 19건 중 10건).
여기서는 질문·정답 값·답변·**그 문서의 색인 청크**를 함께 주고 두 가지를 묻는다:
  정답 포함 — 답변이 정답 값과 같은 뜻인가(표현 차이 허용)
  근거 있음 — 그 값이 청크에 실제로 있는가
둘 다 있어야 correct. 값은 맞는데 청크에 없으면 **ungrounded**(OCR이 ☑를 지웠는데 "완료"라고 한 경우 — 운에 맡긴 정답)로
따로 센다. 판정자는 기본 Claude(`TABLE_JUDGE=claude`, CLI 필요), `TABLE_JUDGE=vllm`이면 사내 vLLM — #221에서 vLLM 판정이
관대하게 틀렸다. 답을 만든 모델(14B)과 판정자가 다르다는 점을 결과에 남긴다(judge 필드).
"""
import asyncio
import os
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field

from rag.turn.llm_schemas import acomplete_validated

DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"
PROMPT_VERSION = "v2"   # v1은 JSON 키를 안 적어 Claude가 "verdict"로 답했다(22/22 검증 실패)


class AnswerJudgment(BaseModel):
    label: Literal["correct", "wrong", "ungrounded", "not_answered"] = Field(validation_alias=AliasChoices("label", "verdict"))
    reason: str = ""


_SYSTEM = """너는 사내 상담 지식 시스템의 답변을 채점하는 심판이다. 질문, 정답 값(표의 칸 값), 시스템 답변, 그리고 그 표가 색인된 청크를 받는다.
다음 중 하나로 판정하라.
- correct: 답변이 정답 값과 같은 뜻을 말하고(표현·단위·날짜 표기 차이는 허용 — "연 30일"="1년에 30일", "D-3일"="3일 전", "☑"="완료"), 그 값이 청크에 실제로 있다.
- ungrounded: 답변은 정답과 같은 뜻인데, 청크에 그 값(또는 그 뜻의 표시)이 없다 — 답이 근거 없이 나온 것.
- wrong: 답변이 정답과 다른 값·다른 뜻을 말한다(일부만 맞아도 틀린 값이 섞이면 wrong).
- not_answered: 답변이 "확인할 수 없다"류의 거절·회피다.
reason은 한국어 한 문장. 답변에 정답 외 추가 설명이 있어도 정답이 맞게 들어 있으면 correct다.
코드 블록 없이 JSON 한 줄로만 답하라: {"label": "correct|wrong|ungrounded|not_answered", "reason": "판정 근거 한 문장"}"""


def judge_spec() -> tuple[str, str]:
    backend = os.getenv("TABLE_JUDGE", "claude")
    if backend == "claude":
        return backend, os.getenv("TABLE_JUDGE_MODEL", DEFAULT_CLAUDE_MODEL)
    if backend == "vllm":
        from config import settings
        return backend, settings.llm_model
    raise SystemExit(f"TABLE_JUDGE={backend} — claude|vllm 만 지원")


def judge_label() -> str:
    backend, model = judge_spec()
    return f"{backend}:{model}:{PROMPT_VERSION}"


def judge_llm():
    backend, model = judge_spec()
    if backend == "claude":
        import shutil
        if not shutil.which("claude"):
            raise SystemExit("판정자가 claude(기본)인데 claude CLI가 PATH에 없다 — 설치·로그인하거나 TABLE_JUDGE=vllm")
        from eval.claude_client import ClaudeCliClient
        return ClaudeCliClient(model=model)
    from rag.llm import create_llm
    return create_llm()


def build_messages(question: str, expected: str, answer: str, chunks: list[str]) -> list[dict]:
    ctx = "\n\n".join(chunks)[:6000] or "(색인 청크 없음)"
    return [{"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"질문: {question}\n정답 값: {expected}\n\n시스템 답변:\n{answer}\n\n색인 청크:\n{ctx}"}]


async def judge_answer(llm, question: str, expected: str, answer: str, chunks: list[str]) -> AnswerJudgment:
    return await acomplete_validated(llm, build_messages(question, expected, answer, chunks), AnswerJudgment)


async def rejudge(rows: list[dict], chunks_by_doc: dict, *, llm=None, concurrency: int = 3) -> list[dict]:
    """기계 판정 실패(또는 manual) 행에 LLM 판정을 붙인다 — 행에 llm_label·llm_reason·judge를 쓴다(제자리).
    실패(한도·타임아웃)는 llm_label=None·llm_error로 남기고 기계 판정을 유지한다."""
    llm = llm or judge_llm()
    label = judge_label()
    sem = asyncio.Semaphore(concurrency)
    targets = [r for r in rows if not r.get('value_ok') or r.get('manual')]

    async def _one(r):
        async with sem:
            try:
                v = await judge_answer(llm, r['question'], r['expected'], r['answer'],
                                       [c['text'] for c in chunks_by_doc.get((r['id'], r['path']), [])])
                r['llm_label'], r['llm_reason'], r['llm_error'] = v.label, v.reason, None
            except Exception as exc:                       # noqa: BLE001 — 판정 불가는 기계 판정 유지
                r['llm_label'], r['llm_reason'], r['llm_error'] = None, None, f"{type(exc).__name__}: {exc}"[:200]
            r['judge'] = label
    await asyncio.gather(*(_one(r) for r in targets))
    return targets


def final_ok(row: dict) -> bool:
    """최종 생성 판정 — 기계가 맞다고 했거나, LLM이 correct라고 한 것."""
    return bool(row.get('value_ok')) or row.get('llm_label') == 'correct'
