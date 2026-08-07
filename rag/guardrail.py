"""출력 가드레일
생성된 답변을 LLM에 분류시켜 safe 여부를 판단한다.
판단 실패(JSON파싱 불가)는 fail-open
"""
import json
import logging
from dataclasses import dataclass

from rag.llm import LlmClient
from rag.prompts import GUARDRAIL_OUTPUT_PROMPT, build_intent_guard_prompt, build_classify_user_message

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    safe: bool
    reason: str | None = None


@dataclass
class RouteDecision:
    """입력 가드레일 + 인텐트 분류 통합 결과."""
    safe: bool
    intent: str            # 'KNOWLEDGE' | 'OTHER'
    reason: str | None = None


def _as_bool(value, default: bool = True) -> bool:
    """LLM이 bool 대신 문자열('false' 등)로 줄 수 있어 안전 파싱 (bool("false")==True 함정 회피).
    미지정(None)은 default(입력가드는 fail-open=True). 'false'/'no'/'0'류만 False.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ('false', 'no', '0', 'f', 'n')


async def classify_and_guard(llm: LlmClient, query: str, has_attachments: bool = False,
                             domain_hint: str | None = None) -> RouteDecision:
    """입력 안전성 + 인텐트를 한 번의 LLM 호출로 판단한다.

    safe=false면 차단(BLOCKED), 아니면 intent로 라우팅.
    domain_hint는 KNOWLEDGE 정의의 지식 범위 슬롯에 주입 (빈 값은 중립 폴백).
    파싱·호출 실패는 fail-open + KNOWLEDGE (검색 경로 = 안전 측, 과잉거절 방지).
    """
    try:
        raw = await llm.acomplete([
            {'role': 'system', 'content': build_intent_guard_prompt(domain_hint)},
            {'role': 'user', 'content': build_classify_user_message(query, has_attachments)},
        ])
        data = _extract_json(raw)
        intent = str(data.get('intent', 'KNOWLEDGE')).upper()
        if intent not in ('KNOWLEDGE', 'OTHER'):
            intent = 'KNOWLEDGE'
        return RouteDecision(safe=_as_bool(data.get('safe'), default=True), intent=intent, reason=data.get('reason'))
    except Exception:
        logger.exception('LLM error(classify_and_guard)')
        return RouteDecision(safe=True, intent='KNOWLEDGE')

def _extract_json(raw: str) -> dict:
    """LLM 출력에서 JSON 객체만 추출한다.
      모델이 ```json 펜스나 부연 텍스트를 붙이는 경우가 있어,
      첫 '{'부터 마지막 '}'까지만 잘라 파싱한다. 실패 시 예외는 호출부에서 처리.
      """
    start = raw.index('{')
    end = raw.rindex('}') + 1
    return json.loads(raw[start:end])

async def check_output(llm: LlmClient, answer: str) -> GuardrailResult:
    """생성된 답변을 분류해 차단 여부를 반환한다. 스트림 완료 후(사후) 호출."""
    raw = await llm.acomplete([
        {'role': 'system', 'content': GUARDRAIL_OUTPUT_PROMPT},
        {'role': 'user', 'content': answer},
    ])
    try:
        data = _extract_json(raw)
        return GuardrailResult(safe=_as_bool(data['safe']), reason=data.get('reason'))
    except (ValueError, KeyError, TypeError):
        # fail-open: 출력이 형식을 벗어나면 통과로 처리 (모듈 독스트링 참조)
        return GuardrailResult(safe=True, reason='guardrail_parse_failed')

