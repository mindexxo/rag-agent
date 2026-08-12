"""입력 가드레일 + 인텐트 분류.

사용자 질의를 LLM에 한 번 분류시켜 ① 안전한 입력인가 ② KNOWLEDGE인가 OTHER인가를 판단한다.
판단 실패(JSON 파싱 불가·호출 실패)는 fail-open — 검색 경로가 안전 측이라 과잉거절을 피한다.

출력 가드레일(생성된 답변을 사후 검사)은 #26에서 제거했다: 스트리밍에서는 토큰이 이미 전송된
뒤에야 판정이 끝나 차단이 아니라 화면 가림에 그치고, 원문이 DB에서 대체돼 사후 조사도 불가했다.
되살릴 때는 사후 검사가 아닌 설계(전송 전 버퍼링·문장 단위 검사·사전 필터)여야 한다.
"""
import json
import logging
from dataclasses import dataclass

from rag.llm import LlmClient
from rag import otel
from rag.prompts import build_intent_guard_prompt, build_classify_user_message

logger = logging.getLogger(__name__)


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
    with otel.span('classify_and_guard', 'GUARDRAIL') as sp:
        try:
            raw = await llm.acomplete([
                {'role': 'system', 'content': build_intent_guard_prompt(domain_hint)},
                {'role': 'user', 'content': build_classify_user_message(query, has_attachments)},
            ])
            data = _extract_json(raw)
            intent = str(data.get('intent', 'KNOWLEDGE')).upper()
            if intent not in ('KNOWLEDGE', 'OTHER'):
                intent = 'KNOWLEDGE'
            decision = RouteDecision(safe=_as_bool(data.get('safe'), default=True), intent=intent, reason=data.get('reason'))
        except Exception:
            logger.exception('LLM error(classify_and_guard)')
            decision = RouteDecision(safe=True, intent='KNOWLEDGE')   # fail-open (기존 동작 유지)
        otel.set_attrs(sp, {otel.INPUT_VALUE: query, 'kms.intent': decision.intent, 'kms.safe': decision.safe,
                            'kms.block_reason': decision.reason})   # 차단 사유 — 없으면 no-op (#22)
        return decision

def _extract_json(raw: str) -> dict:
    """LLM 출력에서 JSON 객체만 추출한다.
      모델이 ```json 펜스나 부연 텍스트를 붙이는 경우가 있어,
      첫 '{'부터 마지막 '}'까지만 잘라 파싱한다. 실패 시 예외는 호출부에서 처리.
      """
    start = raw.index('{')
    end = raw.rindex('}') + 1
    return json.loads(raw[start:end])

