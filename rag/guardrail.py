"""입력 가드레일 + 인텐트 분류.

사용자 질의를 LLM에 한 번 분류시켜 ① 안전한 입력인가 ② KNOWLEDGE인가 OTHER인가를 판단한다.
판단 실패(호출 실패·구조 신뢰 불가)는 fail-open — 검색 경로가 안전 측이라 과잉거절을 피한다.
폴백은 kms.schema_fallback으로 정상 판정과 구분되게 관측한다 (#43 — 이전엔 구분 불가였다).

출력 형식은 vLLM structured_outputs의 json 스키마로 강제한다(#43) — RouteDecision이
스키마이자 검증(rag/llm_schemas). 구 방어 파서(_extract_json·_as_bool)는 스키마가 대체했고,
무스키마 폴백 경로의 잔여 방어는 acomplete_validated가 공용으로 담당한다.

출력 가드레일(생성된 답변을 사후 검사)은 #26에서 제거했다: 스트리밍에서는 토큰이 이미 전송된
뒤에야 판정이 끝나 차단이 아니라 화면 가림에 그치고, 원문이 DB에서 대체돼 사후 조사도 불가했다.
되살릴 때는 사후 검사가 아닌 설계(전송 전 버퍼링·문장 단위 검사·사전 필터)여야 한다.
"""
import logging

from rag.llm import LlmClient
from rag.llm_schemas import RouteDecision, acomplete_validated
from rag import otel
from rag.prompts import build_intent_guard_prompt, build_classify_user_message

logger = logging.getLogger(__name__)


async def classify_and_guard(llm: LlmClient, query: str, has_attachments: bool = False,
                             domain_hint: str | None = None) -> RouteDecision:
    """입력 안전성 + 인텐트를 한 번의 LLM 호출로 판단한다.

    safe=false면 차단(BLOCKED), 아니면 intent로 라우팅.
    domain_hint는 KNOWLEDGE 정의의 지식 범위 슬롯에 주입 (빈 값은 중립 폴백).
    호출 실패·검증 실패는 fail-open + KNOWLEDGE (검색 경로 = 안전 측, 과잉거절 방지).
    """
    with otel.span('classify_and_guard', 'GUARDRAIL') as sp:
        try:
            decision = await acomplete_validated(llm, [
                {'role': 'system', 'content': build_intent_guard_prompt(domain_hint)},
                {'role': 'user', 'content': build_classify_user_message(query, has_attachments)},
            ], RouteDecision, span=sp)
        except Exception:
            logger.exception('LLM error(classify_and_guard)')
            decision = None
        if decision is None:
            # fail-open (기존 동작 유지). 폴백으로 만들어진 KNOWLEDGE가 모델이 판단한
            # KNOWLEDGE와 똑같이 보이던 문제(#43) — 이 속성 하나가 그 구분이다.
            otel.set_attrs(sp, {'kms.schema_fallback': True})
            decision = RouteDecision(safe=True, intent='KNOWLEDGE')
        otel.set_attrs(sp, {otel.INPUT_VALUE: query, 'kms.intent': decision.intent, 'kms.safe': decision.safe,
                            'kms.block_reason': decision.reason})   # 차단 사유 — 없으면 no-op (#22)
        return decision
