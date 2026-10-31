"""입력 가드레일 + 인텐트 분류 — 한 번의 LLM 호출로 입력 안전성과 인텐트(RouteDecision)를 판단한다.

RETRY·ATTACHMENT는 전이 인텐트라 prepare()의 디스패처가 상태를 보고 해소한다.
판단 실패는 폴백 없이 전파한다 — 안전·KNOWLEDGE로 채워 넣으면 입력 가드가 꺼진 채 통과된다.
출력 가드레일(답변 사후 검사)은 없다 — 스트리밍에선 이미 전송된 뒤라 차단이 되지 않는다.
"""

import logging

from rag import otel
from rag.llm import LlmClient
from rag.turn.llm_schemas import RouteDecision, acomplete_validated
from rag.turn.prompts import build_classify_user_message, build_intent_guard_prompt

logger = logging.getLogger(__name__)


async def classify_and_guard(
    llm: LlmClient,
    query: str,
    has_attachments: bool = False,
    domain_hint: str | None = None,
    prior_turns: list[dict] | None = None,
) -> RouteDecision:
    """입력 안전성 + 인텐트를 판단한다. safe=false면 차단, 아니면 intent로 라우팅.

    prior_turns는 직전 답을 반박하는 발화를 회상과 가르는 데 쓴다(자르는 예산은 호출부가 정한다).
    """
    with otel.span('classify_and_guard', 'GUARDRAIL') as sp:
        decision = await acomplete_validated(
            llm,
            [
                {'role': 'system', 'content': build_intent_guard_prompt(domain_hint)},
                {'role': 'user', 'content': build_classify_user_message(query, has_attachments, prior_turns)},
            ],
            RouteDecision,
            span=sp,
        )
        otel.set_attrs(
            sp,
            {
                otel.INPUT_VALUE: query,
                'kms.intent': decision.intent,
                'kms.safe': decision.safe,
                'kms.block_reason': decision.reason,
            },
        )
        return decision
