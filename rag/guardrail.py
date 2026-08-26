"""입력 가드레일 + 인텐트 분류.

사용자 질의를 LLM에 한 번 분류시켜 ① 안전한 입력인가 ② 인텐트가 무엇인가
(KNOWLEDGE·OTHER·RETRY·ATTACHMENT — rag/llm_schemas.RouteDecision)를 판단한다.
RETRY·ATTACHMENT는 표면 패턴 인식용 전이 인텐트라 여기서 라우팅이 끝나지 않는다 —
prepare()의 디스패처가 상태(직전 턴 미답변 여부·첨부 유무)를 보고 즉시 해소한다.
판단 실패(호출 실패·구조 신뢰 불가)는 **그대로 전파**해 턴을 실패로 끝낸다 (#72).

예전엔 fail-open으로 `safe=True, intent=KNOWLEDGE`를 만들어 넣었다. 그건 안전 측이 아니었다 —
**입력 가드가 꺼진 채 통과**시키는 것이고(#22의 방어가 조용히 무력화), 모든 OTHER 질의가 검색
경로로 가 "안녕"에 근거없음으로 답하게 만들었다. 무엇보다 LLM이 죽었다면 생성도 실패하므로
폴백이 턴을 살리지도 못했다 — 에러를 검색까지 다 수행한 뒤로 미룰 뿐이었다.

출력 형식은 vLLM structured_outputs의 json 스키마로 강제한다(#43) — RouteDecision이
스키마이자 검증(rag/llm_schemas). 구 방어 파서(_extract_json·_as_bool)는 스키마가 대체했고,
무스키마 재시도(400/422)는 acomplete_validated가 공용으로 담당한다.

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
    판단 실패는 폴백 없이 전파한다 (#72) — 사유는 모듈 docstring. 이 시점엔 사용자의 질문이
    이미 저장돼 있으므로(턴 시작 자리표시), 실패해도 질문이 유실되지는 않는다.
    """
    with otel.span('classify_and_guard', 'GUARDRAIL') as sp:
        decision = await acomplete_validated(llm, [
            {'role': 'system', 'content': build_intent_guard_prompt(domain_hint)},
            {'role': 'user', 'content': build_classify_user_message(query, has_attachments)},
        ], RouteDecision, span=sp)
        otel.set_attrs(sp, {otel.INPUT_VALUE: query, 'kms.intent': decision.intent, 'kms.safe': decision.safe,
                            'kms.block_reason': decision.reason})   # 차단 사유 — 없으면 no-op (#22)
        return decision
