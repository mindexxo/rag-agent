"""멀티턴 대화 관리 — 소유권 판정·프롬프트 이력 조립·condense 질의 재작성.

턴 시작·종료(자리표시 기록·마감)는 rag/turn/turn_state.py가 맡는다.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag import otel
from rag.llm import LlmClient
from rag.models import Conversation, Message, TurnStatus
from rag.tokens import estimate_tokens
from rag.turn.llm_schemas import CondenseMultiResult, CondenseResult, acomplete_validated
from rag.turn.prompt_texts import (
    CANCELLED_TURN_EMPTY,
    CANCELLED_TURN_SUFFIX,
    CONDENSE_MULTI_SYSTEM_PROMPT,
    CONDENSE_SYSTEM_PROMPT,
    FAILED_TURN_EMPTY,
)
from rag.turn.prompts import build_chat_prompt, build_condense_user_message
from rag.turn.turn_state import HISTORY_ISOLATED

# X-User-Id 미전송 시 created_by 저장과 소유 필터에 똑같이 쓰는 폴백.
DEFAULT_USER = 'test-user'


class ConversationNotFound(LookupError):
    """없거나 삭제됐거나 다른 소유자의 conversation_id — HTTP 경계가 404로 매핑하는 유일한 prepare 실패."""


def owned_filter(tenant_id: str, user_id: str | None):
    """대화 소유 판정 WHERE 표현식(tenant + created_by + 미삭제)의 정의점.

    질의 경로와 routers/conversations.py의 조회가 함께 쓴다. user_id=None은 DEFAULT_USER로 본다.
    created_by가 NULL인 대화는 어느 사용자와도 맞지 않아 보이지 않는다. SQL 술어 전용이다.
    """
    return (
        (Conversation.tenant_id == tenant_id)
        & (Conversation.created_by == (user_id or DEFAULT_USER))
        & (Conversation.deleted_at.is_(None))
    )


async def ensure_conversation(
    session: AsyncSession,
    tenant_id: str,
    conversation_id: int | None,
    user_id: str | None = None,
) -> Conversation:
    """conversation_id=None이면 새 대화를 만들고, 있으면 소유·미삭제를 검증한다(아니면 ConversationNotFound)."""
    if conversation_id is None:
        # 저장값 폴백은 owned_filter의 필터 폴백과 별개 책임이라 여기서 직접 계산한다.
        conversation = Conversation(tenant_id=tenant_id, created_by=user_id or DEFAULT_USER)
        session.add(conversation)
        await session.flush()
        return conversation

    conversation = (
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id).where(owned_filter(tenant_id, user_id))
        )
    ).scalar_one_or_none()
    if conversation is None:
        raise ConversationNotFound(f'Conversation {conversation_id} not found')
    return conversation


async def load_recent_messages(
    session: AsyncSession,
    tenant_id: str,
    conversation_id: int,
    limit: int = 30,
) -> list[Message]:
    """프롬프트 재료용 최근 메시지를 시간순으로 반환한다(격리 턴은 짝째로 뺀다).

    limit은 이력 토큰 예산을 항상 채우는 상한일 뿐이고, 실제 자르기는 build_prior_turns의 예산이 한다.
    """
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.tenant_id == tenant_id)
        # 한 턴의 user·assistant는 created_at이 같아 id로 순서를 확정한다(user가 먼저).
        .order_by(Message.created_at.desc(), Message.id.desc())
        # 아래 격리 필터로 빠지는 만큼 여유 조회한다.
        .limit(limit * 2)
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())[::-1]

    # 격리는 프롬프트 재료에서만 한다(조회 API는 그대로 노출). cancelled·failed는 남겨야 화면과
    # 모델이 보는 이력이 맞아 "다시"의 지시대상이 밀리지 않는다. 격리 집합의 정의는 turn_state.HISTORY_ISOLATED.
    # user·assistant 짝을 함께 뺀다 — 한쪽만 빼면 build_prior_turns의 짝짓기가 밀린다.
    # 바꾸면 eval/_gold_history.py도 확인할 것.
    dropped_questions = {
        m.question_message_id for m in messages if m.role == 'assistant' and m.status in HISTORY_ISOLATED
    }
    kept = [
        m
        for m in messages
        if not (m.id in dropped_questions or (m.role == 'assistant' and m.status in HISTORY_ISOLATED))
    ]
    return kept[-limit:]  # 여유 조회분을 되돌려 원래 창 크기 유지


def _history_content(message: Message) -> str:
    """프롬프트 이력용 답변 텍스트 — 취소·실패 턴 표식의 정의점.

    표식은 조립 시점에만 붙이고 저장본(Message.content)은 바꾸지 않는다. build_prior_turns와
    _condense_call이 함께 써야 두 이력이 어긋나지 않는다. message의 새 속성을 읽기 시작하면
    eval/_gold_history.py도 확인할 것 — eval이 그 속성을 안 채우면 조용히 다른 분기를 탄다.
    """
    if message.role == 'assistant' and message.status == TurnStatus.CANCELLED:
        return message.content + CANCELLED_TURN_SUFFIX if message.content else CANCELLED_TURN_EMPTY
    if message.role == 'assistant' and message.status == TurnStatus.FAILED:
        return FAILED_TURN_EMPTY  # 실패는 항상 content='' (finalize가 빈 답변으로 마감)
    return message.content


async def _condense_call(
    llm: LlmClient,
    query: str,
    messages: list[Message],
    multi: bool,
) -> list[str]:
    """condense_query·condense_to_queries 공용 골격.

    반환 [0]=standalone, [1:]=검색 변형(multi일 때만). span 이름은 둘 다 'condense'(대시보드 연속성).
    """
    history = [{'role': m.role, 'content': _history_content(m)} for m in messages]

    with otel.span('condense', 'LLM') as sp:
        # 호출 실패는 폴백 없이 전파해 턴을 실패로 끝낸다 — 질문은 턴 시작 시 이미 저장돼 있다.
        llm_messages = build_chat_prompt(
            CONDENSE_MULTI_SYSTEM_PROMPT if multi else CONDENSE_SYSTEM_PROMPT,
            build_condense_user_message(query, history),
        )
        parsed = await acomplete_validated(llm, llm_messages, CondenseMultiResult if multi else CondenseResult, span=sp)
        queries = _postprocess_queries(parsed, query)
        attrs = {otel.INPUT_VALUE: query, otel.OUTPUT_VALUE: queries[0], 'kms.history_messages': len(history)}
        if multi:
            attrs['kms.expanded_queries'] = queries[1:]  # DB에 저장 안 되는 유일한 기록처
        otel.set_attrs(sp, attrs)
        return queries


async def condense_query(llm: LlmClient, query: str, messages: list[Message]) -> str:
    """후속 질문을 독립 질문으로 바꾼다. 히스토리가 없으면 LLM 없이 query를 그대로 돌려준다."""
    if not messages:
        return query
    queries = await _condense_call(llm, query, messages, multi=False)
    return queries[0]


async def condense_to_queries(
    llm: LlmClient,
    query: str,
    messages: list[Message],
) -> list[str]:
    """재작성과 검색용 변형을 한 번의 호출로 만든다.

    반환 [0]=standalone(저장·캐시 키·리랭크 기준), [1:]=검색 전용 어휘 변형(최대 2).
    condense_query와 달리 히스토리가 없어도 호출한다 — 멀티턴에서만 부르는 게이트는 호출부 책임
    (eval이 빈 히스토리로 비교를 돌린다).
    """
    return await _condense_call(llm, query, messages, multi=True)


def _postprocess_queries(parsed: CondenseResult | CondenseMultiResult, query: str) -> list[str]:
    """스키마가 강제하지 못하는 것만 보정한다.

    빈 standalone은 원본으로 폴백(minLength 강제가 서버마다 보장되지 않음), 중복 변형은 제거
    (RRF에서 같은 순위를 중복 가산해 원본 상위 결과를 희석한다), maxItems 무시에 대비해 2개로 자른다.
    """
    standalone = parsed.standalone.strip() or query
    raw_variants = getattr(parsed, 'variants', [])  # CondenseResult(단일)엔 variants가 없다
    variants: list[str] = []
    for line in (v.strip() for v in raw_variants):
        if line and line != standalone and line not in variants:
            variants.append(line)
    return [standalone, *variants[:2]]


def trim_messages_for_condense(messages: list, budget_tokens: int) -> list:
    """condense에 넘길 히스토리를 토큰 예산으로 자른다(최신부터 담고 시간순 복원).

    긴 히스토리를 그대로 넘기면 이전 답변의 수치·조건이 재작성 질의에 주입된다.
    """
    selected, used = [], 0
    for m in reversed(messages):
        cost = estimate_tokens(m.content)
        if selected and used + cost > budget_tokens:  # 최소 1개는 보장
            break
        selected.append(m)
        used += cost
    return list(reversed(selected))


def build_prior_turns(messages: list, budget_tokens: int) -> list[dict]:
    """user→assistant 짝을 Q/A 턴으로 묶어 최신부터 토큰 예산 안에서 담는다(시간순 반환).

    검색 근거가 아니라 답변 생성의 대화 맥락 참고용이다.
    """
    turns = []
    pending_question = None

    for message in messages:
        if message.role == "user":
            pending_question = message.content
        elif message.role == "assistant" and pending_question:
            turns.append({"q": pending_question, "a": _history_content(message)})
            pending_question = None

    selected, used = [], 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn["q"]) + estimate_tokens(turn["a"])
        if selected and used + cost > budget_tokens:  # 최소 1턴은 보장
            break
        selected.append(turn)
        used += cost
    return list(reversed(selected))
