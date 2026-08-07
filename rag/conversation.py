"""멀티턴 대화 관리 + condense 질의 재작성
대화 테이블(conversations/messages)을 이용해 이전 턴을 불러오고,
후속 질문('그럼 5일은?') 을 검색 가능한 독립 질문으로 변환한다.

  핵심 흐름:
      ensure_conversation -> load_recent_messages -> condense_query
      -> retrieve는 standalone_query로 수행
      -> 답변 완료 후 user/assistant 메시지 저장
"""
import logging

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag.llm import LlmClient
from rag.models import Conversation, Message

logger = logging.getLogger(__name__)
from rag.tokens import estimate_tokens
from rag.prompts import (
    CONDENSE_SYSTEM_PROMPT,
    build_chat_prompt,
    build_condense_user_message,
)


async def ensure_conversation(
        session: AsyncSession,
        tenant_id: str,
        conversation_id: int | None,
) -> Conversation:
    """대화가 없으면 새로 만들고, 있으면 테넌트 소유 대화인지 검증한다.
      conversation_id=None이면 새 Conversation을 INSERT한다.
      conversation_id가 있으면 tenant_id까지 같이 WHERE로 확인해서
      다른 테넌트 대화 접근을 차단한다.
    """
    if conversation_id is None:
        conversation = Conversation(tenant_id=tenant_id)
        session.add(conversation)
        await session.flush()
        return conversation

    conversation = await session.get(Conversation, conversation_id)

    if conversation is None or conversation.tenant_id != tenant_id:
        raise ValueError(f'Conversation {conversation_id} not found')

    return conversation

async def load_recent_messages(
        session: AsyncSession,
        tenant_id: str,
        conversation_id: int,
        limit: int = 30,
) -> list[Message]:
    """최근 메시지를 오래된 순서로 반환한다.
    DB에서는 최신순으로 limit만큼 가져오고,
    프롬프트에는 시간 순서가 자연스럽도록 reverse해서 넘긴다.

    limit 30(=15턴)은 이력 토큰 예산(history_budget_tokens=2000 ≈ 6~12턴)을 항상
    포화시키는 상한 — 실제 자르기는 build_prior_turns의 예산이 담당한다 (F100 의도 완성).
    (기존 10은 예산보다 먼저 걸려 '요약해줘'가 앞 턴을 놓치는 반쪽 동작이었음.)
    """
    stmt = (
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.tenant_id == tenant_id)
        # id 보조정렬(P1-14): 한 턴의 user·assistant가 같은 커밋이라 created_at 동률 →
        # id(자동증가, user가 먼저라 작음)로 순서 확정. reverse 후 user→assistant 보장.
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(limit)
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())

    return messages[::-1]

async def condense_query(
        llm: LlmClient,
        query: str,
        messages: list[Message]
) -> str:
    """후속 질문을 독립 질문으로 변환한다.

    히스토리가 없으면 LLM을 호출하지 않고 원본 query를 그대로 반환한다.
    LLM 결과가 비어 있거나 호출에 실패하면 원본 query로 폴백한다.
    """
    if not messages:
        return query

    history = [
        {'role': m.role, 'content': m.content}
        for m in messages
    ]

    try:
        llm_messages = build_chat_prompt(
            CONDENSE_SYSTEM_PROMPT,
            build_condense_user_message(query, history),
        )
        result = await llm.acomplete(llm_messages)

        if result is None:
            return query

        return result.strip() or query

    except Exception:
        logger.exception('LLM error(condense_query)')
        return query


async def save_exchange(
        session: AsyncSession,
        tenant_id: str,
        conversation_id: int,
        user_query: str,
        standalone_query: str,
        answer: str,
        sources: list[dict],
        attachments: list[dict] | None = None,
        user_id: str | None = None,
        latency_ms: int | None = None,
        cache_kind: str | None = None,
        cited_docs: list[str] | None = None,
        is_refusal: bool = False,
        intent: str | None = None,
) -> None:
    """사용자 질문과 assistant 답변을 세션에 등록한다.

    user 메시지에는 standalone_query와 (이번 턴에 첨부했다면) 첨부 추출 텍스트를,
    assistant 메시지에는 sources JSON을 저장한다.
    commit은 호출자가 담당한다.
    """
    user_message = Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="user",
        content=user_query,
        standalone_query=standalone_query,
        attachments=attachments,
        user_id=user_id,
    )
    session.add(user_message)
    await session.flush()   # user id 확보 — assistant의 question_message_id FK용

    assistant_message = Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
        sources=sources,
        latency_ms=latency_ms,
        cache_kind=cache_kind,
        cited_docs=cited_docs,
        is_refusal=is_refusal,
        # [원복 필요] intent — models.py 매핑 주석과 세트 (미매핑 kwargs는 TypeError). DB 반영 후 함께 해제
        # intent=intent,
        question_message_id=user_message.id,   # 짝을 데이터로 (미답변 목록이 휴리스틱 없이 JOIN)
    )
    session.add(assistant_message)
    await _touch_conversation(session, tenant_id, conversation_id, first_query=user_query)


async def _touch_conversation(session: AsyncSession, tenant_id: str, conversation_id: int,
                               first_query: str | None = None) -> None:
    """대화의 last_used_at 갱신 + 제목이 비어 있으면 첫 질문으로 세팅 (목록 표시용)."""
    values: dict = {'last_used_at': func.now()}
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .values(**values)
    )
    if first_query:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.tenant_id == tenant_id)
            .where(Conversation.title.is_(None))      # 첫 턴에만 — 이후 질문으로 안 바뀜
            .values(title=first_query[:80])
        )


async def add_pending_turn(
        session: AsyncSession,
        tenant_id: str,
        conversation_id: int,
        user_query: str,
        standalone_query: str,
        attachments: list[dict] | None = None,
        user_id: str | None = None,
) -> Message:
    """user 메시지 + 생성 대기 상태의 assistant 자리표시를 세션에 등록한다.

    생성 시작 전에 호출해 conversation_id·자리표시를 durable하게 만든다(persist-before-stream).
    assistant는 content='', status='generating'으로 시작하고, 완료 시 finalize_turn이 채운다.
    commit은 호출자가 담당한다. 반환: assistant 자리표시 Message (id는 flush 후 확정).
    """
    user_message = Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="user",
        content=user_query,
        standalone_query=standalone_query,
        attachments=attachments,
        user_id=user_id,
    )
    session.add(user_message)
    await session.flush()   # user id 확보 — 자리표시에 짝 FK를 처음부터 박는다
    assistant_message = Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="assistant",
        content="",
        status="generating",
        question_message_id=user_message.id,
    )
    session.add(assistant_message)
    await _touch_conversation(session, tenant_id, conversation_id, first_query=user_query)
    return assistant_message


async def finalize_turn(
        session: AsyncSession,
        tenant_id: str,
        assistant_message_id: int,
        answer: str,
        sources: list[dict],
        status: str = "done",
        latency_ms: int | None = None,
        cited_docs: list[str] | None = None,
        is_refusal: bool = False,
        intent: str | None = None,
) -> None:
    """생성 대기 assistant 자리표시를 최종 결과로 채운다 (id로 재조회 후 UPDATE).

    백그라운드 태스크가 '자기 세션'으로 호출하므로 객체 참조가 아닌 id로 재조회한다.
    tenant 소유를 확인(격리)하고, 없거나 남의 것이면 무시. commit은 호출자가 담당한다.
    실패/차단이면 status='failed'|'blocked', answer=''로 호출.
    """
    msg = await session.get(Message, assistant_message_id)
    if msg is None or msg.tenant_id != tenant_id:
        return
    msg.content = answer
    msg.sources = sources
    msg.status = status
    msg.latency_ms = latency_ms
    msg.cited_docs = cited_docs
    msg.is_refusal = is_refusal
    # [원복 필요] intent — models.py 매핑 주석과 세트. DB 반영 후 함께 해제
    # msg.intent = intent


def trim_messages_for_condense(messages: list, budget_tokens: int) -> list:
    """condense에 넘길 히스토리를 토큰 예산으로 자른다 (최신부터, 시간순 복원).

    답변 맥락(build_prior_turns)과 같은 예산 기준 — '대화 창 크기'의 단일 기준.
    긴 히스토리를 그대로 넘기면 이전 답변의 수치·조건이 재작성 질의에 주입된다
    (실측 2026-07-20: 30개 메시지에서 전제 보존 1/5).
    """
    selected, used = [], 0
    for m in reversed(messages):
        cost = estimate_tokens(m.content)
        if selected and used + cost > budget_tokens:   # 최소 1개는 보장
            break
        selected.append(m)
        used += cost
    return list(reversed(selected))


def build_prior_turns(messages: list, budget_tokens: int) -> list[dict]:
    """최근 메시지 목록에서 최종 답변 프롬프트용 Q/A 턴을 만든다.
    user 다음 assistant가 이어지는 쌍만 prior_turns에 포함한다.
    이 값은 검색 근거가 아니라 최종 답변의 대화 맥락 참고용이다.

    F100: '2턴 고정'에서 '토큰 예산'으로 — 최신 턴부터 budget 안에서 담는다.
    답변 길이가 가변이라 개수 고정은 예산을 넘거나 낭비할 수 있어 부피 기준으로 전환.
    """
    turns = []
    pending_question = None

    for message in messages:
        if message.role == "user":
            pending_question = message.content
        elif message.role == "assistant" and pending_question:
            turns.append({"q": pending_question, "a": message.content})
            pending_question = None

    # 최신 턴부터 역순으로 예산 소진까지 담고, 시간순으로 복원
    selected, used = [], 0
    for turn in reversed(turns):
        cost = estimate_tokens(turn["q"]) + estimate_tokens(turn["a"])
        if selected and used + cost > budget_tokens:   # 최소 1턴은 보장 (첫 턴은 예산 초과해도 포함)
            break
        selected.append(turn)
        used += cost
    return list(reversed(selected))

