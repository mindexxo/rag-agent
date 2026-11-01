"""대화 이력 관리 — 소유 대화·메시지 조회, 목록(검색·발췌), 메시지 이력과 실인용 필터, 피드백·제목·삭제.

소유 판정은 rag/turn/conversation.py의 owned_filter 한 곳이다 — 없거나 남의 것이거나 삭제된 대화는 전부
ConversationNotFound(존재 여부를 노출하지 않는다). HTTP 변환은 라우터가 한다.
"""

from typing import Any, NamedTuple

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Conversation, Message
from rag.turn import conversation_search
from rag.turn.conversation import ConversationNotFound, owned_filter
from schemas.conversations import MessageFeedbackUpdate


class Owner(NamedTuple):
    tenant_id: str
    user_id: str | None  # None은 DEFAULT_USER로 본다(owned_filter)


class ConversationPage(NamedTuple):
    conversations: list[Conversation]
    total: int
    snippets: dict[int, str | None]  # 내용에서 걸린 대화의 발췌 — 제목에서만 걸린 대화는 없다


async def get_owned_conversation(session: AsyncSession, owner: Owner, conversation_id: int) -> Conversation:
    conv = (
        await session.execute(
            select(Conversation).where(Conversation.id == conversation_id).where(owned_filter(*owner))
        )
    ).scalar_one_or_none()
    if conv is None:
        raise ConversationNotFound(f'Conversation {conversation_id} not found')
    return conv


async def get_owned_assistant_message(
    session: AsyncSession, owner: Owner, message_id: int, *, status: str | None = None
) -> Message | None:
    """소유 검증을 통과한 assistant 메시지 — 피드백·취소·재접속 공용. owned_filter로 대화 소유를, tenant WHERE로
    메시지 격리를 함께 건다(messages.id는 전 테넌트 공용 시퀀스라 둘 다 필요하다)."""
    stmt = (
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.id == message_id)
        .where(Message.tenant_id == owner.tenant_id)  # 격리 — 메시지에도 tenant WHERE 명시
        .where(Message.role == 'assistant')
        .where(owned_filter(*owner))
    )
    if status is not None:
        stmt = stmt.where(Message.status == status)
    return (await session.execute(stmt)).scalar_one_or_none()


async def list_conversations(
    session: AsyncSession, owner: Owner, q: str | None, limit: int, offset: int
) -> ConversationPage:
    """최근 사용순 + 제목·내용 검색. q는 필터일 뿐이라 정렬·페이지 크기는 그대로다 — 소유 대화 전체에서 찾아
    최근순으로 준다. count와 페이지 쿼리는 같은 WHERE를 써야 has_more가 어긋나지 않는다.
    limit·offset·q 위생은 호출부가 했다(q는 빈 값이 아닌 str 또는 None).
    """
    where = owned_filter(*owner) & conversation_search.search_filter(owner.tenant_id, q)
    total = (await session.execute(select(func.count(Conversation.id)).where(where))).scalar_one()
    rows = list(
        (
            await session.execute(
                select(Conversation)
                .where(where)
                .order_by(
                    Conversation.last_used_at.desc(), Conversation.id.desc()
                )  # 동점 2차 키 — 페이지 간 중복·누락 방지
                .offset(offset)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    snippets = (
        await conversation_search.snippets_for(session, owner.tenant_id, [c.id for c in rows], q) if q and rows else {}
    )
    return ConversationPage(rows, total, snippets)


async def conversation_messages(session: AsyncSession, owner: Owner, conversation_id: int) -> list[Message]:
    # 소유 검증 — 메시지 tenant 필터만으로는 같은 테넌트 남의 대화가 조회된다
    await get_owned_conversation(session, owner, conversation_id)
    return list(
        (
            await session.execute(
                select(Message)
                .where(Message.tenant_id == owner.tenant_id)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.id)
            )
        )
        .scalars()
        .all()
    )


def cited_sources(m: Message) -> list[dict[str, Any]] | None:
    """이력에 보일 실인용 — sources(후보 스냅샷)를 cited_docs·cited_faq_ids로 거른다. 구 데이터는 sources가 후보 전체라
    필터가 실질이고, 신 데이터는 저장 자체가 인용만이라 no-op이다. 파일명 필터만 두면 FAQ 인용이 이력에서 사라진다."""
    return [
        s
        for s in (m.sources or [])
        if s['filename'] in (m.cited_docs or [])
        or (s.get('faq_id') is not None and s['faq_id'] in (m.cited_faq_ids or []))
    ] or None


async def set_feedback(session: AsyncSession, msg: Message, body: MessageFeedbackUpdate) -> Message:
    """멱등 set — 토글·취소는 FE가 최종 상태를 보낸다. 👍/취소면 태그·텍스트는 강제 NULL(👎 전용 축)."""
    msg.feedback = body.feedback
    msg.feedback_tag = body.tag if body.feedback is False else None
    msg.feedback_text = body.text if body.feedback is False else None
    await session.commit()
    return msg


async def rename_conversation(session: AsyncSession, conv: Conversation, title: str) -> Conversation:
    """자동 제목은 title IS NULL일 때만 세팅되므로 이후 턴이 덮어쓰지 않는다."""
    conv.title = title
    await session.commit()
    return conv


async def delete_conversation(session: AsyncSession, conv: Conversation) -> None:
    """소프트 삭제 — deleted_at 마킹. 이력(메시지·첨부)은 감사 목적으로 보존하고, 생성 중 삭제돼도 행이 남아
    백그라운드 마무리가 정상 동작한다."""
    conv.deleted_at = func.now()
    await session.commit()
