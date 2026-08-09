from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio.session import AsyncSession

from database import get_session
from rag.conversation import DEFAULT_USER
from rag.models import Conversation, Message
from routers.kms import get_tenant_id, get_user_id
from schemas.conversations import (
    ConversationListResponse,
    ConversationMessage,
    ConversationSummary,
    ConversationTitleUpdate,
    MessageFeedbackState,
    MessageFeedbackUpdate,
)

router = APIRouter(prefix='/kms')

# generating이 이 시간 넘게 지속되면 백그라운드 태스크/웹 프로세스 사망으로 고착 → failed 간주.
# (정상 생성은 수십 초. 5분은 넉넉한 "확실히 죽음" 임계)
GENERATION_STALE_SECONDS = 300

MAX_CONVERSATIONS = 10   # 페이지 크기 상한 (#10부터 offset 페이지네이션의 limit 상한 의미)


def _owned(tenant_id: str, user_id: str | None):
    """대화 소유 필터 (#10) — tenant + created_by + 미삭제 (이 라우터의 조회 경로 공통).

    같은 규칙이 rag/conversation.py ensure_conversation(질의 경로, ORM 인스턴스 비교)에도
    있다 — 소유권 규칙을 바꾸면 두 곳을 함께 고칠 것.
    created_by NULL인 기존 개발 데이터는 어느 사용자와도 불일치 → 자연 미노출.
    """
    return (
        (Conversation.tenant_id == tenant_id)
        & (Conversation.created_by == (user_id or DEFAULT_USER))
        & (Conversation.deleted_at.is_(None))
    )


async def _get_owned_conversation(
        session: AsyncSession, conversation_id: int, tenant_id: str, user_id: str | None,
) -> Conversation:
    """소유 대화 로드 — 없거나 남의 것이거나 삭제됐으면 404 (존재 여부 노출 안 함)."""
    conv = (await session.execute(
        select(Conversation).where(Conversation.id == conversation_id).where(_owned(tenant_id, user_id))
    )).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail='대화를 찾을 수 없습니다.')
    return conv


@router.get('/conversations', response_model=ConversationListResponse)
async def list_conversations(
    limit: int = MAX_CONVERSATIONS,
    offset: int = 0,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
):
    """내 대화 목록 (최근 사용순, offset 페이지네이션 #10).

    limit+1개를 조회해 has_more 판정 (count 쿼리 없이 다음 페이지 유무 확인).
    알려진 한계: last_used_at 정렬이 가변이라 스크롤 중 페이지 밀림 가능 — 이 UI에선 감수.
    """
    # 범위 밖(<=0 or 상한 초과)은 상한값으로 클램프 (거부 대신 sane 결과 — 음수 500·무상한 조회 방지, P2)
    if not 1 <= limit <= MAX_CONVERSATIONS:
        limit = MAX_CONVERSATIONS
    if offset < 0:
        offset = 0
    rows = (await session.execute(
        select(Conversation)
        .where(_owned(tenant_id, user_id))
        .order_by(Conversation.last_used_at.desc())
        .offset(offset)
        .limit(limit + 1)
    )).scalars().all()

    return ConversationListResponse(
        items=[ConversationSummary(conversation_id=c.id, title=c.title, updated_at=c.last_used_at)
               for c in rows[:limit]],
        has_more=len(rows) > limit,
    )


@router.get('/conversations/{conversation_id}/messages', response_model=list[ConversationMessage])
async def get_conversation_messages(
        conversation_id: int,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    # 소유 검증 (#10) — 이전엔 메시지 tenant 필터뿐이라 같은 테넌트 남의 대화도 조회 가능했음
    await _get_owned_conversation(session, conversation_id, tenant_id, user_id)

    # lazy 스윕: 오래 고착된 generating을 failed로 자기치유 (태스크가 소유하지만 웹 프로세스
    # 사망 시 아무도 finalize 못 하므로, 조회 시점에 정리. 정상 진행 중인 건 최근이라 미매칭).
    # 시간 비교는 서버측(func.now())으로 — created_at 컬럼이 naive/aware 혼선을 피한다.
    res = await session.execute(
        update(Message)
        .where(Message.tenant_id == tenant_id)
        .where(Message.conversation_id == conversation_id)
        .where(Message.status == "generating")
        .where(Message.created_at < func.now() - timedelta(seconds=GENERATION_STALE_SECONDS))
        .values(status="failed")
    )
    if res.rowcount:
        await session.commit()

    msgs = (await session.execute(
        select(Message)
        .where(Message.tenant_id == tenant_id)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.id)
    )).scalars().all()
    return [
        ConversationMessage(
            message_id=m.id,
            role=m.role,
            content=m.content,
            status=m.status,
            sources=m.sources,
            # 첨부 본문은 노출하지 않고 파일명만 (FE 뱃지/말풍선 표시용)
            attachments=[a['filename'] for a in m.attachments] if m.attachments else None,
            cited_docs=m.cited_docs,
            feedback=m.feedback,
            feedback_tag=m.feedback_tag,
        )
        for m in msgs
    ]


@router.patch('/messages/{message_id}/feedback', response_model=MessageFeedbackState)
async def set_message_feedback(
        message_id: int,
        body: MessageFeedbackUpdate,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    """답변 피드백 👍/👎 + 사유 태그 저장 (#8) — 멱등 set, 토글/취소는 FE가 최종 상태를 보내는 방식.

    assistant 메시지 + 본인 소유 대화만 허용 — 아니면 전부 404 (존재 여부 비노출,
    _get_owned_conversation과 같은 원칙). 👍/취소 시 태그는 강제 NULL (👎 전용 축).
    """
    msg = (await session.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.id == message_id)
        .where(Message.tenant_id == tenant_id)     # 격리 — 메시지에도 tenant WHERE 명시
        .where(Message.role == 'assistant')
        .where(Message.status == 'done')           # 실패/차단/생성중 턴엔 평가할 답변이 없음 — 집계 오염 방지
        .where(_owned(tenant_id, user_id))
    )).scalar_one_or_none()
    if msg is None:
        raise HTTPException(status_code=404, detail='메시지를 찾을 수 없습니다.')

    msg.feedback = body.feedback
    msg.feedback_tag = body.tag if body.feedback is False else None
    await session.commit()
    return MessageFeedbackState(message_id=msg.id, feedback=msg.feedback, feedback_tag=msg.feedback_tag)


@router.patch('/conversations/{conversation_id}', response_model=ConversationSummary)
async def rename_conversation(
        conversation_id: int,
        body: ConversationTitleUpdate,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    """제목 변경 (#10). 자동 제목은 title IS NULL일 때만 세팅되므로 이후 턴이 덮어쓰지 않는다."""
    conv = await _get_owned_conversation(session, conversation_id, tenant_id, user_id)
    conv.title = body.title
    await session.commit()
    return ConversationSummary(conversation_id=conv.id, title=conv.title, updated_at=conv.last_used_at)


@router.delete('/conversations/{conversation_id}', status_code=204)
async def delete_conversation(
        conversation_id: int,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
    """소프트 삭제 (#10) — deleted_at 마킹. 이력(메시지·첨부)은 감사 목적 보존.

    generating 중 삭제돼도 row가 남아 백그라운드 finalize가 정상 동작 (엣지 검증 완료 — 이슈 참조).
    """
    conv = await _get_owned_conversation(session, conversation_id, tenant_id, user_id)
    conv.deleted_at = func.now()
    await session.commit()
    return Response(status_code=204)
