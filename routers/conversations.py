from datetime import timedelta

from fastapi import APIRouter, Depends
from sqlalchemy import select, update, func
from sqlalchemy.ext.asyncio.session import AsyncSession

from database import get_session
from rag.models import Conversation, Message
from routers.kms import get_tenant_id
from schemas.conversations import ConversationSummary, ConversationMessage

router = APIRouter(prefix='/kms')

# generating이 이 시간 넘게 지속되면 백그라운드 태스크/웹 프로세스 사망으로 고착 → failed 간주.
# (정상 생성은 수십 초. 5분은 넉넉한 "확실히 죽음" 임계)
GENERATION_STALE_SECONDS = 300

MAX_CONVERSATIONS = 10   # 최근 대화 목록 상한

@router.get('/conversations', response_model=list[ConversationSummary])
async def list_conversations(
    limit: int = MAX_CONVERSATIONS,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
):
    """최근 대화 N개. 각 대화의 첫 사용자 질문을 미리보기 제목으로."""
    # 범위 밖(<=0 or 상한 초과)은 상한값으로 클램프 (거부 대신 sane 결과 — 음수 500·무상한 조회 방지, P2)
    if not 1 <= limit <= MAX_CONVERSATIONS:
        limit = MAX_CONVERSATIONS
    conversations = (await session.execute(
        select(Conversation)
        .where(Conversation.tenant_id == tenant_id)
        .order_by(Conversation.last_used_at.desc())
        .limit(limit)
    )).scalars().all()

    return [ConversationSummary(conversation_id=c.id, title=c.title) for c in conversations]

@router.get('/conversations/{conversation_id}/messages', response_model=list[ConversationMessage])
async def get_conversation_messages(
        conversation_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session),
):
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
            role=m.role,
            content=m.content,
            status=m.status,
            sources=m.sources,
            # 첨부 본문은 노출하지 않고 파일명만 (FE 뱃지/말풍선 표시용)
            attachments=[a['filename'] for a in m.attachments] if m.attachments else None,
            cited_docs=m.cited_docs,
        )
        for m in msgs
    ]
