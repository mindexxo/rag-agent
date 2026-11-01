"""대화 API — 목록·메시지 이력·피드백·이름 변경·삭제, 생성 취소와 SSE 재접속."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio.session import AsyncSession
from starlette.responses import StreamingResponse

from database import get_session
from rag import clients
from rag.models import Message, TurnStatus
from rag.turn import cancellation, history, stream_resume, streaming
from routers.deps import get_tenant_id, get_user_id, sanitize_page
from routers.errors import conversation_http_errors
from schemas.conversations import (
    CONVERSATION_SEARCH_MAX,
    ConversationListResponse,
    ConversationMessage,
    ConversationSummary,
    ConversationTitleUpdate,
    MessageFeedbackState,
    MessageFeedbackUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/kms')

MAX_CONVERSATIONS = 50  # offset 페이지네이션의 limit 상한


async def _owned_assistant_message(
    session: AsyncSession, owner: history.Owner, message_id: int, *, status: str | None = None
) -> Message:
    """피드백·취소·재접속 공용 — 없거나 남의 것이면 전부 404(존재 여부 비노출)."""
    msg = await history.get_owned_assistant_message(session, owner, message_id, status=status)
    if msg is None:
        raise HTTPException(status_code=404, detail='메시지를 찾을 수 없습니다.')
    return msg


@router.get('/conversations', response_model=ConversationListResponse)
async def list_conversations(
    limit: int = MAX_CONVERSATIONS,
    offset: int = 0,
    q: str | None = Query(None, max_length=CONVERSATION_SEARCH_MAX),
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> ConversationListResponse:
    """내 대화 목록 (최근 사용순, offset 페이지네이션 #10) + 제목·내용 검색 (#28).

    q를 주면 제목 또는 대화 내용에 부분일치하는 대화만 남기고, 내용에서 걸린 대화엔
    발췌(snippet)를 함께 내려준다. q는 필터일 뿐이라 정렬·페이지 크기는 그대로다 —
    "최근 10개 안에서 찾는" 게 아니라 소유 대화 전체에서 찾아 최근순 10개를 준다.

    total은 페이지 쿼리와 같은 WHERE에 count만 씌운 값이다. 두 쿼리가 같은 필터를
    공유해야 has_more가 어긋나지 않으므로 조건은 반드시 한 곳(where)에서 만들어 쓴다.
    알려진 한계: last_used_at 정렬이 가변이라 스크롤 중 페이지 밀림 가능 — 이 UI에선 감수.
    """
    limit, offset, q = sanitize_page(limit, offset, q, MAX_CONVERSATIONS)
    page = await history.list_conversations(session, history.Owner(tenant_id, user_id), q, limit, offset)
    return ConversationListResponse(
        items=[
            ConversationSummary(
                conversation_id=c.id, title=c.title, updated_at=c.last_used_at, snippet=page.snippets.get(c.id)
            )
            for c in page.conversations
        ],
        # len(conversations)를 쓴다 — limit을 쓰면 마지막 부분 페이지에서 어긋난다.
        has_more=offset + len(page.conversations) < page.total,
        total=page.total,
    )


@router.get('/conversations/{conversation_id}/messages', response_model=list[ConversationMessage])
async def get_conversation_messages(
    conversation_id: int,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> list[ConversationMessage]:
    with conversation_http_errors():
        msgs = await history.conversation_messages(session, history.Owner(tenant_id, user_id), conversation_id)
    return [
        ConversationMessage(
            message_id=m.id,
            role=m.role,
            content=m.content,
            status=m.status,
            citations=history.cited_sources(m),
            # 첨부 본문은 노출하지 않고 파일명만 (FE 뱃지/말풍선 표시용)
            attachments=[a['filename'] for a in m.attachments] if m.attachments else None,
            feedback=m.feedback,
            feedback_tag=m.feedback_tag,
            feedback_text=m.feedback_text,
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
) -> MessageFeedbackState:
    """답변 피드백 👍/👎 + 사유 태그 저장 (#8) — 멱등 set, 토글/취소는 FE가 최종 상태를 보내는 방식.

    assistant 메시지 + 본인 소유 대화만 허용 — 아니면 전부 404 (존재 여부 비노출,
    _get_owned_conversation과 같은 원칙). 👍/취소 시 태그·텍스트는 강제 NULL (👎 전용 축).
    """
    # status='done' — 실패/차단/생성중 턴엔 평가할 답변이 없음 (집계 오염 방지)
    msg = await _owned_assistant_message(session, history.Owner(tenant_id, user_id), message_id, status=TurnStatus.DONE)
    msg = await history.set_feedback(session, msg, body)
    return MessageFeedbackState(
        message_id=msg.id, feedback=msg.feedback, feedback_tag=msg.feedback_tag, feedback_text=msg.feedback_text
    )


@router.post('/messages/{message_id}/cancel', status_code=204)
async def cancel_generation(
    message_id: int,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """진행 중인 답변 생성을 중단한다 (#30) — 정지 버튼.

    연결 끊김(탭 닫기·순단)과는 다르게 취급한다. 끊김은 의사가 불명해서 생성을 완주시켜
    저장하지만(#26), 이 엔드포인트는 명시적 의사 표현이므로 실제로 멈추고 GPU를 회수한다.
    실익은 UX보다 처리량이다 — 버려질 생성이 동시 상한 슬롯을 물고 있으면 다른 상담원이 막힌다.

    **concurrency_guard를 거치지 않는다** — 429가 나는 상황이야말로 취소가 가장 필요한 때인데
    취소 요청까지 슬롯을 요구하면 정작 못 멈춘다.

    응답: 204=이 프로세스에서 취소했거나 이미 취소된 턴(멱등) / 202=다른 인스턴스 소유로
    추정해 신호만 발행(도달 보장 없음) / 404=취소할 대상이 아님.
    """
    # 소유·상태 검증을 **반드시 먼저** 한다. 레지스트리는 message_id만 키로 쓰고 tenant를
    # 모르므로, 검증 없이 건드리면 남의 테넌트 생성을 id 추측만으로 죽일 수 있다 —
    # messages.id는 전 테넌트 공용 시퀀스라 순차 추측이 쉽다.
    # 이 await가 pop-then-cancel의 원자성을 깨지 않는다: 그 원자성은 cancel_local 내부의
    # pop↔cancel 사이에 await가 없다는 성질이고, 두 요청이 여기를 함께 통과해도 pop은 한쪽만
    # 성공한다(다른 쪽은 발행 경로로 빠져 헛수고 한 번을 할 뿐).
    msg = await _owned_assistant_message(session, history.Owner(tenant_id, user_id), message_id)

    # 소유가 확인됐으므로 이제 레지스트리를 봐도 안전하다. 상태보다 먼저 보는 이유: 스테일
    # 스윕(rag.turn.turn_state.GENERATION_STALE_SECONDS, cron 1분 주기)이 '정말 진행 중인'
    # 생성을 failed로 바꿔놓을 수 있는데(느린 GPU·동시성 포화 — LLM 타임아웃 300초보다
    # 임계가 크지만 초과 대기는 가능하다), 상태만 믿으면 살아 있는 태스크를 멈출 방법이
    # 사라진다. 태스크가 손에 있으면 DB가 뭐라 하든 멈추는 게 사용자 의사에 맞다.
    if cancellation.cancel_local(message_id):
        return Response(status_code=204)  # 이 프로세스가 들고 있었다
    if msg.status == TurnStatus.CANCELLED:
        return Response(status_code=204)  # 따닥 두 번째 — 결과가 같으니 성공으로 (멱등)
    if msg.status != TurnStatus.GENERATING:
        # done·blocked — 멈출 게 없다. 즉시 경로(캐시히트 등)도 여기로 온다(태스크가 없음).
        raise HTTPException(status_code=404, detail='진행 중인 생성이 아닙니다.')

    # 다른 인스턴스 소유로 추정 — 발행만 하고 도달은 보장하지 않는다.
    try:
        await cancellation.request_cancel(message_id)
    except Exception as e:
        # Redis 순단. 202("접수했다")로 답하면 거짓이 되고 500은 원인을 감춘다 —
        # 재시도 가능한 상황임을 알린다. 구독측(subscribe_forever)은 재연결하지만
        # 발행측은 요청 단위라 재시도 주체가 클라이언트다.
        logger.exception('취소 신호 발행 실패 (message_id=%s)', message_id)
        raise HTTPException(
            status_code=503, detail='취소 요청을 전달할 수 없습니다. 잠시 후 다시 시도해 주세요.'
        ) from e
    return Response(status_code=202)


@router.get('/messages/{message_id}/events')
async def resume_events(
    message_id: int,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """진행 중인 생성에 재접속해 스트림을 이어받는다 (#75) — 새로고침 복구용.

    쌓인 이벤트를 처음부터 재생하고, 아직 끝나지 않았으면 실시간으로 이어 붙인다.
    최초 연결(/kms/query)은 인메모리 큐로 흐르고 이 경로만 Redis Stream을 읽는다 —
    같은 인스턴스에 붙어도 큐를 재사용할 수 없다(1:1 소비라 원 리더와 토큰을 나눠 갖고,
    지나간 분량이 큐에 없어 재생할 재료도 없다). 그래서 인스턴스 수와 무관하게 동작한다.

    **소유 검증이 여기서 끝나야 한다** — messages.id는 전 테넌트 공용 시퀀스라, 빠뜨리면
    id 추측만으로 남의 생성을 실시간으로 엿볼 수 있다 (취소 엔드포인트와 같은 함정).

    status 필터를 걸지 않는 이유: 이력 조회와 구독 사이에 생성이 끝나는 레이스가 있다.
    그 창에 걸려도 TTL 안이면 전체 재생 + done으로 매끄럽게 끝난다. 스트림이 없으면(TTL
    만료·즉시 경로·Redis 쓰기 실패) 404 — FE는 이력 재조회(폴링)로 강등하면 된다.
    """
    await _owned_assistant_message(session, history.Owner(tenant_id, user_id), message_id)

    key = stream_resume.stream_key(tenant_id, message_id)
    try:
        exists = await clients.shared_redis.exists(key)  # 호출 시점 속성 읽기 (테스트가 교체한다)
    except Exception as e:
        # 쓰기와 달리 읽기 실패는 삼키지 않는다 — 여기가 실패하면 이 요청은 존재 이유를 잃는다.
        logger.exception('재접속 스트림 조회 실패 (message_id=%s)', message_id)
        raise HTTPException(status_code=503, detail='일시적으로 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.') from e
    if not exists:
        raise HTTPException(status_code=404, detail='재생할 스트림이 없습니다.')

    return StreamingResponse(
        stream_resume.reconnect_reader(tenant_id, message_id),
        media_type='text/event-stream',
        headers=streaming.SSE_HEADERS,
    )


@router.patch('/conversations/{conversation_id}', response_model=ConversationSummary)
async def rename_conversation(
    conversation_id: int,
    body: ConversationTitleUpdate,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> ConversationSummary:
    """제목 변경 (#10). 자동 제목은 title IS NULL일 때만 세팅되므로 이후 턴이 덮어쓰지 않는다."""
    with conversation_http_errors():
        conv = await history.get_owned_conversation(session, history.Owner(tenant_id, user_id), conversation_id)
    conv = await history.rename_conversation(session, conv, body.title)
    return ConversationSummary(conversation_id=conv.id, title=conv.title, updated_at=conv.last_used_at)


@router.delete('/conversations/{conversation_id}', status_code=204)
async def delete_conversation(
    conversation_id: int,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> Response:
    """소프트 삭제 (#10) — deleted_at 마킹. 이력(메시지·첨부)은 감사 목적 보존.

    generating 중 삭제돼도 row가 남아 백그라운드 finalize가 정상 동작 (엣지 검증 완료 — 이슈 참조).
    """
    with conversation_http_errors():
        conv = await history.get_owned_conversation(session, history.Owner(tenant_id, user_id), conversation_id)
    await history.delete_conversation(session, conv)
    return Response(status_code=204)
