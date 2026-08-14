import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy import func, or_, select, true, update
from sqlalchemy.ext.asyncio.session import AsyncSession

from database import get_session
from rag import cancellation
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

logger = logging.getLogger(__name__)

router = APIRouter(prefix='/kms')

# generating이 이 시간 넘게 지속되면 백그라운드 태스크/웹 프로세스 사망으로 고착 → failed 간주.
#
# 임계는 '정상 생성의 최대 소요'보다 **확실히 커야** 한다. 살아 있는 요청을 먼저 failed로
# 선고하면 그 태스크가 완주할 때 finalize_turn이 그 행을 done으로 덮어써(좀비 플립), 이미
# 실패로 보고된 턴이 조용히 성공이 되고 과거 통계까지 소급 변한다. LLM 호출 타임아웃이
# 300초(rag/llm.py)라 이전 값 300은 그 경계에 딱 붙어 있었다 — 동시성 포화로 vLLM 큐에서
# 밀리면 정상 요청이 스윕에 걸릴 수 있었다. 스윕은 '죽은 프로세스 회수'가 목적이라 늦게
# 정리해도 손해가 없으므로 여유를 뒀다.
GENERATION_STALE_SECONDS = 500

MAX_CONVERSATIONS = 10   # 페이지 크기 상한 (#10부터 offset 페이지네이션의 limit 상한 의미)

SNIPPET_RADIUS = 40      # 스니펫: 매칭 지점 전후 문자 수 (#28) — 목록 한 줄에 들어가는 폭

# LIKE 패턴에서 특수 의미를 갖는 문자. ESCAPE 절과 짝이며, 백슬래시가 먼저 나와야 한다
# (뒤에 두면 %·_를 이스케이프하며 넣은 백슬래시를 다시 이스케이프한다).
_LIKE_ESCAPE_CHAR = '\\'
_LIKE_SPECIALS = (_LIKE_ESCAPE_CHAR, '%', '_')


def _escape_like(raw: str) -> str:
    """사용자 입력을 LIKE 패턴 리터럴로 만든다 (#28).

    이스케이프하지 않으면 검색어의 %·_가 와일드카드로 동작한다 — 'a_b'로 검색하면
    'axb'까지 걸린다(실측). SQLAlchemy .ilike()는 자동 이스케이프를 하지 않으므로
    호출부가 escape=_LIKE_ESCAPE_CHAR를 함께 넘겨야 이 치환이 의미를 갖는다.
    """
    for ch in _LIKE_SPECIALS:
        raw = raw.replace(ch, _LIKE_ESCAPE_CHAR + ch)
    return raw


def _build_snippet(content: str, q: str) -> str | None:
    """매칭 지점 전후 SNIPPET_RADIUS자를 잘라낸 발췌 (#28). 잘린 쪽엔 말줄임을 붙인다.

    위치 탐색을 SQL의 position()이 아니라 파이썬에서 하는 이유: position()은 대소문자를
    구분해서 ILIKE로 매칭된 영문 검색어를 못 찾고 0을 반환한다 — 그러면 조용히 '항상 앞
    80자'를 반환하는 오동작이 된다(실측). 여기선 양쪽 다 소문자로 맞춰 그 함정이 없다.
    또 이스케이프한 SQL 패턴이 아니라 원본 q를 쓴다 — 두 경로가 아예 분리돼 섞일 수 없다.

    파이썬 str 슬라이싱은 코드포인트 단위라 한글도 '40자'가 글자 수 그대로 맞는다.
    """
    idx = content.lower().find(q.lower())
    if idx < 0:
        # ILIKE는 매칭했는데 여기선 못 찾는 경우(유니코드 케이스폴딩 차이 등) — 발췌를 포기한다.
        # 대화 자체는 결과에 남고 snippet만 비므로, 목록이 깨지지 않는다.
        return None
    start = max(0, idx - SNIPPET_RADIUS)
    end = min(len(content), idx + len(q) + SNIPPET_RADIUS)
    prefix = '…' if start > 0 else ''
    suffix = '…' if end < len(content) else ''
    return f'{prefix}{content[start:end]}{suffix}'


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


def _search_filter(tenant_id: str, pattern: str | None):
    """제목 또는 대화 내용 부분일치 필터 (#28). pattern=None이면 필터 없음(기존 목록 동작).

    메시지 매칭을 JOIN이 아니라 EXISTS로 거는 게 핵심이다 — JOIN이면 한 대화에서 두 건이
    매칭될 때 그 대화가 결과에 두 번 나오고 total도 부풀어 오른다(실측: EXISTS 2 vs JOIN 3).
    EXISTS는 세미조인이라 매칭 개수와 무관하게 대화 행이 정확히 하나다. 덕분에 DISTINCT나
    윈도우 함수 없이 "대화당 1행"이 저절로 성립한다.

    status로 거르지 않는다 — 차단 턴(blocked)도 검색에 걸려야 한다. 프롬프트 재료에서
    비정상 턴을 빼는 load_recent_messages와는 목적이 다르다(그건 맥락 오염 방지).
    attachments(첨부 추출 텍스트)는 건드리지 않는다 — 조회 API가 파일명만 노출하는 정책과
    같은 이유로, 고객 개인 문서 본문이 검색으로 새면 안 된다.
    """
    if pattern is None:
        return true()
    return or_(
        Conversation.title.ilike(pattern, escape=_LIKE_ESCAPE_CHAR),
        select(Message.id)
        .where(Message.conversation_id == Conversation.id)
        .where(Message.tenant_id == tenant_id)      # 격리 — 상관 서브쿼리에도 WHERE 명시
        .where(Message.content.ilike(pattern, escape=_LIKE_ESCAPE_CHAR))
        .exists(),
    )


async def _snippets_for(session: AsyncSession, tenant_id: str, conversation_ids: list[int],
                        pattern: str, q: str) -> dict[int, str | None]:
    """페이지에 실린 대화들의 발췌를 한 번의 쿼리로 모아온다 (#28).

    대화당 '가장 최근 매칭 메시지'를 골라야 하는데, (conversation_id, 최신순) 정렬로 받아
    각 대화의 첫 등장만 취하면 된다 — setdefault가 그 역할. created_at 동률일 때를 위해
    id 보조정렬을 함께 쓴다(한 턴의 user·assistant가 같은 커밋이라 시각이 같다).

    알려진 한계: 대화당 LIMIT을 걸지 않아 매칭이 많은 대화는 그 행을 전부 실어온다.
    윈도우 함수(row_number)나 LATERAL이면 DB에서 1건으로 줄일 수 있지만, 리포에 전례가
    없는 구문이라 현 데이터 규모에서는 단순함을 택했다 — 느려지면 그때 교체할 것.
    """
    rows = (await session.execute(
        select(Message.conversation_id, Message.content)
        .where(Message.conversation_id.in_(conversation_ids))
        .where(Message.tenant_id == tenant_id)      # 격리 — 메시지에도 WHERE 명시
        .where(Message.content.ilike(pattern, escape=_LIKE_ESCAPE_CHAR))
        .order_by(Message.conversation_id, Message.created_at.desc(), Message.id.desc())
    )).all()

    latest: dict[int, str] = {}
    for conversation_id, content in rows:
        latest.setdefault(conversation_id, content)
    return {cid: _build_snippet(content, q) for cid, content in latest.items()}


@router.get('/conversations', response_model=ConversationListResponse)
async def list_conversations(
    limit: int = MAX_CONVERSATIONS,
    offset: int = 0,
    q: str | None = None,
    tenant_id: str = Depends(get_tenant_id),
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
):
    """내 대화 목록 (최근 사용순, offset 페이지네이션 #10) + 제목·내용 검색 (#28).

    q를 주면 제목 또는 대화 내용에 부분일치하는 대화만 남기고, 내용에서 걸린 대화엔
    발췌(snippet)를 함께 내려준다. q는 필터일 뿐이라 정렬·페이지 크기는 그대로다 —
    "최근 10개 안에서 찾는" 게 아니라 소유 대화 전체에서 찾아 최근순 10개를 준다.

    total은 페이지 쿼리와 같은 WHERE에 count만 씌운 값이다. 두 쿼리가 같은 필터를
    공유해야 has_more가 어긋나지 않으므로 조건은 반드시 한 곳(where)에서 만들어 쓴다.
    알려진 한계: last_used_at 정렬이 가변이라 스크롤 중 페이지 밀림 가능 — 이 UI에선 감수.
    """
    # 범위 밖(<=0 or 상한 초과)은 상한값으로 클램프 (거부 대신 sane 결과 — 음수 500·무상한 조회 방지, P2)
    if not 1 <= limit <= MAX_CONVERSATIONS:
        limit = MAX_CONVERSATIONS
    if offset < 0:
        offset = 0
    # 빈 문자열·공백뿐인 q는 미전송과 동일 취급 — '%%'는 전건 매칭이라 필터를 건 척만 하게 된다
    q = (q or '').strip() or None
    pattern = f'%{_escape_like(q)}%' if q else None

    where = _owned(tenant_id, user_id) & _search_filter(tenant_id, pattern)
    total = (await session.execute(
        select(func.count(Conversation.id)).where(where)
    )).scalar_one()
    rows = (await session.execute(
        select(Conversation)
        .where(where)
        .order_by(Conversation.last_used_at.desc())
        .offset(offset)
        .limit(limit)
    )).scalars().all()

    # 제목에서만 걸린 대화는 여기서 값이 안 나와 snippet=None이 된다 (제목은 이미 목록에 보임)
    snippets = (await _snippets_for(session, tenant_id, [c.id for c in rows], pattern, q)
                if q and rows else {})

    return ConversationListResponse(
        items=[ConversationSummary(conversation_id=c.id, title=c.title, updated_at=c.last_used_at,
                                   snippet=snippets.get(c.id))
               for c in rows],
        # limit+1 조회 트릭은 제거 — total이 있으면 다음 페이지 유무가 산술로 나온다.
        # len(rows)를 쓰는 게 limit보다 안전하다(마지막 부분 페이지에서도 정확).
        has_more=offset + len(rows) < total,
        total=total,
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
):
    """답변 피드백 👍/👎 + 사유 태그 저장 (#8) — 멱등 set, 토글/취소는 FE가 최종 상태를 보내는 방식.

    assistant 메시지 + 본인 소유 대화만 허용 — 아니면 전부 404 (존재 여부 비노출,
    _get_owned_conversation과 같은 원칙). 👍/취소 시 태그·텍스트는 강제 NULL (👎 전용 축).
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
    msg.feedback_text = body.text if body.feedback is False else None
    await session.commit()
    return MessageFeedbackState(message_id=msg.id, feedback=msg.feedback,
                                feedback_tag=msg.feedback_tag, feedback_text=msg.feedback_text)


@router.post('/messages/{message_id}/cancel', status_code=204)
async def cancel_generation(
        message_id: int,
        tenant_id: str = Depends(get_tenant_id),
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session),
):
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
    msg = (await session.execute(
        select(Message)
        .join(Conversation, Message.conversation_id == Conversation.id)
        .where(Message.id == message_id)
        .where(Message.tenant_id == tenant_id)     # 격리 — 메시지에도 tenant WHERE 명시
        .where(Message.role == 'assistant')
        .where(_owned(tenant_id, user_id))
    )).scalar_one_or_none()
    if msg is None:
        raise HTTPException(status_code=404, detail='메시지를 찾을 수 없습니다.')

    # 소유가 확인됐으므로 이제 레지스트리를 봐도 안전하다. 상태보다 먼저 보는 이유: 300초 스테일
    # 스윕이 '정말 진행 중인' 생성을 failed로 바꿔놓을 수 있는데(느린 GPU·동시성 포화. LLM
    # 타임아웃이 300초라 도달 가능한 구간이다), 상태만 믿으면 살아 있는 태스크를 멈출 방법이
    # 사라진다. 태스크가 손에 있으면 DB가 뭐라 하든 멈추는 게 사용자 의사에 맞다.
    if cancellation.cancel_local(message_id):
        return Response(status_code=204)           # 이 프로세스가 들고 있었다
    if msg.status == 'cancelled':
        return Response(status_code=204)           # 따닥 두 번째 — 결과가 같으니 성공으로 (멱등)
    if msg.status != 'generating':
        # done·blocked — 멈출 게 없다. 즉시 경로(캐시히트 등)도 여기로 온다(태스크가 없음).
        raise HTTPException(status_code=404, detail='진행 중인 생성이 아닙니다.')

    # 다른 인스턴스 소유로 추정 — 발행만 하고 도달은 보장하지 않는다.
    try:
        await cancellation.request_cancel(message_id)
    except Exception:
        # Redis 순단. 202("접수했다")로 답하면 거짓이 되고 500은 원인을 감춘다 —
        # 재시도 가능한 상황임을 알린다. 구독측(subscribe_forever)은 재연결하지만
        # 발행측은 요청 단위라 재시도 주체가 클라이언트다.
        logger.exception('취소 신호 발행 실패 (message_id=%s)', message_id)
        raise HTTPException(status_code=503, detail='취소 요청을 전달할 수 없습니다. 잠시 후 다시 시도해 주세요.')
    return Response(status_code=202)


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
