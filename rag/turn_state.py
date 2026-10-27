"""턴 상태 기계 — 시작(add_pending_turn)·종료(finalize_turn)·회수(sweep_stale_generating)의 정의점.

상태 전이 (값 어휘의 정의점은 rag/models.py의 TurnStatus):
    (user)      → done                                          (항상)
    (assistant) generating → {done, cancelled, failed, blocked}  (모두 종결)

모든 함수가 session을 받고 commit하지 않는다 — 커밋 시점은 호출자가 정한다.
"""

import logging
from datetime import timedelta

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Attachment, Conversation, Message, TurnStatus

logger = logging.getLogger(__name__)

# 아래 파생 집합은 축이 독립이라 하나로 합치지 말 것 — blocked는 이력에서 빼지만 RETRY 대상이
# 아니고, cancelled·failed는 RETRY 대상이지만 이력에는 남는다.

# 프롬프트 이력에서 격리 — 소비: conversation.load_recent_messages.
# blocked: 차단 결정이 다음 턴 프롬프트로 재진입해 재해석되면 안 된다 (조회 API엔 그대로 노출).
# generating: 이번 턴 자신의 자리표시 — 자기 자신으로 되먹임되면 안 된다.
HISTORY_ISOLATED = (TurnStatus.BLOCKED, TurnStatus.GENERATING)

# SSE done 이벤트의 finish_reason 어휘 = 전체 − {generating} — 소비: streaming.TurnResult.
TERMINAL = tuple(s for s in TurnStatus if s is not TurnStatus.GENERATING)


# 답을 못 받은 턴 — RETRY의 대상. 사용자에겐 '멈췄든 오류가 났든' 같은 사건이다.
UNANSWERED = (TurnStatus.CANCELLED, TurnStatus.FAILED)


def last_unanswered_turn(messages: list[Message]) -> tuple[Message, Message] | None:
    """직전 턴이 답을 못 받은 턴이면 (실질 질문 user, 그 assistant)를 돌려준다 — RETRY 디스패치의 근거.

    blocked는 제외한다 — 차단 결정을 "다시"로 뒤집게 하면 안 된다. 페어링(user 바로 다음
    assistant)이 깨졌으면 None.
    "다시" 연타로 재실행 턴이 또 취소되면 user 행 content는 "다시"뿐이므로, 같은 standalone_query로
    이어진 연속 미응답 짝을 거슬러 올라가 체인 머리(실질 질문)의 user를 돌려준다.
    """
    if len(messages) < 2:
        return None
    last = messages[-1]
    if last.role != 'assistant' or last.status not in UNANSWERED:
        return None
    i = len(messages) - 2
    user = messages[i]
    if user.role != 'user' or user.id != last.question_message_id:
        return None
    while i >= 2:
        prev_a, prev_u = messages[i - 1], messages[i - 2]
        if not (
            prev_a.role == 'assistant'
            and prev_a.status in UNANSWERED
            and prev_u.role == 'user'
            and prev_u.id == prev_a.question_message_id
            and user.standalone_query is not None
            and prev_u.standalone_query == user.standalone_query
        ):
            break
        user = prev_u
        i -= 2
    return user, last


def _new_user_message(
    tenant_id: str, conversation_id: int, user_query: str, attachments: list[Attachment] | None, user_id: str | None
) -> Message:
    """user Message 생성. standalone_query는 턴 시작 시점엔 아직 모르므로 finalize_turn이 백필한다."""
    return Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="user",
        content=user_query,
        attachments=attachments,
        user_id=user_id,
    )


async def _touch_conversation(
    session: AsyncSession, tenant_id: str, conversation_id: int, first_query: str | None = None
) -> None:
    """대화의 last_used_at 갱신 + 제목이 비어 있으면 첫 질문으로 세팅 (목록 표시용)."""
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
        .values(last_used_at=func.now())
    )
    if first_query:
        await session.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .where(Conversation.tenant_id == tenant_id)
            .where(Conversation.title.is_(None))  # 첫 턴에만 — 이후 질문으로 안 바뀜
            .values(title=first_query[:80])
        )


async def add_pending_turn(
    session: AsyncSession,
    tenant_id: str,
    conversation_id: int,
    user_query: str,
    attachments: list[Attachment] | None = None,
    user_id: str | None = None,
) -> Message:
    """user 메시지 + generating 상태의 assistant 자리표시를 세션에 등록하고 자리표시를 반환한다.

    라우팅·condense·검색보다 먼저 호출해야 한다 — 그 뒤 어디서 실패해도 질문이 남게 하려는 것이다.
    user_query는 사용자가 친 원문이다(RETRY의 질의 치환은 이보다 뒤에 일어난다).
    """
    user_message = _new_user_message(tenant_id, conversation_id, user_query, attachments, user_id)
    session.add(user_message)
    await session.flush()  # user id 확보 — 자리표시에 짝 FK를 처음부터 박는다
    assistant_message = Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="assistant",
        content="",
        status=TurnStatus.GENERATING,
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
    intent: str | None = None,
    standalone_query: str | None = None,
    cache_kind: str | None = None,
    block_reason: str | None = None,
    cited_faq_ids: list[int] | None = None,
) -> None:
    """자리표시를 최종 결과로 채운다 — 모든 경로의 유일한 종료 지점.

    백그라운드 태스크가 자기 세션으로 호출하므로 id로 재조회한다. 없거나 남의 테넌트면 무시.
    standalone_query는 짝 user 행에 백필한다(condense 이후에야 확정되는 값이다).
    """
    status = TurnStatus(status)  # 어휘 밖 값은 저장 전에 ValueError
    msg = await session.get(Message, assistant_message_id)
    if msg is None or msg.tenant_id != tenant_id:
        return
    if msg.status != TurnStatus.GENERATING:
        # 스테일 스윕이 먼저 failed로 바꾼 뒤 태스크가 완주한 경우(좀비 플립). 완성된 답변은
        # 버리지 않고 덮어쓰되, 나중에 추적할 수 있게 경고를 남긴다.
        logger.warning(
            'finalize가 %s 상태를 %s로 덮어쓴다 (message_id=%s) — 스테일 스윕과 경쟁했을 수 있다',
            msg.status,
            status,
            assistant_message_id,
        )
    msg.content = answer
    msg.sources = sources
    msg.status = status
    msg.latency_ms = latency_ms
    msg.cited_docs = cited_docs
    msg.cited_faq_ids = cited_faq_ids
    msg.intent = intent
    msg.cache_kind = cache_kind  # 'semantic'=캐시 재생 답변
    msg.block_reason = block_reason

    # id만으로 유일하지만 쓰기 경로는 예외 없이 테넌트를 명시한다.
    if msg.question_message_id is not None:
        await session.execute(
            update(Message)
            .where(Message.id == msg.question_message_id)
            .where(Message.tenant_id == tenant_id)
            .values(standalone_query=standalone_query)
        )


# generating이 이 시간 넘게 지속되면 태스크·프로세스 사망으로 보고 failed 처리한다.
# LLM 호출 타임아웃(rag/llm/openai_compat.py)보다 확실히 커야 한다 — 살아 있는 요청을 먼저
# failed로 선고하면 좀비 플립이 생긴다.
GENERATION_STALE_SECONDS = 500


async def sweep_stale_generating(session: AsyncSession) -> int:
    """고착된 generating 턴을 failed로 정리하고 처리한 행 수를 반환한다.

    생성은 웹 프로세스의 asyncio 태스크라 프로세스가 죽으면 자리표시만 남는다. 회수는 워커
    cron(rag/worker.py)만 한다 — GET이 하면 열어본 대화만 치유된다. 상태만 바꾸고 아무것도
    노출하지 않으므로 전 테넌트 일괄이다. 시간 비교는 서버측 func.now()로 naive/aware 혼선을 피한다.
    """
    result = await session.execute(
        update(Message)
        .where(Message.status == TurnStatus.GENERATING)
        .where(Message.created_at < func.now() - timedelta(seconds=GENERATION_STALE_SECONDS))
        .values(status=TurnStatus.FAILED.value)
    )
    return result.rowcount
