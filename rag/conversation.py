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

from rag import otel
from rag.llm import LlmClient
from rag.models import Conversation, Message

logger = logging.getLogger(__name__)
from rag.tokens import estimate_tokens
from rag.prompt_texts import CONDENSE_MULTI_SYSTEM_PROMPT, CONDENSE_SYSTEM_PROMPT
from rag.prompts import build_chat_prompt, build_condense_user_message


# X-User-Id 미전송 시 created_by·스코핑에 쓰는 폴백 (#10).
# 헤더 필수화는 ICCS 통합의 정식 인증으로 자연 해결 — 지금은 폴백으로 일관 저장·일관 필터.
DEFAULT_USER = 'test-user'


def owned_filter(tenant_id: str, user_id: str | None):
    """대화 소유 판정 WHERE 표현식 — tenant + created_by + 미삭제 (#10). **단일 정의점(#46).**

    ensure_conversation(질의 경로)과 routers/conversations.py의 조회 전부(목록·검색·
    피드백/취소의 Message JOIN)가 이 함수를 공유한다 — 이전엔 같은 규칙이 두 곳에
    중복 구현돼 docstring끼리 "함께 고칠 것"을 약속하고 있었다.
    user_id=None(X-User-Id 미전송)의 DEFAULT_USER 폴백도 여기서 흡수한다.
    기존 created_by NULL 대화(개발 데이터)는 어느 사용자와도 불일치 → 미노출.

    SQL 술어 전용이다 — SQLAlchemy 표현식이라 로드된 객체에 파이썬으로 평가할 수 없다.
    비-DB 경로에서 소유권을 물어야 하는 날이 오면 그건 별도 함수다.
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
    """대화가 없으면 새로 만들고, 있으면 소유 대화인지 검증한다.
      conversation_id=None이면 새 Conversation을 INSERT (created_by 저장, #10).
      conversation_id가 있으면 owned_filter로 소유 검증 —
      남의 대화 접근·삭제된 대화로의 질의 계속을 차단한다.
    """
    if conversation_id is None:
        # 저장값의 폴백은 필터와 별개 책임이라 여기서 직접 계산한다 — owned_filter의
        # 폴백과 표현이 겹쳐 보여도 하나는 "무엇을 저장하나", 하나는 "무엇이 보이나"다.
        conversation = Conversation(tenant_id=tenant_id, created_by=user_id or DEFAULT_USER)
        session.add(conversation)
        await session.flush()
        return conversation

    conversation = (await session.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(owned_filter(tenant_id, user_id))
    )).scalar_one_or_none()
    if conversation is None:
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
        # 아래 격리 필터로 빠지는 만큼 여유 조회 — limit이 '토큰 예산을 항상 포화시키는 상한'이라는
        # 전제를 유지하려면 제외분을 메워야 한다. 2배면 최근 절반이 차단·실패여도 limit행을 확보한다 (#22)
        .limit(limit * 2)
    )
    result = await session.execute(stmt)
    messages = list(result.scalars().all())[::-1]

    # 비정상 턴 격리 (#22) — 프롬프트 재료에서만 제외한다 (대화 조회 API는 그대로 노출).
    # blocked: 가드가 막은 입력이 다음 턴 프롬프트(condense·prior_turns·OTHER 이력)로
    #          재진입하면 차단 결정이 한 턴짜리로 휘발된다. 가드는 현재 질문만 검사한다.
    # failed/generating: content=''이라 빈 답변 턴이 맥락에 낀다.
    # user·assistant 짝을 함께 뺀다 — 한쪽만 빼면 build_prior_turns의 짝짓기가 밀린다.
    dropped_questions = {m.question_message_id for m in messages
                         if m.role == 'assistant' and m.status != 'done'}
    kept = [m for m in messages
            if not (m.id in dropped_questions or (m.role == 'assistant' and m.status != 'done'))]
    return kept[-limit:]   # 여유 조회분을 되돌려 원래 창 크기 유지

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

    with otel.span('condense', 'LLM') as sp:
        try:
            llm_messages = build_chat_prompt(
                CONDENSE_SYSTEM_PROMPT,
                build_condense_user_message(query, history),
            )
            result = await llm.acomplete(llm_messages)
            standalone = (result or '').strip() or query
        except Exception:
            logger.exception('LLM error(condense_query)')
            standalone = query
        otel.set_attrs(sp, {otel.INPUT_VALUE: query, otel.OUTPUT_VALUE: standalone,
                            'kms.history_messages': len(history)})
        return standalone


async def condense_to_queries(
        llm: LlmClient,
        query: str,
        messages: list[Message],
) -> list[str]:
    """질의 재작성 의미 확장(#5) — 1콜로 검색용 멀티쿼리를 만든다.

    반환 [0]은 기존 condense 출력과 같은 역할(standalone — 저장·캐시 키·리랭크 기준),
    [1:]는 검색 전용 어휘 변형(최대 2). condense_query와 달리 히스토리가 없어도
    LLM을 호출한다 — 운영은 service가 멀티턴에서만 부르지만(단일턴 확장은 실측상 손실),
    eval(retrieval_v2 --expand)이 빈 히스토리로 A/B를 돌려야 해서 게이트는 호출부 책임.
    실패·빈 결과 시 [query] 폴백 = 재작성·변형 없이 원본 단독 검색 (기능 자동 off).
    """
    history = [
        {'role': m.role, 'content': m.content}
        for m in messages
    ]

    with otel.span('condense', 'LLM') as sp:
        try:
            llm_messages = build_chat_prompt(
                CONDENSE_MULTI_SYSTEM_PROMPT,
                build_condense_user_message(query, history),
            )
            result = await llm.acomplete(llm_messages)
            queries = _parse_multi_queries(result, query)
        except Exception:
            logger.exception('LLM error(condense_to_queries)')
            queries = [query]
        otel.set_attrs(sp, {otel.INPUT_VALUE: query, otel.OUTPUT_VALUE: queries[0],
                            'kms.expanded_queries': queries[1:],   # 멀티쿼리 변형 (#5) — DB에 저장 안 되는 유일한 기록처
                            'kms.history_messages': len(history)})
        return queries


def _parse_multi_queries(result: str | None, query: str) -> list[str]:
    """condense_to_queries 출력 파싱 — 계측 래핑으로 본체가 깊어져 분리."""
    if result is None:
        return [query]
    lines = [l.strip() for l in result.strip().splitlines() if l.strip()]
    # 머리말 라벨 방어 — 모델이 "검색용 독립 질문:" 같은 라벨 줄을 앞에 붙이는 경우가
    # 실측됨(#5 mt003 3회 중 2회). 첫 줄을 무조건 standalone으로 쓰는 계약이라, 라벨이
    # 검색 쿼리·생성 질문이 되어 거절 답변으로 이어진다. 콜론 종결 줄은 질문일 수 없어 제거.
    lines = [l for l in lines if not l.endswith((':', '：'))]
    if not lines:
        return [query]
    standalone = lines[0]
    # standalone·변형 간 중복 제거 — 같은 줄이 반복되면 RRF에서 같은 순위 리스트를
    # 중복 가산해 원본 상위 결과를 희석시킨다 (#3 A/B의 goodpeople_rl005 실측 부작용)
    variants: list[str] = []
    for line in lines[1:]:
        if line != standalone and line not in variants:
            variants.append(line)
    return [standalone, *variants[:2]]


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
        status: str = "done",
        block_reason: str | None = None,
) -> Message:
    """사용자 질문과 assistant 답변을 세션에 등록한다.

    user 메시지에는 standalone_query와 (이번 턴에 첨부했다면) 첨부 추출 텍스트를,
    assistant 메시지에는 sources JSON을 저장한다.
    commit은 호출자가 담당한다. 반환: assistant Message (id는 flush로 확정 —
    즉시 경로 meta의 assistant_message_id·피드백 PATCH 대상, #8).
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
        intent=intent,   # 라우팅 결과 — 답변률 분모(KNOWLEDGE) 집계용. DB 반영 완료로 원복 (#13)
        question_message_id=user_message.id,   # 짝을 데이터로 (미답변 목록이 휴리스틱 없이 JOIN)
        status=status,             # 입력 차단이면 'blocked' — SQL 집계·이력 격리의 유일한 식별자 (#22)
        block_reason=block_reason,
    )
    session.add(assistant_message)
    await session.flush()   # assistant id 확정 (#8)
    await _touch_conversation(session, tenant_id, conversation_id, first_query=user_query)
    return assistant_message


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
    실패면 status='failed', answer=''로 호출.

    block_reason은 다루지 않는다 — 차단 턴은 입력 가드 전용이고 그 경로는 save_exchange를
    쓴다(자리표시 없는 즉시 경로). 자리표시는 항상 block_reason=NULL로 생성되므로 그대로 둔다.
    """
    msg = await session.get(Message, assistant_message_id)
    if msg is None or msg.tenant_id != tenant_id:
        return
    if msg.status != 'generating':
        # 자리표시가 아닌 상태를 덮어쓰는 중. 대표 경우: 스테일 스윕이 먼저 failed로 바꿔놨는데
        # 태스크가 살아서 완주했다(좀비 플립). 덮어쓰는 건 맞다 — 완성된 답변을 버릴 수 없다.
        # 다만 조용히 넘기면 "실패로 봤던 턴이 왜 done이 됐나"를 나중에 추적할 수 없어 경고를 남긴다.
        logger.warning('finalize가 %s 상태를 %s로 덮어쓴다 (message_id=%s) — 스테일 스윕과 경쟁했을 수 있다',
                       msg.status, status, assistant_message_id)
    msg.content = answer
    msg.sources = sources
    msg.status = status
    msg.latency_ms = latency_ms
    msg.cited_docs = cited_docs
    msg.is_refusal = is_refusal
    msg.intent = intent   # 라우팅 결과 — 답변률 분모(KNOWLEDGE) 집계용. DB 반영 완료로 원복 (#13)


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

