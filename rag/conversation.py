"""멀티턴 대화 관리 — 소유권·이력 조립·condense 질의 재작성·턴 상태.
대화 테이블(conversations/messages)을 이용해 이전 턴을 불러오고,
후속 질문('그럼 5일은?') 을 검색 가능한 독립 질문으로 변환한다.

  질의 흐름:
      ensure_conversation(owned_filter로 소유 검증)
      -> add_pending_turn(**턴 시작 시점** — user + assistant 자리표시를 커밋, #72)
      -> load_recent_messages -> condense_query/condense_to_queries
      -> retrieve는 standalone_query로 수행
      -> finalize_turn (모든 경로 공용 종료 지점 — 차단·캐시히트·근거없음·생성 전부)

  저장을 라우팅보다 앞에 두는 이유: 그 뒤 어디서 실패해도 사용자의 질문은 남아야 한다.
  예전엔 즉시 경로가 save_exchange로 원샷 저장해 prepare() 실패 시 질문이 통째로 유실됐다.

  질의 흐름 밖: sweep_stale_generating — 고착 턴 회수, 워커 cron 전용(#46).

  주: 접근 제어·이력 조립·condense·턴 상태 네 관심사가 한 파일에 있다. 커지면
  턴 상태 기계(save/pending/finalize/sweep)를 먼저 떼는 게 경계가 명확하다 —
  질의 흐름 완료 이후의 저장/복구만 다뤄 결합이 약하다. (8번 축 재방문 시 후보)
"""
import logging
from datetime import timedelta

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag import otel
from rag.llm import LlmClient
from rag.llm_schemas import CondenseMultiResult, CondenseResult, acomplete_validated
from rag.models import Conversation, Message

logger = logging.getLogger(__name__)
from rag.tokens import estimate_tokens
from rag.prompt_texts import (CANCELLED_TURN_EMPTY, CANCELLED_TURN_SUFFIX, FAILED_TURN_EMPTY,
                              CONDENSE_MULTI_SYSTEM_PROMPT, CONDENSE_SYSTEM_PROMPT)
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
    # generating: 이번 턴의 자리표시다(#72). 턴 시작에 커밋되므로 빼지 않으면 방금 넣은
    #          질문이 자기 자신의 '이전 대화'로 되먹임된다.
    # cancelled/failed는 격리하지 않는다(#59·#72) — 질문은 정상이고, 빼면 화면(조회 API는
    #          노출)과 모델이 보는 이력이 어긋나 "다시"의 지시대상이 밀린다. 실제로 실패 턴
    #          뒤의 "다시"가 그 전 턴 답을 반복하는 버그가 이것 때문이었다.
    #          빈 답변·잘린 답변의 표시는 _history_content가 담당(조립 시점 표식).
    # user·assistant 짝을 함께 뺀다 — 한쪽만 빼면 build_prior_turns의 짝짓기가 밀린다.
    _ISOLATED = ('blocked', 'generating')
    dropped_questions = {m.question_message_id for m in messages
                         if m.role == 'assistant' and m.status in _ISOLATED}
    kept = [m for m in messages
            if not (m.id in dropped_questions
                    or (m.role == 'assistant' and m.status in _ISOLATED))]
    return kept[-limit:]   # 여유 조회분을 되돌려 원래 창 크기 유지


UNANSWERED = ('cancelled', 'failed')   # 답을 못 받은 턴 — RETRY의 대상 (#59·#72)


def last_unanswered_turn(messages: list[Message]) -> tuple[Message, Message] | None:
    """직전 턴이 답을 못 받은 턴이면 (실질 질문 user, 그 assistant)를 돌려준다 (#59·#72).

    RETRY 디스패치의 상태 근거 — "무엇을 다시 할지"는 분류기(표면 패턴)가 아니라
    이 사실(직전 턴이 답을 못 냈는가)이 결정한다.

    **취소와 실패를 함께 본다**(#72). 사용자에겐 "내가 멈췄든 오류가 났든 답을 못 받았다"가
    같은 사건이다. 예전엔 cancelled만 봐서, 실패 턴 뒤의 "다시"가 그 **전전** 턴 답을
    반복하는 버그가 있었다(실측: adererror 대화 16820). 폴백 제거(#72)로 failed가 잦아져
    노출 빈도도 올라갔다. blocked는 제외 — 차단 결정을 "다시"로 뒤집게 하면 안 된다
    (애초에 load_recent_messages가 이력에서 격리한다).
    페어링 불변식(user 바로 다음 assistant)이 깨진 행이면 크래시 대신 None.

    체인 되감기("다시" 연타): 재실행 턴이 다시 취소되면 그 user 행엔 원 발화("다시")만
    남고 실질 질문은 물려받은 standalone_query로만 이어진다 — content를 그대로 쓰면
    두 번째 "다시"부터 질문 슬롯이 "다시"로 퇴화한다(리뷰 발견, 확신 88). 같은
    standalone으로 연결된 연속 취소 짝을 거슬러 올라가 체인 머리(실질 질문)의 user를
    돌려준다. 서로 다른 질문이 연속 취소된 경우엔 standalone이 달라 자동으로 멈춘다.
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
        if not (prev_a.role == 'assistant' and prev_a.status in UNANSWERED
                and prev_u.role == 'user' and prev_u.id == prev_a.question_message_id
                and user.standalone_query is not None
                and prev_u.standalone_query == user.standalone_query):
            break
        user = prev_u
        i -= 2
    return user, last


def _history_content(message: Message) -> str:
    """프롬프트 이력용 답변 텍스트 — 답 못 받은 턴(취소·실패) 표식의 단일 정의점 (#59·#72).

    DB 저장본(Message.content)은 절대 바꾸지 않는다 — 표식은 조립 시점에만 붙는다.
    소비처는 build_prior_turns(생성·OTHER 맥락)와 _condense_call(재작성 이력) 둘 —
    양쪽이 각자 붙이면 모델이 보는 두 이력이 어긋난다(교차 기능 갭의 전형).
    """
    if message.role == 'assistant' and message.status == 'cancelled':
        return message.content + CANCELLED_TURN_SUFFIX if message.content else CANCELLED_TURN_EMPTY
    if message.role == 'assistant' and message.status == 'failed':
        return FAILED_TURN_EMPTY   # 실패는 항상 content='' (finalize가 빈 답변으로 마감)
    return message.content

async def _condense_call(
        llm: LlmClient,
        query: str,
        messages: list[Message],
        multi: bool,
) -> list[str]:
    """condense_query·condense_to_queries 공용 골격(#46) — 두 함수의 34줄이 6단계
    (history 조립 → span → 프롬프트 조립 → LLM 호출 → 파싱/폴백 → 계측) 전부 동일했다.

    갈리는 건 프롬프트 상수·파싱·계측 속성뿐이라 multi 하나로 분기한다 — 호출부가
    영원히 2개뿐이고(#15 DSPy 실험 종결로 condense는 손튜닝 확정) 차이가 파싱 1줄 +
    otel 키 1개라, callable 파서 주입은 두 고정 케이스를 위한 과한 전략 패턴이다.

    반환은 항상 list[str]: [0]=standalone, [1:]=검색 변형(멀티만 — 단일은 항상 []).
    span 이름은 둘 다 'condense' 그대로(Phoenix 대시보드 연속성). multi=True일 때만
    kms.expanded_queries를 남긴다 — 단일 경로 span엔 원래 이 속성이 없었다는 사실을 보존.
    """
    history = [
        {'role': m.role, 'content': _history_content(m)}
        for m in messages
    ]

    with otel.span('condense', 'LLM') as sp:
        # 폴백 없음 (#72). 재작성을 판단하지 못한 상태에서 "원문 그대로 검색"으로 넘어가면
        # 근거 없는 추측으로 턴을 진행하는 것이고, 애초에 LLM이 죽었다면 생성도 실패한다 —
        # 폴백은 턴을 살리는 게 아니라 에러를 검색까지 다 수행한 뒤로 미룰 뿐이었다.
        # 예외는 그대로 전파해 호출자가 실패로 끝낸다. 질문 자체는 이미 저장돼 있다(#72 자리표시).
        llm_messages = build_chat_prompt(
            CONDENSE_MULTI_SYSTEM_PROMPT if multi else CONDENSE_SYSTEM_PROMPT,
            build_condense_user_message(query, history),
        )
        parsed = await acomplete_validated(
            llm, llm_messages, CondenseMultiResult if multi else CondenseResult, span=sp)
        queries = _postprocess_queries(parsed, query)
        attrs = {otel.INPUT_VALUE: query, otel.OUTPUT_VALUE: queries[0],
                 'kms.history_messages': len(history)}
        if multi:
            attrs['kms.expanded_queries'] = queries[1:]   # 멀티쿼리 변형 (#5) — DB에 저장 안 되는 유일한 기록처
        otel.set_attrs(sp, attrs)
        return queries


async def condense_query(
        llm: LlmClient,
        query: str,
        messages: list[Message]
) -> str:
    """후속 질문을 독립 질문으로 변환한다.

    히스토리가 없으면 LLM을 호출하지 않고 원본 query를 그대로 반환한다 —
    이 게이트는 이 함수만의 계약이라 공용 골격(_condense_call) 밖에 남긴다.
    LLM 결과가 비어 있거나 호출에 실패하면 원본 query로 폴백한다.
    """
    if not messages:
        return query
    queries = await _condense_call(llm, query, messages, multi=False)
    return queries[0]


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
    return await _condense_call(llm, query, messages, multi=True)


def _postprocess_queries(parsed: CondenseResult | CondenseMultiResult, query: str) -> list[str]:
    """스키마 검증을 통과한 결과의 후처리 (#43) — 스키마가 강제 못 하는 것만 남았다.

    구 _parse_multi_queries의 줄 분리·라벨 줄 제거("검색용 독립 질문:" 실측 2/3)는
    JSON 필드 이름이 구조를 확정하며 소멸. 남는 건 필드 간 관계다:
    - 빈 standalone → 원본 폴백 (minLength가 전 서버에서 강제된다는 보장 없음)
    - standalone·변형 간/변형끼리 중복 제거 — 같은 줄이 반복되면 RRF에서 같은 순위
      리스트를 중복 가산해 원본 상위 결과를 희석시킨다 (#3 goodpeople_rl005 실측)
    - variants[:2] 방어 슬라이스 (maxItems를 서버가 무시할 경우 대비)
    """
    standalone = parsed.standalone.strip() or query
    raw_variants = getattr(parsed, 'variants', [])   # CondenseResult(단일)엔 variants가 없다
    variants: list[str] = []
    for line in (v.strip() for v in raw_variants):
        if line and line != standalone and line not in variants:
            variants.append(line)
    return [standalone, *variants[:2]]


def _new_user_message(tenant_id: str, conversation_id: int, user_query: str,
                      attachments: list[dict] | None, user_id: str | None) -> Message:
    """user Message 생성.

    standalone_query는 받지 않는다 — 이 행은 턴 **시작** 시점(라우팅·condense 전)에 만들어지므로
    그 값을 아직 모른다. finalize_turn이 종료 시점에 백필한다 (#72). intent 컬럼이 이미 같은 리듬이다.
    """
    return Message(
        tenant_id=tenant_id,
        conversation_id=conversation_id,
        role="user",
        content=user_query,
        attachments=attachments,
        user_id=user_id,
    )


# save_exchange는 #72에서 삭제됐다. 즉시 경로(차단·캐시히트·근거없음)가 user+assistant를 한 번에
# INSERT하던 "원샷 저장"이었는데, 그 형태는 라우팅이 끝난 뒤에만 가능했다 — 그래서 prepare()가
# 실패하면 질문이 통째로 유실됐다. 이제 모든 경로가 begin_turn(턴 시작) → finalize_turn(종료)
# 한 수명주기를 따르고, 즉시 경로는 자리표시를 곧바로 finalize한다.


async def _touch_conversation(session: AsyncSession, tenant_id: str, conversation_id: int,
                               first_query: str | None = None) -> None:
    """대화의 last_used_at 갱신 + 제목이 비어 있으면 첫 질문으로 세팅 (목록 표시용)."""
    await session.execute(
        update(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .values(last_used_at=func.now())
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
        attachments: list[dict] | None = None,
        user_id: str | None = None,
) -> Message:
    """user 메시지 + 생성 대기 상태의 assistant 자리표시를 세션에 등록한다.

    **턴 시작 시점**에 호출한다 — 라우팅·condense·검색보다 앞이다 (#72). 그 뒤 어디서 실패해도
    질문이 남게 하는 것이 목적이라, 이보다 늦으면 의미가 없다. 모든 경로(차단·other·knowledge)가
    이 함수 하나로 턴을 연다.
    assistant는 content='', status='generating'으로 시작하고, 완료 시 finalize_turn이 채운다.
    commit은 호출자가 담당한다. 반환: assistant 자리표시 Message (id는 flush 후 확정).

    user_query는 **사용자가 실제 친 원문**이다 — RETRY 재실행(#59)의 질의 치환은 이 시점보다
    뒤에 일어나므로, 여기 저장되는 값이 곧 화면·기록에 남아야 할 발화다.
    """
    user_message = _new_user_message(tenant_id, conversation_id, user_query, attachments, user_id)
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
        intent: str | None = None,
        standalone_query: str | None = None,
        cache_kind: str | None = None,
        block_reason: str | None = None,
) -> None:
    """자리표시를 최종 결과로 채운다 (id로 재조회 후 UPDATE) — **모든 경로의 유일한 종료 지점**.

    백그라운드 태스크가 '자기 세션'으로 호출하므로 객체 참조가 아닌 id로 재조회한다.
    tenant 소유를 확인(격리)하고, 없거나 남의 것이면 무시. commit은 호출자가 담당한다.
    실패면 status='failed', answer=''로 호출.

    #72로 즉시 경로(차단·캐시히트·근거없음)도 이 함수를 쓴다 — 그래서 예전엔 save_exchange만
    다루던 block_reason·cache_kind가 여기로 합류했다. 해당 없는 경로에선 항상 None이라 무해하다.

    standalone_query는 **짝 user 행**에 백필한다. 그 값은 턴 시작(자리표시 INSERT) 시점엔 아직
    없고 condense 이후에야 확정되기 때문이다. 백필 시점이 종료인 이유: load_recent_messages의
    격리 필터가 non-done 턴을 이력에서 빼므로, 이 턴이 남의 이력에 보이기 시작하는 순간이
    곧 finalize 시점이다 — 그보다 먼저 채워둘 이유가 없다.
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
    msg.intent = intent   # 라우팅 결과 — 답변률 분모(KNOWLEDGE) 집계용. DB 반영 완료로 원복 (#13)
    msg.cache_kind = cache_kind      # 'semantic'=캐시 재생 답변 (기간별 히트율 집계용)
    msg.block_reason = block_reason  # 입력 가드 차단 사유 — 차단 턴만 값이 있다 (#22)

    # 짝 user 행에 standalone_query 백필. tenant WHERE는 격리 원칙상 필수 —
    # question_message_id만으로도 유일하지만, 쓰기 경로는 예외 없이 테넌트를 명시한다.
    if msg.question_message_id is not None:
        await session.execute(
            update(Message)
            .where(Message.id == msg.question_message_id)
            .where(Message.tenant_id == tenant_id)
            .values(standalone_query=standalone_query)
        )


# generating이 이 시간 넘게 지속되면 백그라운드 태스크/웹 프로세스 사망으로 고착 → failed 간주.
#
# 임계는 '정상 생성의 최대 소요'보다 **확실히 커야** 한다. 살아 있는 요청을 먼저 failed로
# 선고하면 그 태스크가 완주할 때 finalize_turn이 그 행을 done으로 덮어써(좀비 플립), 이미
# 실패로 보고된 턴이 조용히 성공이 되고 과거 통계까지 소급 변한다. LLM 호출 타임아웃이
# 300초(rag/llm.py)라 이전 값 300은 그 경계에 딱 붙어 있었다 — 동시성 포화로 vLLM 큐에서
# 밀리면 정상 요청이 스윕에 걸릴 수 있었다. 스윕은 '죽은 프로세스 회수'가 목적이라 늦게
# 정리해도 손해가 없으므로 여유를 뒀다.
GENERATION_STALE_SECONDS = 500


async def sweep_stale_generating(session: AsyncSession) -> int:
    """고착된 generating 턴을 failed로 정리한다. 처리한 행 수 반환. **부수효과: UPDATE**
    (commit은 호출자 담당 — sweep_stale_cache와 같은 규칙).

    생성 태스크는 arq가 아니라 웹 앱의 asyncio 태스크다 — 웹 프로세스가 죽으면 코루틴이
    증발해 자리표시(status='generating')만 DB에 남고, _finalize_out_of_band의 DB 기록
    실패도 같은 상태를 남긴다. 그 회수를 arq cron(rag/worker.py, 1분 주기)이 맡는다(#46·#72).

    이전엔 GET /conversations/{id}/messages가 그 대화에 한해 조회 시점에 정리했다(lazy).
    cron으로 옮긴 이유: lazy는 **열어본 대화만** 치유해서, 재방문 없는 대화의 고착 행이
    영원히 generating으로 남아 운영 리포트 집계에 유령 상태로 끼고 cancel이 "진행 중"으로
    오판했다. 대가는 회복 지연 — 최대 500초(조회 즉시)에서 최대 ~560초(500초+주기 1분)로.

    전 테넌트 일괄이다(sweep_stale_cache와 같은 위생 성격) — 상태만 바꾸고 아무것도
    반환·노출하지 않으므로 격리 위반이 아니다. 시간 비교는 서버측(func.now()) —
    created_at 컬럼의 naive/aware 혼선을 피한다.

    500초 넘게 살아 있는 생성(동시성 포화로 vLLM 큐 대기 등)도 걸린다 — lazy 시절엔
    "마침 누가 조회해야" 걸렸지만 cron은 결정론적으로 걸리므로 finalize_turn의 좀비 플립
    경고가 더 자주 보일 수 있다. 방어(무조건 덮어쓰기+경고, cancel의 레지스트리 우선)는
    그대로 유효하다.
    """
    result = await session.execute(
        update(Message)
        .where(Message.status == 'generating')
        .where(Message.created_at < func.now() - timedelta(seconds=GENERATION_STALE_SECONDS))
        .values(status='failed')
    )
    return result.rowcount


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
            turns.append({"q": pending_question, "a": _history_content(message)})
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

