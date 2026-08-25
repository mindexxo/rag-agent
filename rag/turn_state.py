"""턴 상태 기계 — 시작(add_pending_turn)·종료(finalize_turn)·회수(sweep_stale_generating)의
단일 정의점 (#85, rag/conversation.py에서 분리).

  상태 전이 (값 어휘의 단일 정의점은 rag/models.py의 TurnStatus):
      (user)      → done                                          (항상)
      (assistant) generating → {done, cancelled, failed, blocked}  (모두 종결)

  분리 근거: conversation.py의 네 관심사(접근제어·이력조립·condense·턴상태) 중 턴상태는
  "질의 흐름 완료 이후의 저장/복구"만 다뤄 결합이 약했고(그 파일 docstring이 #46부터 예고),
  #59(취소 재실행)·#72(실패 상태)가 정확히 이 관심사에 얹히며 예고한 트리거가 발생했다.
  이력조립(검색 품질과 함께 변한다)과 턴상태(저장 안정성과 함께 변한다)는 변경 이유가 다르다.

  세션 규약: 모든 함수가 session을 받고 **commit하지 않는다** — 호출자 소관
  (sweep_stale_cache와 같은 규칙). 이동 전과 동일.

  알려진 한계 — 좀비 플립: 스테일 스윕이 먼저 failed로 정리한 뒤 원래 태스크가 완주하면
  finalize_turn이 done으로 덮어쓴다(경고 로그만 — 완성된 답변을 버릴 수 없다는 판단).
  실환경 빈도는 낮고(임계 500초) 이번 분리에서 고치지 않는다 — 상세는 finalize_turn.
"""
import logging
from datetime import timedelta

from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Conversation, Message, TurnStatus

logger = logging.getLogger(__name__)

# ── TurnStatus의 파생 집합 — 교집합 없음, 뭉치면 안 된다 ─────────────────────────
# 두 집합은 소비처가 완전히 다르다(이력 조립 vs RETRY 판정). 하나의 "비정상" 집합으로
# 합치면 이 두 축이 독립이라는 사실이 사라진다 — blocked는 이력에서 빼지만 RETRY 대상이
# 아니고, cancelled·failed는 RETRY 대상이지만 이력에는 남는다(질문 자체는 정상이라
# 빼면 화면과 모델 이력이 어긋난다).

# 프롬프트 이력에서 격리 — 소비: conversation.load_recent_messages.
# blocked: 차단 결정이 다음 턴 프롬프트로 재진입해 재해석되면 안 된다 (조회 API엔 그대로 노출).
# generating: 이번 턴 자신의 자리표시 — 자기 자신으로 되먹임되면 안 된다.
HISTORY_ISOLATED = (TurnStatus.BLOCKED, TurnStatus.GENERATING)

# SSE done 이벤트의 finish_reason 어휘 = 전체 − {generating} — 소비: streaming.TurnResult.
# 계산형으로 둔다: 상태가 추가되면 자동으로 따라와 나열이 다시 흩어지지 않는다.
TERMINAL = tuple(s for s in TurnStatus if s is not TurnStatus.GENERATING)



# 답을 못 받은 턴 — RETRY의 대상 (#59·#72). 사용자에겐 '멈췄든 오류가 났든' 같은 사건이다.
# 상태를 추가하면 eval/_gold_history.py의 계약 docstring도 확인할 것 (#77)
UNANSWERED = (TurnStatus.CANCELLED, TurnStatus.FAILED)


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
    status = TurnStatus(status)   # 5종 밖 값이 조용히 저장되던 것을 즉시 ValueError로 (#85)
    msg = await session.get(Message, assistant_message_id)
    if msg is None or msg.tenant_id != tenant_id:
        return
    if msg.status != TurnStatus.GENERATING:
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
        .where(Message.status == TurnStatus.GENERATING)
        .where(Message.created_at < func.now() - timedelta(seconds=GENERATION_STALE_SECONDS))
        .values(status=TurnStatus.FAILED.value)
    )
    return result.rowcount
