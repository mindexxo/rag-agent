"""RAG 서비스 — 한 턴의 수명을 조율한다.

    prepare()   턴 시작 저장(commit) → 라우팅·검색어·근거·캐시히트 확정
    generate()  확정 답변이 있으면 그대로, 없으면 LLM 스트리밍
    finalize()  턴 종료 — 자리표시를 최종 결과로 UPDATE (**모든 경로 공용**)

턴은 항상 "자리표시 생성 → finalize" 한 수명주기를 따른다 (#72). 예전엔 즉시 경로만
save_exchange로 원샷 INSERT하는 두 번째 저장 형태가 있었는데, 그건 라우팅이 끝난 뒤에만
가능해 prepare() 실패 시 사용자의 질문이 통째로 유실됐다.

즉시 경로와 생성 경로를 가르는 기준은 **답이 이미 확정됐느냐**다 —
그 판정은 PreparedRag.resolved_answer 한 곳에 있고 generate()·needs_generation이 함께 본다.
저장 형태가 아니라 "백그라운드 태스크가 필요하냐"만 가르는 축이다.

전역 싱글톤 금지 — 요청마다 tenant_id가 다르므로 생성자 인자로 받는다.
"""
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from config import settings
from rag.conversation import (build_prior_turns, condense_query, condense_to_queries,
                              ensure_conversation, load_recent_messages,
                              trim_messages_for_condense)
from rag.models import TurnStatus
from rag.turn_state import add_pending_turn, finalize_turn, last_unanswered_turn
from rag.guardrail import classify_and_guard
from rag.clients import shared_llm
from rag.citation_labels import sources_from_chunks
from rag.models import Conversation, Message
from rag.prompt_texts import BLOCKED_INPUT_ANSWER, NO_EVIDENCE_ANSWER, SMALLTALK_ANSWER
from rag.prompts import (build_chat_prompt, build_citation_constraint,
                         build_knowledge_generation_prompt, build_other_system_prompt,
                         build_other_user_message)
from rag.retriever import RetrievalResult, retrieve, RetrievedChunk

from schemas.kms import SourceCitation, QueryAttachment
from typing import Literal
from rag import cache

logger = logging.getLogger(__name__)


@dataclass
class PreparedRag:
    """답변 생성 전에 확정된 RAG 실행 컨텍스트.

    prepare 단계에서 대화, 검색 질의, 검색 결과, 출처 정보를 미리 계산해두고
    generate/save 단계가 같은 컨텍스트를 공유하도록 한다.
    """
    conversation_id: int
    original_query: str
    standalone_query: str
    prior_turns: list[dict]
    retrieval: RetrievalResult | None
    sources: list[SourceCitation]
    source_doc_ids: list[int]
    cached_answer: str | None = None
    cache_kind: Literal["semantic"] | None = None
    # 라우팅 결과. knowledge=RAG, other=대화/메타/역할밖 제약 생성, blocked=입력 차단
    route: Literal["knowledge", "other", "blocked"] = "knowledge"
    attachments: list[dict] = field(default_factory=list)      # 컨텍스트 주입용 — 히스토리 저장분 + 이번 턴 신규 합본
    # 이번 턴 신규 첨부(new_attachments) 필드는 #72에서 삭제됐다 — 저장이 턴 시작으로 옮겨가
    # _open_turn이 new_attachment_dicts를 직접 user 메시지에 넣는다. PreparedRag까지 들고 갈 이유가 없다.
    domain_hint: str | None = None   # [임시] 테넌트 지식 범위 설명 — 생성 프롬프트 주입용, 저장 안 함 (#1)
    assistant_message_id: int | None = None   # 생성 경로: 자리표시 assistant 메시지 id (백그라운드 태스크가 UPDATE할 대상)
    block_reason: str | None = None   # route='blocked'일 때 가드 판정 사유 — 저장·집계용 (#22)
    faq_versions: dict[int, datetime] | None = None   # 근거 FAQ {id: updated_at} 스냅샷 — cache.save_answer 낙관적 검증 기준 (#16)
    # 검색이 만든 원본(standalone) 쿼리 dense 벡터 — maybe_cache가 save_answer로 넘겨
    # TEI 재임베딩을 없앤다 (#50, 사유는 rag/cache.py:get_semantic docstring).
    # 캐시 히트 분기는 채우지 않는다: should_cache=False라 maybe_cache가 아예 안 불려
    # 죽은 값이 된다. 이 필드는 "저장에 쓸 벡터"라는 의미만 갖는다.
    # repr=False — 1024차원 float가 로그·트레이스백에 새는 것 차단.
    query_embedding: list[float] | None = field(default=None, repr=False)
    # RETRY 재실행 턴(#59)에서만 채워진다 — 사용자가 실제 입력한 발화("다시").
    # 검색·프롬프트는 original_query(=직전 실질 질문)를 쓴다.
    # 예전엔 recorded_query 프로퍼티가 이 값으로 "저장할 질문"을 늦게 계산했는데, #72에서
    # 저장이 턴 시작으로 옮겨가며 불필요해졌다 — _open_turn이 사용자가 친 원문을 그대로 넣고
    # RETRY의 질의 치환은 그보다 뒤에 일어난다. 필드는 라우팅 판정 신호로 남는다.
    display_query: str | None = None
    # ATTACHMENT 검색 스킵 턴(#63) — intent_label이 'ATTACHMENT'로 저장하는 근거.
    attachment_grounded: bool = False

    def __post_init__(self) -> None:
        """검색 여부와 route를 짝지어 생성 시점에 강제한다 (#36).

        knowledge는 캐시 히트까지 포함해 항상 검색을 거치므로 retrieval이 있어야 하고,
        blocked/other는 검색을 하지 않으므로 없어야 한다. 이전엔 이 불변식을 _routed() 헬퍼의
        관례로만 지켰고 dataclass는 어긋난 조합을 막지 않았다 — 그런데 should_cache가
        `retrieval is not None`으로 blocked/other를 걸러내므로, 관례가 깨지면 차단된 턴이
        캐시에 들어가는 식으로 **조용히** 틀린다. 생성자가 거부하게 만들어 그 여지를 없앤다.
        """
        if (self.route == "knowledge") != (self.retrieval is not None):
            raise ValueError(
                f"PreparedRag 불변식 위반 — route={self.route!r}인데 retrieval이 "
                f"{'있다' if self.retrieval is not None else '없다'}. "
                "knowledge는 검색을 거치고(캐시 히트 포함), blocked/other는 거치지 않는다."
            )
        if self.assistant_message_id is None:
            # 같은 이유로 여기서 막는다 (#72). 이 값이 비면 finalize_turn의
            # session.get(Message, None)이 조용히 None을 받아 **아무것도 UPDATE하지 않고**
            # 정상 반환한다 — 턴이 영영 generating으로 남고 에러도 안 난다.
            # 턴 시작 저장(_open_turn)이 항상 선행하므로 정상 경로에선 비는 일이 없다.
            raise ValueError(
                "PreparedRag 불변식 위반 — assistant_message_id가 없다. "
                "모든 턴은 _open_turn이 만든 자리표시 id를 들고 다녀야 finalize가 동작한다."
            )

    @property
    def resolved_answer(self) -> str | None:
        """prepare() 시점에 이미 확정된 답변 문구. None이면 LLM 생성이 필요하다 (#36).

        즉시 경로와 생성 경로를 가르는 기준이 곧 이것이다 — rag/streaming.py가 말하는
        "답이 이미 있느냐". needs_generation과 generate()가 **이 한 곳만** 본다:
        이전엔 같은 5갈래 판정이 두 곳에 물리적으로 중복돼 있었고, 주석으로 동기화를
        약속하고 있었다(한쪽만 고치면 조용히 어긋나는 구조).

        OTHER는 검색 없이도 LLM이 대화성 응답을 만들어야 하므로 확정 답변이 없다 —
        cached_answer보다 먼저 걸러내야 기존 분기 순서와 일치한다.
        """
        if self.route == "blocked":
            return BLOCKED_INPUT_ANSWER
        if self.route == "other":
            return None
        if self.cached_answer is not None:
            return self.cached_answer
        if self.no_evidence:
            return NO_EVIDENCE_ANSWER
        return None

    @property
    def terminal_status(self) -> TurnStatus:
        """즉시 경로 저장 status — 입력 차단 턴만 BLOCKED, 나머지는 DONE (단일 정의점 #54).
        생성 경로의 status(done/cancelled/failed)는 실행 결과값이라 여기서 정할 수 없다."""
        return TurnStatus.BLOCKED if self.route == "blocked" else TurnStatus.DONE

    @property
    def intent_label(self) -> str | None:
        """저장용 인텐트 라벨 — 답변률 분모(KNOWLEDGE) 판별에 쓴다.
        blocked는 인텐트 판정 자체가 무의미(unsafe 입력)라 NULL.

        ATTACHMENT(#63)는 route=knowledge지만 별도 라벨로 저장한다 — 첨부만 근거인
        턴이 KNOWLEDGE로 남으면 stats의 KB 커버리지 지표(근거미확인율·지식 갭)가
        오염되고, 이월 첨부 케이스는 사후 필터도 불가능하다(리뷰 발견). 전이 인텐트
        원칙의 의도적 예외: 라우팅에선 소멸하지만 "근거 유형이 다르다"는 사실은 남긴다."""
        if self.attachment_grounded:
            return "ATTACHMENT"
        return {"knowledge": "KNOWLEDGE", "other": "OTHER"}.get(self.route)

    @property
    def citation_candidates(self) -> tuple[list[SourceCitation], list[str]]:
        """인용 후보 (검색 출처, 첨부 파일명) — 꼬리 제약과 파서가 공유하는 단일 파생점 (#65).

        이전엔 rag/service.py(제약 생성)와 rag/streaming.py(파서 호출)가 같은 컴프리헨션을
        각자 써서 "우연히 같은" 상태였다. 제약이 무효였을 때는 어긋나도 무해했지만(각주 누락),
        #65에서 강제가 실제로 걸리면 어긋남이 **에러 없는 오귀속**이 된다 — 축소된 범위 안의
        다른 문서가 강제로 인용되고 파서의 범위 검증은 통과한다. 파생을 한 곳으로 모아
        한쪽만 고치는 미래의 수정이 구조적으로 불가능하게 만든다.
        순서 계약(출처 다음 첨부)은 rag/citation_labels.py가 정의점이다.
        """
        return self.sources, [a['filename'] for a in (self.attachments or [])]

    @property
    def no_evidence(self) -> bool:
        """검색 근거 없음 판정 — 첨부가 있으면 첨부 기반 답변이 가능하므로 False (단일 정의점)."""
        return self.retrieval is not None and self.retrieval.no_evidence and not self.attachments

    @property
    def is_cache_hit(self) -> bool:
        return self.cached_answer is not None

    @property
    def needs_generation(self) -> bool:
        """LLM 스트리밍이 필요한 경로인가 (= 백그라운드 태스크 대상).
        resolved_answer가 없으면 생성해야 한다 — generate()와 물리적으로 같은 값을 본다."""
        return self.resolved_answer is None

    @property
    def should_cache(self) -> bool:
        """새로 LLM이 만든, 근거 있는 응답만 캐시 대상이다.
        첨부가 있는 대화의 답변은 특정 고객 문서에 종속되므로 공용 캐시에 넣지 않는다."""
        return (
                self.cached_answer is None
                and self.retrieval is not None
                and not self.retrieval.no_evidence
                and not self.attachments
        )

class RagService:
    def __init__(self, tenant_id: str, session: AsyncSession, user_id: str | None = None):
        # 요청마다 인스턴스 생성 -> tenant_id 격리 보장
        self.tenant_id = tenant_id
        self.session = session
        self.user_id = user_id          # 지표 씨앗 — user 메시지에 기록 (X-User-Id)
        self._llm = shared_llm          # 공용 싱글톤 재사용 (요청마다 HTTP 풀 생성 방지 — P1-12)
        # vLLM 스트림 종료 사유(stop|length|...) — generate() 완주 후 streaming.py가 읽어
        # 지표·스팬에 기록한다 (#129). TurnStatus finish_reason과 이름이 같지만 다른 축이다
        # (AGENTS.md '헷갈리는 것'). 인스턴스가 요청 단위라 동시 요청 간 안 섞인다.
        self.last_vllm_finish_reason: str | None = None


    async def prepare(
            self,
            query: str,
            conversation_id: int | None = None,
            attachments: list[QueryAttachment] | None = None,
            domain_hint: str | None = None,
    ) -> PreparedRag:
        """한 턴을 처리할 컨텍스트를 확정한다 — 라우팅·검색어·근거·캐시히트가 여기서 결정된다.

        **쓰기부터 시작한다** (#72) — 대화(신규면)와 이번 턴(user + assistant 자리표시)을 먼저
        INSERT·commit하고 라우팅에 들어간다. 되돌릴 수 없다. 재호출하면 턴이 중복 생성되므로
        한 요청에 한 번만 부를 것. 캐시 히트면 hit_count도 오른다.
        LLM도 2회 부른다(인텐트 분류·질의 재작성). 이름이 가벼워 보이지만 요청 지연의
        상당 부분이 여기서 나온다 — 캐시 히트 턴에서는 사실상 전부다.

        **왜 쓰기가 먼저인가**: 이 함수 안에서 실패하면(분류·재작성 판단 실패, 검색·캐시 오류,
        연결 끊김) 예전엔 사용자의 질문이 통째로 사라졌다 — 신규 대화는 빈 껍데기만 남고,
        기존 대화는 흔적조차 없었다. 저장을 라우팅보다 앞에 두면 어디서 죽든 질문은 남는다.
        자리표시는 status='generating'이라 load_recent_messages의 격리 필터(#22)가 자기 자신을
        이력에서 자동으로 빼준다 — 방금 넣은 질문이 '이전 대화'로 되먹임되지 않는다.

        성격이 다른 넷을 단계로 나눠 뒀다 (#36):
          _resolve_conversation  대화 확보 + 첨부 합본
          _open_turn             턴 시작 저장 (쓰기·commit)
          (본문)                 라우팅 판정 → blocked/other/첨부전용(ATTACHMENT)이면 여기서 끝
          _prepare_knowledge     질의 재작성 → 검색 → 출처·FAQ스냅샷 → 캐시 조회(쓰기)
        """
        conversation, attachment_dicts, new_attachment_dicts = await self._resolve_conversation(
            conversation_id, attachments)
        assistant_message_id = await self._open_turn(conversation.id, query, new_attachment_dicts)

        # RETRY(#59)가 질의를 치환하면 실패 마감이 그 '치환된' 질문을 남겨야 한다 — 사유는
        # _fail_turn docstring. 요청당 인스턴스라(클래스 docstring) 이 자리에 둬도 안전하다.
        self._resolved_query = query
        try:
            return await self._route(
                conversation, assistant_message_id, query,
                attachment_dicts, new_attachment_dicts, domain_hint)
        except Exception:
            # 자리표시를 즉시 failed로 닫는다. 없으면 sweep_stale_generating(임계 500초 +
            # cron 1분)이 회수할 때까지 최대 ~560초간 generating으로 남는다 — 폴백을 걷어낸
            # 뒤로 이 경로가 잦아졌으므로 지연을 방치하지 않는다. 원 예외는 그대로 올린다.
            await self._fail_turn(assistant_message_id, self._resolved_query)
            raise

    async def _route(
            self,
            conversation: Conversation,
            assistant_message_id: int,
            query: str,
            attachment_dicts: list[dict],
            new_attachment_dicts: list[dict],
            domain_hint: str | None,
    ) -> PreparedRag:
        """라우팅 판정 본문 — prepare()가 실패 안전망으로 감싸는 구간이다 (#72)."""
        messages = await load_recent_messages(self.session, self.tenant_id, conversation.id)
        display_query: str | None = None   # RETRY 재실행에서만 채워짐 — _routed가 늦은 바인딩으로 읽는다

        def _routed(route: str, block_reason: str | None = None) -> PreparedRag:
            # 검색·인용 없이 라우팅 결과만 담는 PreparedRag (blocked/other 공용).
            # retrieval=None이 route와 짝을 이룬다 — __post_init__이 그 불변식을 강제한다.
            return PreparedRag(
                block_reason=block_reason,
                conversation_id=conversation.id,
                assistant_message_id=assistant_message_id,
                original_query=query,
                standalone_query=query,
                prior_turns=build_prior_turns(messages, settings.history_budget_tokens),
                retrieval=None,
                sources=[],
                source_doc_ids=[],
                route=route,
                attachments=attachment_dicts,
                domain_hint=domain_hint,
                display_query=display_query,
            )

        # 입력 가드레일 + 인텐트 분류 (통합 1회 호출) — 히스토리 유무와 무관하게 항상 실행
        decision = await classify_and_guard(self._llm, query, has_attachments=bool(attachment_dicts), domain_hint=domain_hint)
        if not decision.safe:
            logger.warning('입력 가드 차단 (tenant=%s, conversation=%s): %s',
                           self.tenant_id, conversation.id, decision.reason)
            return _routed("blocked", block_reason=decision.reason)

        # RETRY 디스패치 (#59) — 분류기는 "재요청 발화"라는 표면 사실만 인식하고,
        # 무엇을 다시 할지는 여기서 직전 턴 상태(팩트)로 결정론 해소한다 (Rasa repeat-intent 패턴).
        # RETRY는 여기서 소멸하는 전이 인텐트 — 이후 파이프라인·저장 계층은 이 개념을 모른다.
        precomputed_standalone: str | None = None
        if decision.intent == "RETRY":
            pair = last_unanswered_turn(messages)
            if pair is None:
                # 되돌릴 턴 없음(직전이 done이거나 이력 없음) — 회상·재설명은 OTHER가 담당
                return _routed("other")
            prev_user, prev_assistant = pair     # prev_user = 체인 머리(실질 질문 — "다시" 연타여도)
            display_query = query                # 저장·화면용 — 사용자가 친 "다시" 그대로
            query = prev_user.content            # 검색·프롬프트용 — 답을 못 받은 실질 질문
            self._resolved_query = query         # 이 턴도 실패하면 이 값이 백필돼야 체인이 이어진다
            if prev_assistant.intent == "OTHER":
                return _routed("other")          # 원래 OTHER였던 턴(대화 요약 등)은 원래 경로로 재실행
            # 재시도의 의미 = 원본 검색 재현 — 그 턴의 condense 결과를 재사용 (LLM 1콜 절감,
            # 취소 턴이 낀 이력으로 재작성돼 다른 검색어가 나오는 변형 차단)
            # 주의: 원 턴이 ATTACHMENT(검색 스킵)였다는 사실은 저장에 안 남아(전이 인텐트),
            # 재실행은 일반 knowledge로 검색이 돈다 — 의도된 수용(#63 결정: 빈도 극저,
            # 규칙 1의 지시 정의가 오도 완화. 복원하려면 재분류 1콜이 필요해 과설계로 기각).
            precomputed_standalone = prev_user.standalone_query or prev_user.content
        elif decision.intent == "ATTACHMENT":
            # ATTACHMENT 디스패치 (#63) — 분류기는 "첨부 자체를 대상으로 한 요청"이라는 표면
            # 사실만 인식하고, 여기서 첨부 유무(팩트)로 해소한다. RETRY와 같은 전이 인텐트.
            if not attachment_dicts:
                return _routed("other")          # 가리킬 첨부가 없다 — OTHER가 되묻기 유도
            return self._prepare_attachment_only(
                conversation.id, assistant_message_id, query, messages,
                attachment_dicts, domain_hint)
        elif decision.intent == "OTHER":
            return _routed("other")

        return await self._prepare_knowledge(
            conversation.id, assistant_message_id, query, messages,
            attachment_dicts, new_attachment_dicts, domain_hint,
            display_query=display_query, precomputed_standalone=precomputed_standalone)

    def _prepare_attachment_only(
            self,
            conversation_id: int,
            assistant_message_id: int,
            query: str,
            messages: list[Message],
            attachment_dicts: list[dict],
            domain_hint: str | None,
    ) -> PreparedRag:
        """ATTACHMENT 경로(#63): 검색·재작성을 통째로 생략하고 첨부만 근거로 생성한다.

        _prepare_knowledge에 스킵 플래그를 꽂지 않고 분리한 이유 — 그쪽 단계(condense→
        retrieve→FAQ 스냅샷→캐시 조회)를 전부 안 타므로, 플래그면 한 함수가 사실상 두
        함수가 된다. retrieval은 None이 아니라 빈 결과다: "검색을 안 했다"는 상태 표현이자
        route=knowledge ⟺ retrieval 존재 불변식(__post_init__)의 요구.
        no_evidence=False — 근거(첨부)가 실재하므로 의미상으로도 맞다.
        캐시는 무접점: should_cache가 첨부 있음으로 항상 False (기존 게이트).
        """
        return PreparedRag(
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
            original_query=query,
            standalone_query=query,   # condense 생략 — 검색을 안 하므로 재작성이 무의미
            prior_turns=build_prior_turns(messages, settings.history_budget_tokens),
            retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
            sources=[],
            source_doc_ids=[],
            attachments=attachment_dicts,
            domain_hint=domain_hint,
            attachment_grounded=True,   # 저장 라벨 'ATTACHMENT' — stats KB 지표에서 제외 (#63)
        )

    async def _open_turn(self, conversation_id: int, query: str,
                         new_attachment_dicts: list[dict]) -> int:
        """턴 시작 저장 — user 메시지 + assistant 자리표시를 커밋하고 자리표시 id를 반환한다 (#72).

        query는 사용자가 실제 친 원문이다. RETRY(#59)의 질의 치환은 이 시점보다 뒤에 일어나므로,
        여기 저장되는 값이 곧 화면·기록에 남아야 할 발화다 (recorded_query가 늦게 계산하던 값과 동치).
        standalone_query는 아직 모른다 — finalize_turn이 종료 시점에 백필한다.

        이 commit은 신규 대화의 INSERT까지 함께 확정한다. 대화만 따로 커밋하던 예전 구조는
        대화와 첫 턴 사이에 실패 창을 남겨 빈 대화를 만들었다 — 한 트랜잭션으로 묶어 없앤다.
        """
        assistant = await add_pending_turn(
            self.session, self.tenant_id, conversation_id, query,
            attachments=new_attachment_dicts or None, user_id=self.user_id,
        )
        await self.session.flush()   # 자리표시 id 확보 — 이후 전 경로가 이 id로 UPDATE한다
        await self.session.commit()  # 여기서부터 질문은 durable
        return assistant.id

    async def _fail_turn(self, assistant_message_id: int, resolved_query: str) -> None:
        """prepare() 실패 시 자리표시를 failed로 닫는다 (#72). 정리 자체의 실패는 삼킨다 —
        원 예외를 가리면 진짜 원인이 사라진다. 삼켜도 스테일 스윕이 결국 회수한다.

        resolved_query를 짝 user 행의 standalone_query에 백필한다. 실패 턴도 RETRY 대상이라
        (last_unanswered_turn) "다시" 연타의 체인 되감기가 이 값의 동일성으로 이어지는데,
        비워두면 두 번째 "다시"부터 질문 슬롯이 "다시"로 퇴화한다 — #59가 취소 턴에서 막아둔
        바로 그 함정이다. RETRY 재실행 턴이면 치환된 실질 질문이 들어와야 체인이 성립한다.
        """
        try:
            await finalize_turn(self.session, self.tenant_id, assistant_message_id,
                                '', [], status=TurnStatus.FAILED, standalone_query=resolved_query)
            await self.session.commit()
        except Exception:
            logger.exception('자리표시 failed 마감 실패 (message_id=%s)', assistant_message_id)

    async def _resolve_conversation(
            self,
            conversation_id: int | None,
            attachments: list[QueryAttachment] | None,
    ) -> tuple[Conversation, list[dict], list[dict]]:
        """대화를 확보하고 컨텍스트에 실을 첨부를 합본한다.

        **커밋하지 않는다** (#72). 신규 대화 INSERT는 세션에만 올리고, 곧바로 이어지는
        _open_turn의 commit이 대화와 첫 턴을 **한 트랜잭션으로** 확정한다. 예전엔 여기서
        따로 commit해 persist-before-stream(REVIEW findings ②)을 보장했는데, 그러면 대화만
        커밋된 뒤 턴 저장 전에 실패하는 창이 남아 빈 대화가 생겼다. 뒤 commit이 meta 전송보다
        한참 앞이라 그 보장은 그대로 유지된다.

        반환은 (대화, 주입용 첨부, 이번 턴 신규 첨부) — 신규분은 _open_turn이 user 메시지에
        저장하고, 주입용은 프롬프트 <첨부 문서> 블록에 들어간다. **둘은 별개다**: max_attachments가
        주입만 제한하므로 주입용이 비어도 신규분은 있을 수 있다 (그 어긋남이 #36의 캐시 유실 버그였다).
        """
        conversation = await ensure_conversation(self.session, self.tenant_id, conversation_id, user_id=self.user_id)

        new_attachment_dicts = [{'filename': a.filename, 'text': a.text} for a in (attachments or [])]
        # 누적 첨부 중 최신 max_attachments개만 컨텍스트에 주입 (오래된 것 제외 — 30K 예산 관리).
        # 저장(save)은 전부 유지되므로 히스토리엔 남고, 주입만 최신분으로 제한. FE에 고정 안내 문구.
        history = await self._load_history_attachments(conversation.id, settings.max_attachments)
        all_attachments = history + new_attachment_dicts
        # max_attachments<=0이면 주입 안 함. (list[-0:]는 '전체'라 0을 그대로 슬라이스하면 정반대 동작 — P2)
        attachment_dicts = all_attachments[-settings.max_attachments:] if settings.max_attachments > 0 else []
        return conversation, attachment_dicts, new_attachment_dicts

    async def _prepare_knowledge(
            self,
            conversation_id: int,
            assistant_message_id: int,
            query: str,
            messages: list[Message],
            attachment_dicts: list[dict],
            new_attachment_dicts: list[dict],
            domain_hint: str | None,
            display_query: str | None = None,
            precomputed_standalone: str | None = None,
    ) -> PreparedRag:
        """KNOWLEDGE 경로: 질의 재작성 → 검색 → 출처·FAQ 스냅샷 → semantic 캐시 조회.

        **부수효과**: 캐시가 히트하면 get_semantic이 hit_count·last_hit_at을 UPDATE한다
        (이름은 조회지만 쓰기가 있다).
        display_query/precomputed_standalone은 RETRY 재실행(#59) 전용 — 일반 경로는 None.
        """
        # 질의 재작성 — 히스토리는 condense 전용 예산으로 (답변용 2000과 용도 분리).
        # 플래그 on(#5): 멀티턴이면 같은 자리 1콜로 멀티쿼리(재작성 1 + 어휘 변형 2). 첫 줄만
        # standalone으로 저장·캐시에 쓰이고, 변형은 검색 전용.
        # 단일턴은 플래그와 무관하게 현행 경로(LLM 스킵) — 단일턴 확장은 세 차례 측정(분리형·
        # 선언형·절차형)에서 일관되게 Hit@1 손실(변형의 RRF 희석)이라 멀티턴 전용으로 게이트(#5).
        expanded: list[str] = []
        if precomputed_standalone is not None:
            # RETRY 재실행(#59) — 재시도의 의미 = 원본 검색 재현. condense를 다시 태우면
            # 취소 턴이 낀 이력으로 재작성돼 원래와 다른 검색어가 나올 수 있다.
            standalone_query = precomputed_standalone
        else:
            history_for_condense = trim_messages_for_condense(messages, settings.condense_history_budget_tokens)
            if settings.condense_multi_query_enabled and messages:
                queries = await condense_to_queries(self._llm, query, history_for_condense)
                standalone_query, expanded = queries[0], queries[1:]
            else:
                standalone_query = await condense_query(self._llm, query, history_for_condense)

        # 검색 (exact 캐시 제거 — semantic 캐시가 검색 후 doc집합 비교로 처리)
        retrieval = await retrieve(self.session, self.tenant_id, standalone_query, expanded_queries=expanded)

        # 인용 후보 (문서 단위) — 순서가 곧 인용 번호(citation_labels 불변식):
        # 프롬프트 [번호]·꼬리 제약·꼬리 파서가 전부 이 리스트 순서를 공유한다.
        sources = [] if retrieval.no_evidence else sources_from_chunks(retrieval.chunks)

        source_doc_ids = _source_doc_ids(retrieval.chunks)
        # 근거 FAQ 버전 스냅샷 — 생성이 끝난 cache.save_answer 시점에 재조회·등치 비교해,
        # 생성 중 FAQ가 수정됐으면 저장을 스킵한다 (write-back 레이스 차단, #16)
        faq_versions = await cache.snapshot_faq_versions(self.session, self.tenant_id, source_doc_ids)
        prior_turns = build_prior_turns(messages, settings.history_budget_tokens)

        # semantic 캐시 조회 — 첨부가 있으면 우회한다. **주입용·신규분 둘 다 봐야 한다** (#36):
        # max_attachments<=0이면 주입용이 강제로 비므로, 신규 첨부만 확인하면 이 가드를 통과해
        # 캐시 답변이 재생되고 이번 턴 첨부가 save()에서 유실됐다 (설정 하나로 재현되는 데이터 유실).
        if not retrieval.no_evidence and not attachment_dicts and not new_attachment_dicts:
            semantic_hit = await cache.get_semantic(
                self.session, self.tenant_id, standalone_query, source_doc_ids,
                # 검색이 방금 만든 벡터를 그대로 넘긴다 — 같은 문자열을 다시 임베딩하지 않는다 (#50)
                query_embedding=retrieval.query_embedding,
                llm=self._llm)   # 임계 아래 대역 후보의 재사용 판정기 (#113)
            if semantic_hit is not None:
                return PreparedRag(
                    conversation_id=conversation_id,
                    assistant_message_id=assistant_message_id,
                    original_query=query,
                    standalone_query=standalone_query,
                    prior_turns=prior_turns,
                    retrieval=retrieval,
                    cached_answer=semantic_hit.answer,
                    cache_kind="semantic",
                    source_doc_ids=semantic_hit.source_doc_ids,
                    sources=semantic_hit.sources,
                    attachments=attachment_dicts,
                    domain_hint=domain_hint,
                    display_query=display_query,
                )

        return PreparedRag(
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
            original_query=query,
            standalone_query=standalone_query,
            prior_turns=prior_turns,
            retrieval=retrieval,
            sources=sources,
            source_doc_ids=source_doc_ids,
            attachments=attachment_dicts,
            domain_hint=domain_hint,
            faq_versions=faq_versions,
            query_embedding=retrieval.query_embedding,   # maybe_cache가 save_answer로 넘긴다 (#50)
            display_query=display_query,
        )

    async def _load_history_attachments(self, conversation_id: int, limit: int) -> list[dict]:
        """대화 히스토리(messages)에 저장된 첨부를 시간순으로 모아 반환한다.

        첨부는 첨부한 턴의 user 메시지에 저장되지만(히스토리의 일부),
        프롬프트 규칙상 이전 대화는 근거가 아니므로 여기서 모아
        매 턴 <첨부 문서> 근거 블록으로 재주입한다.

        주입은 어차피 최신 limit개만 쓰므로(max_attachments 슬라이스) 조회도 최신
        limit행으로 제한 — 행당 첨부가 1개 이상이라 limit행이면 첨부 limit개 이상 확보.

        차단 턴의 첨부는 제외한다(#63) — 가드가 질문을 막아도 그 턴의 첨부가 이후 턴
        프롬프트로 재진입하면 차단이 반쪽이 된다(첨부 텍스트에 실린 인젝션이 통과).
        이력 격리(#22)와 같은 원칙: 저장·조회 API는 그대로, 프롬프트 주입만 막는다.
        user 행은 항상 status='done'이라 그 턴의 assistant(blocked) 존재로 판별한다.
        failed/cancelled 턴의 첨부는 유지 — 질문·첨부 모두 정상인 턴이다(사용자 결정).
        """
        if limit <= 0:   # 주입 안 함 — 조회 자체 생략
            return []
        assistant = aliased(Message)
        blocked_turn = (
            select(assistant.id)
            .where(assistant.tenant_id == self.tenant_id)   # 격리 — 서브쿼리에도 명시
            .where(assistant.question_message_id == Message.id)
            .where(assistant.status == 'blocked')
            .exists()
        )
        rows = (await self.session.execute(
            select(Message.attachments)
            .where(Message.tenant_id == self.tenant_id)   # 격리 — WHERE 절 명시
            .where(Message.conversation_id == conversation_id)
            .where(Message.attachments.is_not(None))
            .where(~blocked_turn)
            .order_by(Message.id.desc())
            .limit(limit)
        )).scalars().all()[::-1]   # 최신 limit행 → 시간순 복원
        # row가 JSON null(과거 저장분)일 수 있어 방어 — SQL NULL과 JSON null은 IS NOT NULL 판정이 다름
        return [attachment for row in rows if row for attachment in row]

    def _set_vllm_finish(self, reason: str) -> None:
        """astream의 on_finish_reason 콜백 (#129) — 값 보관만. 기록은 streaming.py가 한다."""
        self.last_vllm_finish_reason = reason

    async def generate(self, prepared: PreparedRag) -> AsyncIterator[str]:
        """답변 토큰을 생성한다. 갈래는 셋뿐이다 (#36):

        1. 확정 답변이 있으면(입력 차단·캐시 히트·근거 없음) LLM을 부르지 않고 그대로 반환.
           판정은 resolved_answer 한 곳 — needs_generation도 같은 값을 본다.
        2. OTHER는 검색 없이 이전 대화를 실어 제약된 프롬프트로 생성
           (인사·대화 요약·회상·자기소개 등. 서비스 사실은 프롬프트 방화벽으로 차단).
        3. 그 외는 KNOWLEDGE 정상 경로 — standalone_query와 검색 청크로 답변 프롬프트를 만든다.

        3번에 도달했다는 것은 resolved_answer가 None이고 route도 other가 아니라는 뜻이므로
        route는 knowledge이고, __post_init__ 불변식에 따라 retrieval이 반드시 존재한다 —
        예전처럼 retrieval을 null 검사 없이 역참조해도 안전한 근거가 구조에 있다.
        """
        # 이 값 하나를 SSE로 흘려도 되는 이유: 확정 답변은 쪼갤 필요가 없다. 예전엔 캐시 답변을
        # 40자로 잘라 yield했지만 소비처(immediate_stream)가 곧바로 ''.join으로 되붙여
        # 단일 delta 이벤트로 보낸다 — 청크 경계를 관측하는 곳이 없었다 (#26 정리의 잔재).
        answer = prepared.resolved_answer
        if answer is not None:
            yield answer
            return

        self.last_vllm_finish_reason = None   # 호출마다 리셋 — 이전 턴 값 누출 방지 (#129)

        if prepared.route == "other":
            try:
                user_msg = build_other_user_message(prepared.original_query, prepared.prior_turns)
                async for token in self._llm.astream(
                        build_chat_prompt(build_other_system_prompt(prepared.domain_hint), user_msg),
                        on_finish_reason=self._set_vllm_finish):
                    yield token
            except Exception:
                logger.exception('LLM error(other gen)')
                yield SMALLTALK_ANSWER   # 생성 실패 시 폴백
            return

        # "질문:" 자리는 원 질문, 검색에 쓴 재작성 질의는 참고로 병기한다 (#48).
        # 검색·캐시 키·메시지 저장은 그대로 standalone_query를 쓴다 — 바뀌는 건 이 슬롯뿐이다.
        prompt = build_knowledge_generation_prompt(
            prepared.original_query,
            prepared.retrieval.chunks,
            standalone_query=prepared.standalone_query,
            prior_turns=prepared.prior_turns,
            attachments=prepared.attachments,
            domain_hint=prepared.domain_hint,
        )

        # 출처 꼬리 강제 (#56 도입 → #65에서 정규식→structural_tag 재구현. 정규식이 왜 무효였는지는
        # build_citation_constraint docstring — 되돌리지 말 것). 유효 범위의 후보 번호만 꼬리에 온다.
        # 서버가 미지원(400 등, 첫 토큰 전에 터진다)이면 제약 없이 재시도(fail-open) —
        # 이때도 꼬리 파서의 번호 범위 검증은 동일하므로 조용한 오답 확정은 없다.
        constraint = build_citation_constraint(*prepared.citation_candidates)
        stream = self._llm.astream(prompt, extra_body=constraint,
                                   on_finish_reason=self._set_vllm_finish)
        try:
            first = await anext(stream, None)
        except Exception:
            logger.warning('꼬리 제약 요청 실패 — 제약 없이 재시도 (fail-open, #56·#65)')
            stream = self._llm.astream(prompt, on_finish_reason=self._set_vllm_finish)
            first = await anext(stream, None)
        if first is not None:
            yield first
            async for token in stream:
                yield token

    async def finalize(self, prepared: PreparedRag, answer: str, citations: list[SourceCitation],
                       status: TurnStatus = TurnStatus.DONE, latency_ms: int | None = None) -> None:
        """턴을 종료한다 — 자리표시를 최종 결과로 UPDATE한다. **모든 경로 공용** (#72).

        즉시 경로(차단·캐시히트·근거없음)는 요청 세션에서, 생성 경로는 백그라운드 태스크가
        '자기 세션으로 만든 RagService'에서 부른다 (self.session=태스크 세션).
        commit은 호출자가 담당. 실패면 status='failed', answer=''로 호출.

        예전엔 즉시 경로만 save_exchange로 user+assistant를 원샷 INSERT했는데, 그 형태는
        라우팅이 끝난 뒤에만 가능해 prepare() 실패 시 질문이 유실됐다 — #72에서 턴 시작
        저장으로 옮기면서 종료 지점도 이 함수 하나로 합쳐졌다.

        캐시 적재는 여기서 하지 않는다(#56 재배치) — done 전송보다 앞(크리티컬 패스)에 있으면
        임베딩 TEI 왕복만큼 각주 표시가 늦고, 저장 실패가 완결된 턴을 failed로 오염시켰다.
        호출자가 done을 내보낸 뒤 maybe_cache를 따로 부른다.

        citations: 실제 인용된 출처만 (#56) — 호출자(스트림 조립부)가 확정해 넘긴다.
        정상 완료(done)에만 의미 — 취소/실패는 status 가드가 어차피 비운다.
        sources 컬럼도 인용만 저장한다(저장=응답=스트림 정합, #56 확정).
        """
        source_dicts = [] if status != TurnStatus.DONE else [c.model_dump() for c in citations]
        await finalize_turn(
            self.session, self.tenant_id, prepared.assistant_message_id, answer, source_dicts,
            status=status, latency_ms=latency_ms,
            # 인용 확정은 정상 완료(done)에만 의미 — blocked/failed/cancelled는 항상 []/False
            cited_docs=[c.filename for c in citations] if status == TurnStatus.DONE else [],
            intent=prepared.intent_label,
            # 턴 시작 시점엔 몰랐던 값들 — 짝 user 행의 standalone_query까지 여기서 확정한다
            standalone_query=prepared.standalone_query,
            cache_kind=prepared.cache_kind,   # 'semantic'=캐시 재생 답변 (기간별 히트율 집계용)
            # 입력 차단 턴은 status·block_reason으로 식별 가능해야 한다 — 이력 격리
            # (load_recent_messages)와 차단 집계가 이 값에 의존 (#22)
            block_reason=prepared.block_reason,
        )


    async def maybe_cache(self, prepared: PreparedRag, answer: str,
                          citations: list[SourceCitation]) -> None:
        """근거 있는 신규 LLM 응답만 semantic 캐시에 적재한다.

        생성 경로가 done 전송 '뒤'에 부른다(#56 재배치, 사용자 결정 8/18) — 캐시는 있으면
        좋은 부가물이지 턴 완결의 조건이 아니다. INSERT를 크리티컬 패스에서 빼 각주
        (done.citations) 표시가 빨라지고, 저장 실패는 로그로만 남는다(턴은 이미 done).
        (재배치 당시엔 임베딩 TEI 왕복도 여기 있었다 — #50에서 prepared.query_embedding
        재사용으로 없어져 지금 뒤로 미루는 비용은 INSERT뿐이다.)
        근거 없는 답변을 제외하는 이유: 문서가 추가되면 답이 바뀌어야 한다 (§14 규칙 6과 같은 취지).
        캐시도 인용만 저장한다(#56) — 히트 재생 시 prepared.sources로 복원돼 그대로
        citations가 된다. 무효화 키(source_doc_ids)는 검색 근거 전체 그대로 — 캐시 정확성의
        기준은 "검색된 문서 집합"이지 인용 집합이 아니다.
        """
        # 판정을 거절 문구 부분일치(옛 is_refusal)에서 **실인용 개수**로 바꿨다 (#61).
        # 캐시가 묻는 질문은 "거절했나"가 아니라 "근거 없이 나온 답인가"이고, citations가
        # 그 질문에 직접 답한다. 폐기 사유·실측은 rag/citation_tail.py 모듈 docstring(단일 정의점).
        # should_cache 스코프가 knowledge 실생성 경로로 한정되므로(즉시 경로는 should_cache=False)
        # 여기서의 citations는 항상 resolve_citations의 실결과다.
        if not prepared.should_cache or not citations:
            return
        await cache.save_answer(
            self.session,
            self.tenant_id,
            prepared.standalone_query,
            answer,
            citations,
            prepared.source_doc_ids,
            faq_versions=prepared.faq_versions,
            query_embedding=prepared.query_embedding,   # 검색이 만든 벡터 재사용 (#50)
        )


def _source_doc_ids(chunks: list[RetrievedChunk]) -> list[int]:
    """검색 청크의 출처 id 목록 (중복 제거) — semantic 캐시의 집합 비교·무효화 키.

    FAQ 출처는 문서 id와 충돌하지 않도록 음수 네임스페이스(-faq_id)로 표현한다.
    FAQ 수정 시 invalidate_source(tenant, -faq_id)가 같은 키를 지우는 것과 짝 (routers/faqs.py).
    """
    ids = {chunk.document_id for chunk in chunks if chunk.document_id is not None}
    ids |= {-chunk.faq_id for chunk in chunks if chunk.faq_id is not None}
    return list(ids)
