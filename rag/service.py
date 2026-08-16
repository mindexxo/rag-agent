"""RAG 서비스 — 한 턴의 수명을 조율한다.

    prepare()   라우팅·검색어·근거·캐시히트를 확정 (부수효과 있음: 대화 commit, 캐시 hit_count)
    generate()  확정 답변이 있으면 그대로, 없으면 LLM 스트리밍
    save()      즉시 경로의 INSERT   /   begin_turn()+finalize()  생성 경로의 자리표시→UPDATE

즉시 경로와 생성 경로를 가르는 기준은 **답이 이미 확정됐느냐**다 —
그 판정은 PreparedRag.resolved_answer 한 곳에 있고 generate()·needs_generation이 함께 본다.

전역 싱글톤 금지 — 요청마다 tenant_id가 다르므로 생성자 인자로 받는다.
"""
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.conversation import ensure_conversation, load_recent_messages, condense_query, condense_to_queries, build_prior_turns, trim_messages_for_condense, save_exchange, add_pending_turn, finalize_turn
from rag import otel
from rag.guardrail import classify_and_guard
from rag.clients import shared_llm
from rag.models import Conversation, Document, Message
from rag.answer_check import cited_filenames, is_refusal
from rag.prompt_texts import BLOCKED_INPUT_ANSWER, NO_EVIDENCE_ANSWER, SMALLTALK_ANSWER
from rag.prompts import (build_chat_prompt, build_other_system_prompt, build_other_user_message,
                         build_system_prompt, build_user_message)
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
    new_attachments: list[dict] = field(default_factory=list)  # 이번 턴에 새로 동봉된 것 — save 시 user 메시지에 저장
    domain_hint: str | None = None   # [임시] 테넌트 지식 범위 설명 — 생성 프롬프트 주입용, 저장 안 함 (#1)
    assistant_message_id: int | None = None   # 생성 경로: 자리표시 assistant 메시지 id (백그라운드 태스크가 UPDATE할 대상)
    block_reason: str | None = None   # route='blocked'일 때 가드 판정 사유 — 저장·집계용 (#22)
    faq_versions: dict[int, datetime] | None = None   # 근거 FAQ {id: updated_at} 스냅샷 — cache.set 낙관적 검증 기준 (#16)

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
    def intent_label(self) -> str | None:
        """저장용 인텐트 라벨 — 답변률 분모(KNOWLEDGE) 판별에 쓴다.
        blocked는 인텐트 판정 자체가 무의미(unsafe 입력)라 NULL."""
        return {"knowledge": "KNOWLEDGE", "other": "OTHER"}.get(self.route)

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


    async def prepare(
            self,
            query: str,
            conversation_id: int | None = None,
            attachments: list[QueryAttachment] | None = None,
            domain_hint: str | None = None,
    ) -> PreparedRag:
        """한 턴을 처리할 컨텍스트를 확정한다 — 라우팅·검색어·근거·캐시히트가 여기서 결정된다.

        **부수효과가 있다** — 새 대화면 INSERT + commit(되돌릴 수 없음), 캐시 히트면 hit_count가
        오른다. 재호출하면 대화가 중복 생성되므로 한 요청에 한 번만 부를 것.
        LLM도 2회 부른다(인텐트 분류·질의 재작성). 이름이 가벼워 보이지만 요청 지연의
        상당 부분이 여기서 나온다 — 캐시 히트 턴에서는 사실상 전부다.

        성격이 다른 셋을 단계로 나눠 뒀다 (#36):
          _resolve_conversation  대화 확보(쓰기) + 첨부 합본
          (본문)                 라우팅 판정 → blocked/other면 여기서 끝
          _prepare_knowledge     질의 재작성 → 검색 → 출처·FAQ스냅샷 → 캐시 조회(쓰기)
        """
        conversation, attachment_dicts, new_attachment_dicts = await self._resolve_conversation(
            conversation_id, attachments)
        messages = await load_recent_messages(self.session, self.tenant_id, conversation.id)

        def _routed(route: str, block_reason: str | None = None) -> PreparedRag:
            # 검색·인용 없이 라우팅 결과만 담는 PreparedRag (blocked/other 공용).
            # retrieval=None이 route와 짝을 이룬다 — __post_init__이 그 불변식을 강제한다.
            return PreparedRag(
                block_reason=block_reason,
                conversation_id=conversation.id,
                original_query=query,
                standalone_query=query,
                prior_turns=build_prior_turns(messages, settings.history_budget_tokens),
                retrieval=None,
                sources=[],
                source_doc_ids=[],
                route=route,
                attachments=attachment_dicts,
                new_attachments=new_attachment_dicts,
                domain_hint=domain_hint,
            )

        # 입력 가드레일 + 인텐트 분류 (통합 1회 호출) — 히스토리 유무와 무관하게 항상 실행
        decision = await classify_and_guard(self._llm, query, has_attachments=bool(attachment_dicts), domain_hint=domain_hint)
        if not decision.safe:
            logger.warning('입력 가드 차단 (tenant=%s, conversation=%s): %s',
                           self.tenant_id, conversation.id, decision.reason)
            return _routed("blocked", block_reason=decision.reason)
        if decision.intent == "OTHER":
            return _routed("other")

        return await self._prepare_knowledge(
            conversation.id, query, messages, attachment_dicts, new_attachment_dicts, domain_hint)

    async def _resolve_conversation(
            self,
            conversation_id: int | None,
            attachments: list[QueryAttachment] | None,
    ) -> tuple[Conversation, list[dict], list[dict]]:
        """대화를 확보하고 컨텍스트에 실을 첨부를 합본한다.

        **부수효과**: conversation_id가 None이면 INSERT 후 즉시 commit한다 — 되돌릴 수 없다.
        persist-before-stream: meta 이벤트로 id를 FE에 노출하기 전에 durable해야, disconnect·
        blocked·error로 스트림이 끊겨도 FE가 받은 id가 유효하다 (REVIEW findings ②).

        반환은 (대화, 주입용 첨부, 이번 턴 신규 첨부) — 신규분은 save()가 user 메시지에 저장하고,
        주입용은 프롬프트 <첨부 문서> 블록에 들어간다. **둘은 별개다**: max_attachments가 주입만
        제한하므로 주입용이 비어도 신규분은 있을 수 있다 (그 어긋남이 #36의 캐시 유실 버그였다).
        """
        conversation = await ensure_conversation(self.session, self.tenant_id, conversation_id, user_id=self.user_id)
        if conversation_id is None:
            await self.session.commit()

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
            query: str,
            messages: list[Message],
            attachment_dicts: list[dict],
            new_attachment_dicts: list[dict],
            domain_hint: str | None,
    ) -> PreparedRag:
        """KNOWLEDGE 경로: 질의 재작성 → 검색 → 출처·FAQ 스냅샷 → semantic 캐시 조회.

        **부수효과**: 캐시가 히트하면 get_semantic이 hit_count·last_hit_at을 UPDATE한다
        (이름은 조회지만 쓰기가 있다).
        """
        # 질의 재작성 — 히스토리는 condense 전용 예산으로 (답변용 2000과 용도 분리).
        # 플래그 on(#5): 멀티턴이면 같은 자리 1콜로 멀티쿼리(재작성 1 + 어휘 변형 2). 첫 줄만
        # standalone으로 저장·캐시에 쓰이고, 변형은 검색 전용.
        # 단일턴은 플래그와 무관하게 현행 경로(LLM 스킵) — 단일턴 확장은 세 차례 측정(분리형·
        # 선언형·절차형)에서 일관되게 Hit@1 손실(변형의 RRF 희석)이라 멀티턴 전용으로 게이트(#5).
        expanded: list[str] = []
        history_for_condense = trim_messages_for_condense(messages, settings.condense_history_budget_tokens)
        if settings.condense_multi_query_enabled and messages:
            queries = await condense_to_queries(self._llm, query, history_for_condense)
            standalone_query, expanded = queries[0], queries[1:]
        else:
            standalone_query = await condense_query(self._llm, query, history_for_condense)

        # 검색 (exact 캐시 제거 — semantic 캐시가 검색 후 doc집합 비교로 처리)
        retrieval = await retrieve(self.session, self.tenant_id, standalone_query, expanded_queries=expanded)

        sources = []
        if not retrieval.no_evidence:
            sources = await _build_sources(self.session, self.tenant_id, retrieval.chunks)

        source_doc_ids = _source_doc_ids(retrieval.chunks)
        # 근거 FAQ 버전 스냅샷 — 생성이 끝난 cache.set 시점에 재조회·등치 비교해,
        # 생성 중 FAQ가 수정됐으면 저장을 스킵한다 (write-back 레이스 차단, #16)
        faq_versions = await cache.snapshot_faq_versions(self.session, self.tenant_id, source_doc_ids)
        prior_turns = build_prior_turns(messages, settings.history_budget_tokens)

        # semantic 캐시 조회 — 첨부가 있으면 우회한다. **주입용·신규분 둘 다 봐야 한다** (#36):
        # max_attachments<=0이면 주입용이 강제로 비므로, 신규 첨부만 확인하면 이 가드를 통과해
        # 캐시 답변이 재생되고 이번 턴 첨부가 save()에서 유실됐다 (설정 하나로 재현되는 데이터 유실).
        if not retrieval.no_evidence and not attachment_dicts and not new_attachment_dicts:
            semantic_hit = await cache.get_semantic(
                self.session, self.tenant_id, standalone_query, source_doc_ids)
            if semantic_hit is not None:
                return PreparedRag(
                    conversation_id=conversation_id,
                    original_query=query,
                    standalone_query=standalone_query,
                    prior_turns=prior_turns,
                    retrieval=retrieval,
                    cached_answer=semantic_hit.answer,
                    cache_kind="semantic",
                    source_doc_ids=semantic_hit.source_doc_ids,
                    sources=semantic_hit.sources,
                    attachments=attachment_dicts,
                    new_attachments=new_attachment_dicts,
                    domain_hint=domain_hint,
                )

        return PreparedRag(
            conversation_id=conversation_id,
            original_query=query,
            standalone_query=standalone_query,
            prior_turns=prior_turns,
            retrieval=retrieval,
            sources=sources,
            source_doc_ids=source_doc_ids,
            attachments=attachment_dicts,
            new_attachments=new_attachment_dicts,
            domain_hint=domain_hint,
            faq_versions=faq_versions,
        )

    async def _load_history_attachments(self, conversation_id: int, limit: int) -> list[dict]:
        """대화 히스토리(messages)에 저장된 첨부를 시간순으로 모아 반환한다.

        첨부는 첨부한 턴의 user 메시지에 저장되지만(히스토리의 일부),
        프롬프트 규칙상 이전 대화는 근거가 아니므로 여기서 모아
        매 턴 <첨부 문서> 근거 블록으로 재주입한다.

        주입은 어차피 최신 limit개만 쓰므로(max_attachments 슬라이스) 조회도 최신
        limit행으로 제한 — 행당 첨부가 1개 이상이라 limit행이면 첨부 limit개 이상 확보.
        """
        if limit <= 0:   # 주입 안 함 — 조회 자체 생략
            return []
        rows = (await self.session.execute(
            select(Message.attachments)
            .where(Message.tenant_id == self.tenant_id)   # 격리 — WHERE 절 명시
            .where(Message.conversation_id == conversation_id)
            .where(Message.attachments.is_not(None))
            .order_by(Message.id.desc())
            .limit(limit)
        )).scalars().all()[::-1]   # 최신 limit행 → 시간순 복원
        # row가 JSON null(과거 저장분)일 수 있어 방어 — SQL NULL과 JSON null은 IS NOT NULL 판정이 다름
        return [attachment for row in rows if row for attachment in row]

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
        # 단일 token 이벤트로 보낸다 — 청크 경계를 관측하는 곳이 없었다 (#26 정리의 잔재).
        answer = prepared.resolved_answer
        if answer is not None:
            yield answer
            return

        if prepared.route == "other":
            try:
                user_msg = build_other_user_message(prepared.original_query, prepared.prior_turns)
                async for token in self._llm.astream(build_chat_prompt(build_other_system_prompt(prepared.domain_hint), user_msg)):
                    yield token
            except Exception:
                logger.exception('LLM error(other gen)')
                yield SMALLTALK_ANSWER   # 생성 실패 시 폴백
            return

        prompt = build_chat_prompt(
            build_system_prompt(prepared.domain_hint),
            build_user_message(
                prepared.standalone_query,
                prepared.retrieval.chunks,
                prior_turns=prepared.prior_turns,
                attachments=prepared.attachments,
            ),
        )

        async for token in self._llm.astream(prompt):
            yield token

    async def save(self, prepared: PreparedRag, answer: str, latency_ms: int | None = None) -> None:
        """완성된 답변을 대화 메시지로 등록한다.
        이 함수는 session에 메시지를 add만 하고 commit은 호출자가 담당한다.
        """
        # 거절 답변(규칙 3)엔 인용을 저장하지 않는다 — 대화 복원 시 "확인 불가 + 인용" 모순 방지.
        # 캐시 제외(아래)와 같은 in 비교를 사용해 판정 기준을 단일화.
        source_dicts = [] if is_refusal(answer) else [
            source.model_dump()
            for source in prepared.sources
        ]

        assistant = await save_exchange(
            self.session,
            self.tenant_id,
            prepared.conversation_id,
            prepared.original_query,
            prepared.standalone_query,
            answer,
            source_dicts,
            attachments=prepared.new_attachments or None,
            user_id=self.user_id,
            latency_ms=latency_ms,
            cache_kind=prepared.cache_kind,   # 'semantic'=캐시 재생 답변 (기간별 히트율 집계용)
            # 저장 순간에 사실 확정 — 조회는 순수 SQL (거절이면 sources처럼 인용도 비움: 모순 방지)
            cited_docs=[] if is_refusal(answer) else cited_filenames(answer, prepared.sources),
            is_refusal=is_refusal(answer),
            intent=prepared.intent_label,
            # 입력 차단 턴은 status로 식별 가능해야 한다 — 이력 격리(load_recent_messages)와
            # 차단 집계가 모두 이 값에 의존 (#22). 출력 차단은 finalize_turn 쪽이 담당.
            status='blocked' if prepared.route == 'blocked' else 'done',
            block_reason=prepared.block_reason,
        )
        # 즉시 경로는 begin_turn을 안 거쳐 meta의 assistant_message_id가 비어 있었음 —
        # #16의 persist-before-stream 덕에 meta 전송 전에 id 확정 가능 → 피드백 대상 노출 (#8)
        prepared.assistant_message_id = assistant.id

        # 신규 LLM 응답만 캐시에 저장한다.
        # 게이트를 통과했어도 LLM이 스스로 거절한 답변은 제외 —
        # 문서가 추가되면 답이 바뀌어야 하므로 (§14 규칙 6과 같은 취지).
        # in 비교: 모델이 거절 문구 앞뒤에 인용 등을 덧붙이는 변형까지 잡는다.
        await self._maybe_cache(prepared, answer)

    async def begin_turn(self, prepared: PreparedRag) -> None:
        """생성 경로에서 스트림 시작 전에 user 메시지 + assistant 자리표시(generating)를
        등록·commit하고 prepared.assistant_message_id를 세팅한다 (persist-before-stream).
        요청 세션(self.session)으로 실행 — 이 시점엔 요청이 살아있다.
        """
        assistant = await add_pending_turn(
            self.session,
            self.tenant_id,
            prepared.conversation_id,
            prepared.original_query,
            prepared.standalone_query,
            attachments=prepared.new_attachments or None,
            user_id=self.user_id,
        )
        await self.session.flush()
        prepared.assistant_message_id = assistant.id
        await self.session.commit()   # generating 행 durable → 이후 태스크 spawn

    async def finalize(self, prepared: PreparedRag, answer: str, status: str = "done",
                       latency_ms: int | None = None) -> None:
        """생성 완료/실패 시 assistant 자리표시를 UPDATE하고, 성공이면 캐시에 저장한다.
        백그라운드 태스크가 '자기 세션으로 만든 RagService'에서 호출한다 (self.session=태스크 세션).
        commit은 호출자(태스크)가 담당. 실패면 status='failed', answer=''로 호출.
        """
        source_dicts = [] if (status != "done" or is_refusal(answer)) else [
            source.model_dump() for source in prepared.sources
        ]
        await finalize_turn(
            self.session, self.tenant_id, prepared.assistant_message_id, answer, source_dicts,
            status=status, latency_ms=latency_ms,
            # 거절 판정·인용 확정은 정상 완료(done)에만 의미 — blocked/failed는 항상 False/[]
            cited_docs=cited_filenames(answer, prepared.sources) if status == "done" and not is_refusal(answer) else [],
            is_refusal=is_refusal(answer) if status == "done" else False,
            intent=prepared.intent_label,
        )
        if status == "done":
            await self._maybe_cache(prepared, answer)


    async def _maybe_cache(self, prepared: PreparedRag, answer: str) -> None:
        """근거 있는 신규 LLM 응답만 semantic 캐시에 적재한다 — save·finalize 공용 (#36).

        두 곳에 같은 7줄이 있었고 변경 이유도 같았다(캐시 저장 정책이 바뀌면 둘 다 바뀐다).
        거절 답변을 제외하는 이유: 문서가 추가되면 답이 바뀌어야 한다 (§14 규칙 6과 같은 취지).
        in 비교라 모델이 거절 문구 앞뒤에 인용 등을 덧붙인 변형까지 잡는다.
        """
        if not prepared.should_cache or is_refusal(answer):
            return
        await cache.save_answer(
            self.session,
            self.tenant_id,
            prepared.standalone_query,
            answer,
            prepared.sources,
            prepared.source_doc_ids,
            faq_versions=prepared.faq_versions,
        )


async def _build_sources(
        session: AsyncSession,
        tenant_id: str,
        chunks: list[RetrievedChunk]
) -> list[SourceCitation]:
    """검색 청크 목록에서 API 응답용 문서 단위 sources를 만든다.

    FAQ 청크(F3)가 섞여 있으면 'FAQ' 인용 1건으로 접는다 — 원본 파일이 없으므로
    document_id 없이 내려가고, FE는 document_id 없는 인용을 비클릭으로 처리한다.
    """
    doc_ids = [i for i in _source_doc_ids(chunks) if i > 0]
    citations: list[SourceCitation] = []
    if doc_ids:
        documents = (await session.execute(
            select(Document)
            .where(Document.tenant_id == tenant_id)   # 격리 — 모든 조회에 tenant WHERE (프로젝트 원칙, P2)
            .where(Document.id.in_(doc_ids))
        )).scalars().all()
        citations = [SourceCitation(document_id=doc.id, filename=doc.filename, version=doc.version) for doc in documents]
    if any(chunk.faq_id for chunk in chunks):
        citations.append(SourceCitation(document_id=None, filename='FAQ', version=1))
    return citations


def _source_doc_ids(chunks: list[RetrievedChunk]) -> list[int]:
    """검색 청크의 출처 id 목록 (중복 제거) — semantic 캐시의 집합 비교·무효화 키.

    FAQ 출처는 문서 id와 충돌하지 않도록 음수 네임스페이스(-faq_id)로 표현한다.
    FAQ 수정 시 invalidate_source(tenant, -faq_id)가 같은 키를 지우는 것과 짝 (routers/faqs.py).
    """
    ids = {chunk.document_id for chunk in chunks if chunk.document_id is not None}
    ids |= {-chunk.faq_id for chunk in chunks if chunk.faq_id is not None}
    return list(ids)
