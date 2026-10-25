"""RAG 서비스 — 한 턴의 수명을 조율한다.

    prepare()   턴 시작 저장(commit) → 라우팅·검색어·근거·캐시히트 확정
    generate()  확정 답변이 있으면 그대로, 없으면 LLM 스트리밍
    finalize()  턴 종료 — 자리표시를 최종 결과로 UPDATE (모든 경로 공용)

즉시 경로와 생성 경로를 가르는 기준은 답이 이미 확정됐느냐다 — PreparedRag.resolved_answer 한 곳.
요청마다 tenant_id가 다르므로 전역 싱글톤으로 만들지 말 것.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from config import settings
from rag import cache
from rag.citation_labels import sources_from_chunks
from rag.clients import shared_aux_llm, shared_llm
from rag.conversation import (
    build_prior_turns,
    condense_query,
    condense_to_queries,
    ensure_conversation,
    load_recent_messages,
    trim_messages_for_condense,
)
from rag.guardrail import classify_and_guard
from rag.models import Conversation, Message, TurnStatus
from rag.prompt_texts import BLOCKED_INPUT_ANSWER, NO_EVIDENCE_ANSWER, SMALLTALK_ANSWER
from rag.prompts import (
    build_chat_prompt,
    build_citation_tail,
    build_knowledge_generation_prompt,
    build_other_system_prompt,
    build_other_user_message,
)
from rag.retrieval_types import RetrievalResult, RetrievedChunk
from rag.retriever import retrieve
from rag.turn_state import add_pending_turn, finalize_turn, last_unanswered_turn
from schemas.kms import QueryAttachment, SourceCitation

logger = logging.getLogger(__name__)


@dataclass
class PreparedRag:
    """prepare()가 확정해 generate·finalize·maybe_cache가 공유하는 한 턴의 컨텍스트."""

    conversation_id: int
    original_query: str
    standalone_query: str
    prior_turns: list[dict]
    retrieval: RetrievalResult | None
    sources: list[SourceCitation]
    source_doc_ids: list[int]
    cached_answer: str | None = None
    cache_kind: Literal["semantic"] | None = None
    # knowledge=RAG, other=대화/메타/역할밖 제약 생성, blocked=입력 차단
    route: Literal["knowledge", "other", "blocked"] = "knowledge"
    attachments: list[dict] = field(default_factory=list)  # 컨텍스트 주입용 — 히스토리 저장분 + 이번 턴 신규 합본
    domain_hint: str | None = None  # [임시] 테넌트 지식 범위 설명 — 생성 프롬프트 주입용, 저장 안 함
    assistant_message_id: int | None = (
        None  # 생성 경로: 자리표시 assistant 메시지 id (백그라운드 태스크가 UPDATE할 대상)
    )
    block_reason: str | None = None  # route='blocked'일 때 가드 판정 사유 — 저장·집계용
    faq_versions: dict[int, datetime] | None = None  # 근거 FAQ 스냅샷 — cache.save_answer 낙관적 검증 기준
    # 캐시 저장에 재사용할 검색 질의 벡터(근거는 cache.get_semantic). 캐시 히트 턴은 저장하지 않으므로 비운다.
    # repr=False — 긴 float 배열이 로그·트레이스백에 새지 않게.
    query_embedding: list[float] | None = field(default=None, repr=False)
    # RETRY 재실행 턴에서만 채워진다 — 사용자가 친 발화("다시"). 검색·프롬프트는 original_query(직전 실질 질문).
    display_query: str | None = None
    # 첨부만 근거로 답하는 턴(검색 생략) — intent_label이 'ATTACHMENT'가 된다.
    attachment_grounded: bool = False

    def __post_init__(self) -> None:
        """route와 retrieval 짝(knowledge ⟺ retrieval 존재)과 자리표시 id를 생성 시점에 강제한다.

        짝이 깨지면 should_cache가 blocked/other 턴을 캐시에 넣는 식으로 조용히 틀린다.
        """
        if (self.route == "knowledge") != (self.retrieval is not None):
            raise ValueError(
                f"PreparedRag 불변식 위반 — route={self.route!r}인데 retrieval이 "
                f"{'있다' if self.retrieval is not None else '없다'}. "
                "knowledge는 검색을 거치고(캐시 히트 포함), blocked/other는 거치지 않는다."
            )
        if self.assistant_message_id is None:
            # 비어 있으면 finalize_turn이 에러 없이 아무것도 UPDATE하지 않아 턴이 generating으로 남는다.
            raise ValueError(
                "PreparedRag 불변식 위반 — assistant_message_id가 없다. "
                "모든 턴은 _open_turn이 만든 자리표시 id를 들고 다녀야 finalize가 동작한다."
            )

    @property
    def resolved_answer(self) -> str | None:
        """prepare() 시점에 확정된 답변. None이면 LLM 생성이 필요하다 — 즉시/생성 경로 판정의 정의점.

        OTHER는 LLM이 대화성 응답을 만들어야 하므로 cached_answer보다 먼저 None으로 거른다.
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
        """즉시 경로 저장 status의 정의점 — 입력 차단 턴만 BLOCKED, 나머지는 DONE.
        생성 경로의 status는 실행 결과라 여기서 정하지 않는다."""
        return TurnStatus.BLOCKED if self.route == "blocked" else TurnStatus.DONE

    @property
    def intent_label(self) -> str | None:
        """저장용 인텐트 라벨 — 답변률 분모(KNOWLEDGE) 판별에 쓴다. blocked는 NULL.

        첨부만 근거인 턴은 route=knowledge지만 'ATTACHMENT'로 저장한다 — KNOWLEDGE로 남으면
        KB 커버리지 지표(근거미확인율·지식 갭)가 오염된다."""
        if self.attachment_grounded:
            return "ATTACHMENT"
        return {"knowledge": "KNOWLEDGE", "other": "OTHER"}.get(self.route)

    @property
    def citation_candidates(self) -> tuple[list[SourceCitation], list[str]]:
        """인용 후보 (검색 출처, 첨부 파일명) — 꼬리 제약과 파서가 공유하는 단일 파생점.

        둘이 어긋나면 에러 없이 다른 문서로 오귀속된다. 순서 계약은 rag/citation_labels.py가 정의점이다.
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
        """LLM 스트리밍이 필요한 경로인가(백그라운드 태스크 대상) — generate()와 같은 판정을 본다."""
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
        # 요청마다 인스턴스 생성 — tenant_id 격리와 last_* 값이 요청 간 섞이지 않는 근거
        self.tenant_id = tenant_id
        self.session = session
        self.user_id = user_id  # 지표 씨앗 — user 메시지에 기록 (X-User-Id)
        self._llm = shared_llm  # 공용 싱글톤 — 요청마다 HTTP 풀을 만들지 않게
        self._aux_llm = shared_aux_llm  # 분류·재작성·캐시 판정용 — 설정이 비면 _llm과 같은 객체
        # LLM 레벨 종료 사유(rag/llm/base.FinishReason) — streaming.py가 지표·스팬에 기록한다.
        # SSE done의 TurnStatus finish_reason과는 다른 축이다.
        self.last_finish_reason: str | None = None
        self.last_usage = None  # LlmUsage | None — 상용 공급자만 채운다. streaming.py가 스팬에 기록

    async def prepare(
        self,
        query: str,
        conversation_id: int | None = None,
        attachments: list[QueryAttachment] | None = None,
        domain_hint: str | None = None,
    ) -> PreparedRag:
        """한 턴의 컨텍스트를 확정한다 — 라우팅·검색어·근거·캐시히트.

        쓰기부터 한다: 대화(신규면)와 턴(user + assistant 자리표시)을 먼저 commit해 어디서 실패해도
        질문이 남게 한다. 재호출하면 턴이 중복되므로 한 요청에 한 번만 부를 것.
        캐시 히트면 hit_count도 오르고, LLM을 최대 2회(분류·재작성) 부른다.
        자리표시는 status='generating'이라 load_recent_messages가 이력에서 자동으로 뺀다.
        """
        conversation, attachment_dicts, new_attachment_dicts = await self._resolve_conversation(
            conversation_id, attachments
        )
        assistant_message_id = await self._open_turn(conversation.id, query, new_attachment_dicts)

        # RETRY가 질의를 치환하면 실패 마감이 치환된 질문을 남겨야 한다(_fail_turn 참조).
        self._resolved_query = query
        try:
            return await self._route(
                conversation, assistant_message_id, query, attachment_dicts, new_attachment_dicts, domain_hint
            )
        except Exception:
            # 자리표시를 즉시 failed로 닫는다 — 안 하면 스테일 스윕이 회수할 때까지 generating으로 남는다.
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
        messages = await load_recent_messages(self.session, self.tenant_id, conversation.id)
        display_query: str | None = None  # RETRY 재실행에서만 채워짐 — _routed가 늦은 바인딩으로 읽는다

        def _routed(route: str, block_reason: str | None = None) -> PreparedRag:
            # blocked/other 공용 — 검색·인용 없이 라우팅 결과만 담는다.
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

        # 분류기에도 최근 문답을 준다 — 반박 발화("아깐 2년이라매")는 직전 답과 대조해야 회상과 갈린다.
        # 예산은 condense와 같은 눈금 — 둘 다 참조 해소용이라 생성용 예산을 실을 이유가 없다.
        decision = await classify_and_guard(
            self._aux_llm,
            query,
            has_attachments=bool(attachment_dicts),
            domain_hint=domain_hint,
            prior_turns=build_prior_turns(messages, settings.condense_history_budget_tokens),
        )
        if not decision.safe:
            logger.warning(
                '입력 가드 차단 (tenant=%s, conversation=%s): %s', self.tenant_id, conversation.id, decision.reason
            )
            return _routed("blocked", block_reason=decision.reason)

        # 분류기는 "재요청 발화"만 인식하고, 무엇을 다시 할지는 직전 턴 상태로 결정론 해소한다.
        # RETRY·ATTACHMENT는 여기서 소멸하는 전이 인텐트 — 이후 파이프라인·저장 계층은 모른다.
        precomputed_standalone: str | None = None
        if decision.intent == "RETRY":
            pair = last_unanswered_turn(messages)
            if pair is None:
                # 되돌릴 턴 없음(직전이 done이거나 이력 없음) — 회상·재설명은 OTHER가 담당
                return _routed("other")
            prev_user, prev_assistant = pair  # prev_user = 체인 머리(실질 질문 — "다시" 연타여도)
            display_query = query  # 저장·화면용 — 사용자가 친 "다시" 그대로
            query = prev_user.content  # 검색·프롬프트용 — 답을 못 받은 실질 질문
            self._resolved_query = query  # 이 턴도 실패하면 이 값이 백필돼야 체인이 이어진다
            if prev_assistant.intent == "OTHER":
                return _routed("other")  # 원래 OTHER였던 턴(대화 요약 등)은 원래 경로로 재실행
            # 재시도 = 원본 검색 재현 — 그 턴의 재작성 결과를 재사용한다.
            # 원 턴이 ATTACHMENT였어도 그 사실은 저장에 안 남아 일반 knowledge로 재실행된다(의도된 수용).
            precomputed_standalone = prev_user.standalone_query or prev_user.content
        elif decision.intent == "ATTACHMENT":
            if not attachment_dicts:
                return _routed("other")  # 가리킬 첨부가 없다 — OTHER가 되묻기 유도
            return self._prepare_attachment_only(
                conversation.id, assistant_message_id, query, messages, attachment_dicts, domain_hint
            )
        elif decision.intent == "OTHER":
            return _routed("other")

        return await self._prepare_knowledge(
            conversation.id,
            assistant_message_id,
            query,
            messages,
            attachment_dicts,
            new_attachment_dicts,
            domain_hint,
            display_query=display_query,
            precomputed_standalone=precomputed_standalone,
        )

    def _prepare_attachment_only(
        self,
        conversation_id: int,
        assistant_message_id: int,
        query: str,
        messages: list[Message],
        attachment_dicts: list[dict],
        domain_hint: str | None,
    ) -> PreparedRag:
        """ATTACHMENT 경로 — 검색·재작성을 생략하고 첨부만 근거로 생성한다.

        retrieval은 빈 결과다 — route=knowledge ⟺ retrieval 존재 불변식 때문. 첨부가 있어 캐시는 타지 않는다.
        """
        return PreparedRag(
            conversation_id=conversation_id,
            assistant_message_id=assistant_message_id,
            original_query=query,
            standalone_query=query,  # condense 생략 — 검색을 안 하므로 재작성이 무의미
            prior_turns=build_prior_turns(messages, settings.history_budget_tokens),
            retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
            sources=[],
            source_doc_ids=[],
            attachments=attachment_dicts,
            domain_hint=domain_hint,
            attachment_grounded=True,
        )

    async def _open_turn(self, conversation_id: int, query: str, new_attachment_dicts: list[dict]) -> int:
        """user 메시지 + assistant 자리표시를 커밋하고 자리표시 id를 반환한다.

        query는 사용자가 친 원문 그대로다(RETRY 치환은 이보다 뒤). standalone_query는 finalize_turn이 백필한다.
        이 commit이 신규 대화 INSERT도 함께 확정한다 — 빈 대화가 남는 실패 창을 없앤다.
        """
        assistant = await add_pending_turn(
            self.session,
            self.tenant_id,
            conversation_id,
            query,
            attachments=new_attachment_dicts or None,
            user_id=self.user_id,
        )
        await self.session.flush()  # 자리표시 id 확보 — 이후 전 경로가 이 id로 UPDATE한다
        await self.session.commit()  # 여기서부터 질문은 durable
        return assistant.id

    async def _fail_turn(self, assistant_message_id: int, resolved_query: str) -> None:
        """prepare() 실패 시 자리표시를 failed로 닫는다. 정리 실패는 삼킨다 — 원 예외를 가리지 않게.

        resolved_query를 짝 user 행의 standalone_query에 백필한다 — 실패 턴도 RETRY 대상이라,
        비우면 "다시" 연타 시 두 번째부터 질문 슬롯이 "다시"로 퇴화한다.
        """
        try:
            await finalize_turn(
                self.session,
                self.tenant_id,
                assistant_message_id,
                '',
                [],
                status=TurnStatus.FAILED,
                standalone_query=resolved_query,
            )
            await self.session.commit()
        except Exception:
            logger.exception('자리표시 failed 마감 실패 (message_id=%s)', assistant_message_id)

    async def _resolve_conversation(
        self,
        conversation_id: int | None,
        attachments: list[QueryAttachment] | None,
    ) -> tuple[Conversation, list[dict], list[dict]]:
        """대화를 확보하고 (대화, 주입용 첨부, 이번 턴 신규 첨부)를 반환한다.

        커밋하지 않는다 — _open_turn의 commit이 대화와 첫 턴을 한 트랜잭션으로 확정한다.
        주입용과 신규분은 별개다: max_attachments가 주입만 제한하므로 주입용이 비어도 신규분은 있을 수 있다.
        """
        conversation = await ensure_conversation(self.session, self.tenant_id, conversation_id, user_id=self.user_id)

        new_attachment_dicts = [{'filename': a.filename, 'text': a.text} for a in (attachments or [])]
        # 컨텍스트 예산 때문에 최신 max_attachments개만 주입한다 — 저장은 전부 유지된다.
        history = await self._load_history_attachments(conversation.id, settings.max_attachments)
        all_attachments = history + new_attachment_dicts
        # list[-0:]는 '전체'라 0을 그대로 슬라이스하면 정반대 동작이 된다.
        attachment_dicts = all_attachments[-settings.max_attachments :] if settings.max_attachments > 0 else []
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

        캐시가 히트하면 hit_count를 UPDATE한다(쓰기 부수효과).
        display_query/precomputed_standalone은 RETRY 재실행 전용.
        """
        # 멀티쿼리는 멀티턴 전용 — 첫 줄만 standalone으로 저장·캐시에 쓰이고 변형은 검색 전용.
        # 단일턴 확장은 변형의 RRF 희석으로 Hit@1을 깎아 막아 둔다.
        expanded: list[str] = []
        if precomputed_standalone is not None:
            # 다시 재작성하면 취소 턴이 낀 이력 때문에 원래와 다른 검색어가 나올 수 있다.
            standalone_query = precomputed_standalone
        else:
            history_for_condense = trim_messages_for_condense(messages, settings.condense_history_budget_tokens)
            if settings.condense_multi_query_enabled and messages:
                queries = await condense_to_queries(self._aux_llm, query, history_for_condense)
                standalone_query, expanded = queries[0], queries[1:]
            else:
                standalone_query = await condense_query(self._aux_llm, query, history_for_condense)

        retrieval = await retrieve(self.session, self.tenant_id, standalone_query, expanded_queries=expanded)

        # 순서가 곧 인용 번호 — 프롬프트 [번호]·꼬리 제약·꼬리 파서가 이 순서를 공유한다.
        sources = [] if retrieval.no_evidence else sources_from_chunks(retrieval.chunks)

        source_doc_ids = _source_doc_ids(retrieval.chunks)
        faq_versions = await cache.snapshot_faq_versions(self.session, self.tenant_id, source_doc_ids)
        prior_turns = build_prior_turns(messages, settings.history_budget_tokens)

        # 첨부가 있으면 캐시를 우회한다. 주입용·신규분 둘 다 봐야 한다 — max_attachments<=0이면 주입용이 비어
        # 신규 첨부 턴에 캐시 답변이 재생된다.
        if not retrieval.no_evidence and not attachment_dicts and not new_attachment_dicts:
            semantic_hit = await cache.get_semantic(
                self.session,
                self.tenant_id,
                standalone_query,
                source_doc_ids,
                query_embedding=retrieval.query_embedding,
                llm=self._aux_llm,
                # RETRY 턴이면 query는 이미 직전 실질 질문으로 치환돼 있다("다시"가 아니다).
                original_query=query,
            )
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
            query_embedding=retrieval.query_embedding,
            display_query=display_query,
        )

    async def _load_history_attachments(self, conversation_id: int, limit: int) -> list[dict]:
        """대화에 저장된 첨부를 시간순으로 모은다 — 매 턴 <첨부 문서> 근거 블록으로 재주입하기 위해.

        주입은 최신 limit개만 쓰므로 조회도 최신 limit행으로 제한한다(행당 첨부 1개 이상).
        차단 턴의 첨부는 뺀다 — 이후 턴 프롬프트로 재진입하면 차단이 반쪽이 된다.
        user 행은 항상 done이라 그 턴의 assistant(blocked) 존재로 판별한다. failed/cancelled 턴 첨부는 유지.
        """
        if limit <= 0:
            return []
        assistant = aliased(Message)
        blocked_turn = (
            select(assistant.id)
            .where(assistant.tenant_id == self.tenant_id)  # 서브쿼리에도 tenant 격리
            .where(assistant.question_message_id == Message.id)
            .where(assistant.status == 'blocked')
            .exists()
        )
        rows = (
            (
                await self.session.execute(
                    select(Message.attachments)
                    .where(Message.tenant_id == self.tenant_id)
                    .where(Message.conversation_id == conversation_id)
                    .where(Message.attachments.is_not(None))
                    .where(~blocked_turn)
                    .order_by(Message.id.desc())
                    .limit(limit)
                )
            )
            .scalars()
            .all()[::-1]
        )  # 최신 limit행 → 시간순 복원
        # JSON null 행이 있을 수 있다 — IS NOT NULL은 SQL NULL만 거른다.
        return [attachment for row in rows if row for attachment in row]

    def _set_finish_reason(self, reason: str) -> None:
        """astream 콜백 — 값 보관만. 기록은 streaming.py가 한다."""
        self.last_finish_reason = reason

    def _set_usage(self, usage) -> None:
        """astream 콜백 — 값 보관만. 기록은 streaming.py가 한다."""
        self.last_usage = usage

    async def generate(self, prepared: PreparedRag) -> AsyncIterator[str]:
        """답변 토큰을 생성한다. 갈래는 셋이다.

        1. 확정 답변(resolved_answer)이 있으면 LLM 없이 그대로 반환.
        2. OTHER는 검색 없이 이전 대화를 실어 제약된 프롬프트로 생성.
        3. 그 외는 KNOWLEDGE — 여기 오면 route=knowledge라 불변식상 retrieval이 반드시 있다.
        """
        # 확정 답변은 쪼개지 않는다 — 소비처(immediate_stream)가 단일 delta로 보낸다.
        answer = prepared.resolved_answer
        if answer is not None:
            yield answer
            return

        self.last_finish_reason = None  # 호출마다 리셋 — 이전 호출 값 누출 방지
        self.last_usage = None

        if prepared.route == "other":
            try:
                user_msg = build_other_user_message(prepared.original_query, prepared.prior_turns)
                async for token in self._llm.astream(
                    build_chat_prompt(build_other_system_prompt(prepared.domain_hint), user_msg),
                    on_finish_reason=self._set_finish_reason,
                    on_usage=self._set_usage,
                ):
                    yield token
            except Exception:
                logger.exception('LLM error(other gen)')
                yield SMALLTALK_ANSWER  # 생성 실패 시 폴백
            return

        # "질문:" 자리는 원 질문, 검색에 쓴 재작성 질의는 참고로 병기한다.
        prompt = build_knowledge_generation_prompt(
            prepared.original_query,
            prepared.retrieval.chunks,
            standalone_query=prepared.standalone_query,
            prior_turns=prepared.prior_turns,
            attachments=prepared.attachments,
            domain_hint=prepared.domain_hint,
        )

        # 출처 꼬리 강제 — 의도(CitationTail)만 넘기고 문법은 공급자 구현(rag/llm)이 안다.
        # 꼬리 파서의 번호 범위 검증은 제약과 별개로 항상 돈다.
        async for token in self._llm.astream(
            prompt,
            citations=build_citation_tail(*prepared.citation_candidates),
            on_finish_reason=self._set_finish_reason,
            on_usage=self._set_usage,
        ):
            yield token

    async def finalize(
        self,
        prepared: PreparedRag,
        answer: str,
        citations: list[SourceCitation],
        status: TurnStatus = TurnStatus.DONE,
        latency_ms: int | None = None,
    ) -> None:
        """자리표시를 최종 결과로 UPDATE한다 — 모든 경로 공용. commit은 호출자가 한다.

        생성 경로는 백그라운드 태스크가 자기 세션으로 만든 RagService에서 부른다.
        실패면 status='failed', answer=''로 호출. citations는 실제 인용된 출처만(호출자가 확정)이며
        done일 때만 저장된다. 캐시 적재는 하지 않는다 — 호출자가 done 전송 뒤 maybe_cache를 부른다.
        """
        source_dicts = [] if status != TurnStatus.DONE else [c.model_dump() for c in citations]
        await finalize_turn(
            self.session,
            self.tenant_id,
            prepared.assistant_message_id,
            answer,
            source_dicts,
            status=status,
            latency_ms=latency_ms,
            # 문서 파일명과 FAQ id는 컬럼을 가른다 — 섞으면 집계마다 걸러야 한다.
            # faq_id 없는 옛 캐시의 FAQ 인용은 cited_docs에 'FAQ'로 남는다.
            cited_docs=[c.filename for c in citations if c.faq_id is None] if status == TurnStatus.DONE else [],
            cited_faq_ids=[c.faq_id for c in citations if c.faq_id is not None] if status == TurnStatus.DONE else [],
            intent=prepared.intent_label,
            # 짝 user 행의 standalone_query까지 여기서 확정한다
            standalone_query=prepared.standalone_query,
            cache_kind=prepared.cache_kind,  # 히트율 집계용
            # 이력 격리(load_recent_messages)와 차단 집계가 이 값에 의존한다
            block_reason=prepared.block_reason,
        )

    async def maybe_cache(self, prepared: PreparedRag, answer: str, citations: list[SourceCitation]) -> None:
        """근거 있는 신규 LLM 응답만 semantic 캐시에 적재한다. done 전송 뒤에 부른다 — 턴 완결의 조건이 아니다.

        근거 없는 답변은 문서가 추가되면 바뀌어야 하므로 제외한다. 캐시엔 인용만 저장하고(히트 재생 시
        그대로 citations가 된다), 무효화 키(source_doc_ids)는 검색 근거 전체다 — 정확성의 기준은 검색된 문서 집합이다.
        """
        # 인용이 0이면 근거 없이 나온 답이다 — 거절 문구가 아니라 실인용 개수로 판정한다.
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
            query_embedding=prepared.query_embedding,
            # display_query가 아니다 — RETRY 턴이면 "다시"가 아니라 실질 질문이 저장돼야 판정이 의미를 갖는다.
            original_query=prepared.original_query,
        )


def _source_doc_ids(chunks: list[RetrievedChunk]) -> list[int]:
    """검색 청크의 출처 id 목록(중복 제거) — semantic 캐시의 집합 비교·무효화 키.

    FAQ는 문서 id와 충돌하지 않게 음수(-faq_id) — cache.invalidate_source가 같은 키로 지운다.
    """
    ids = {chunk.document_id for chunk in chunks if chunk.document_id is not None}
    ids |= {-chunk.faq_id for chunk in chunks if chunk.faq_id is not None}
    return list(ids)
