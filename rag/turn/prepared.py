"""한 턴의 확정 컨텍스트 `PreparedRag` — prepare()가 만들고 generate·finalize·maybe_cache·SSE가 읽는다.

route ⟺ retrieval 불변식과 즉시/생성 경로 판정(resolved_answer)의 정의점.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from rag.models import Attachment, TurnStatus
from rag.search.retrieval_types import RetrievalResult
from rag.turn.citation_labels import CitationCandidates
from rag.turn.prompt_texts import BLOCKED_INPUT_ANSWER, NO_EVIDENCE_ANSWER
from schemas.kms import SourceCitation


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
    attachments: list[Attachment] = field(default_factory=list)  # 컨텍스트 주입용 — 히스토리 저장분 + 이번 턴 신규 합본
    domain_hint: str | None = None  # [임시] 테넌트 지식 범위 설명 — 생성 프롬프트 주입용, 저장 안 함
    assistant_message_id: int | None = (
        None  # 생성 경로: 자리표시 assistant 메시지 id (백그라운드 태스크가 UPDATE할 대상)
    )
    block_reason: str | None = None  # route='blocked'일 때 가드 판정 사유 — 저장·집계용
    faq_versions: dict[int, datetime] | None = None  # 근거 FAQ 스냅샷 — cache.save_answer 낙관적 검증 기준
    # 캐시 저장에 재사용할 검색 질의 벡터(근거는 cache.get_semantic). 캐시 히트 턴은 저장하지 않으므로 비운다.
    # repr=False — 긴 float 배열이 로그·트레이스백에 새지 않게.
    query_embedding: list[float] | None = field(default=None, repr=False)
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
    def citation_candidates(self) -> CitationCandidates:
        """인용 후보 (검색 출처, 첨부 파일명) — 꼬리 제약과 파서가 공유하는 단일 파생점.

        둘이 어긋나면 에러 없이 다른 문서로 오귀속된다. 순서 계약은 rag/turn/citation_labels.py가 정의점이다.
        """
        return CitationCandidates(self.sources, [a['filename'] for a in (self.attachments or [])])

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
