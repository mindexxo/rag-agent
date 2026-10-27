"""테넌트 테이블의 SQLAlchemy ORM 매핑. DDL·인덱스의 권위는 schema.sql이다.

청크는 PG에 없다 — 본문·메타·벡터는 OpenSearch에만 있다(매핑은 rag/os_client.py).

테넌트 격리: RLS를 쓰지 않는다. 격리의 유일한 방어선은 쿼리의 WHERE 절이다.
1. 멀티테넌트 테이블을 읽는 모든 쿼리에 `.where(Model.tenant_id == tenant_id)`를 직접 쓴다
   (join·컬럼 지정·집계 쿼리가 많아 래퍼 유틸은 절반도 못 덮는다).
2. UPDATE/DELETE는 대상 id를 tenant 스코프 조회로 먼저 확정한 뒤 그 id로 실행한다.
3. `messages.tenant_id`는 비정규화 컬럼이라 DB가 부모와의 일치를 보장하지 않는다 — 삽입 시 부모 값을 넣는다.
   검색 청크의 격리는 OpenSearch 질의마다 `tenant_filter()`가 건다(rag/os_search.py).
4. 누락 검출은 통합 테스트가 한다(tests/test_tenant_isolation.py·tests/test_integration_isolation.py).
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, TypedDict

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """시각 컬럼을 DDL(TIMESTAMPTZ)에 맞춰 tz-aware로 매핑한다.

    SQLAlchemy 기본(TIMESTAMP WITHOUT TIME ZONE)이면 asyncpg가 aware 값 바인딩을 거부하고,
    naive 우회는 로컬 벽시계를 UTC로 저장해 조용히 어긋난다.
    """

    type_annotation_map = {datetime: DateTime(timezone=True)}


class Folder(Base):
    """1단 폴더 — 검색 참조 제어 전용 그룹(트리 없음)."""

    __tablename__ = "folders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    name: Mapped[str]  # 테넌트 내 유일
    description: Mapped[str | None]  # "언제 이 폴더를 참조하나" — 리랭커 입력·프롬프트에 쓴다
    is_searchable: Mapped[bool] = mapped_column(default=True, server_default="true")
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (UniqueConstraint("tenant_id", "name"),)


# Document.status 6값 중 색인·supersede·DROP 판정 대상인 "살아 있는" 넷의 정의점.
# deleted·failed는 사용자에게 없는 문서다. rag/documents.py와 rag/outbox.py가 같은 집합을 써야 한다.
ALIVE_DOCUMENT_STATUSES = ('pending', 'parsing', 'embedding', 'ready')


class Document(Base):
    """업로드된 원본 문서. 같은 filename 재업로드는 내용과 무관하게 새 version이다."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    filename: Mapped[str]  # 테넌트 내 논리적 문서명 (재업로드 식별 기준)
    mime: Mapped[str]
    blob_path: Mapped[str]  # 웹 업로드는 UUID 이름, CLI는 원본 경로 그대로
    version: Mapped[int] = mapped_column(default=1, server_default="1")  # 같은 filename 내 일련번호
    is_active: Mapped[bool] = mapped_column(
        default=False, server_default="false"
    )  # 검색 대상 버전 (filename당 하나만 true)
    status: Mapped[str] = mapped_column(
        default="pending", server_default="pending"
    )  # 문서 처리 6값 — Message.status(TurnStatus)와 별개 어휘. 살아 있는 넷은 ALIVE_DOCUMENT_STATUSES
    status_reason: Mapped[str | None]  # 실패/삭제 사유
    char_count: Mapped[int | None]
    uploaded_by: Mapped[str | None]
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())
    indexed_at: Mapped[datetime | None]
    description: Mapped[str | None]  # xlsx 표 설명 — 청크에 병합돼 검색을 보강
    folder_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("folders.id", ondelete="SET NULL")
    )  # NULL=미분류
    is_searchable: Mapped[bool] = mapped_column(
        default=True, server_default="true"
    )  # 문서 단위 참조 on/off (버전 정책 is_active와 별개)

    __table_args__ = (UniqueConstraint("tenant_id", "filename", "version"),)


class FaqCategory(Base):
    """FAQ 1단 관리 분류. 폴더와 달리 검색에 영향이 없다(캐시 무효화·색인 fan-out 없음).
    소속 FAQ가 있으면 삭제할 수 없다(라우터 409)."""

    __tablename__ = "faq_categories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    name: Mapped[str]  # 테넌트 내 유일. 길이 상한은 앱(FAQ_CATEGORY_NAME_MAX)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (UniqueConstraint("tenant_id", "name"),)


class Faq(Base):
    """FAQ 항목. 검색 편입은 색인에 항목당 청크 1개로 한다(outbox INDEX_FAQ → os_index.index_faq_chunks)."""

    __tablename__ = "faqs"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "question"
        ),  # 대표 질문 중복의 최후 방어(라우터 409를 지나친 경쟁). 유사 질문은 JSONB라 앱에서만 검사
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    question: Mapped[str]  # 대표 질문
    variants: Mapped[Any] = mapped_column(JSONB, default=list)  # 유사 질문(구어체 변형) 배열
    answer: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")  # 항목 단위 검색 on/off
    # NULL=미분류. cascade 없음 — 카테고리 삭제는 소속 FAQ가 0일 때만 라우터가 허용한다
    category_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("faq_categories.id"))
    created_by: Mapped[str | None]  # X-User-Id. 미전송이면 NULL
    updated_by: Mapped[str | None]  # 모든 PATCH가 덮는다 — 헤더 없으면 NULL로
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class SearchIndexOutbox(Base):
    """색인 작업 트랜잭셔널 outbox. 어휘·처리 규약의 정의점은 rag/outbox.py.

    행은 PG 변경과 같은 트랜잭션에 쓴다 — 커밋되면 색인 할 일이 반드시 남고, 롤백되면 함께 사라진다.
    행은 지우지 않는다(done/failed 행이 이력). 연산은 전부 멱등이라 at-least-once 재시도가 안전하다.
    """

    __tablename__ = "search_index_outbox"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    op: Mapped[str]  # 어휘의 정의점은 rag/outbox.py
    payload: Mapped[Any] = mapped_column(JSONB, default=dict)  # id만 — 처리 시점에 PG 최신값을 읽는다
    status: Mapped[str] = mapped_column(default='pending', server_default='pending')  # pending|done|failed
    attempts: Mapped[int] = mapped_column(default=0, server_default="0")
    last_error: Mapped[str | None]
    next_attempt_at: Mapped[datetime | None]  # 일시 실패의 다음 시도 시각 — NULL이면 지금
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class AnswerCache(Base):
    """LLM 응답 영속 캐시. 의미 임베딩 매칭 + 문서 단위 무효화."""

    __tablename__ = "answer_cache"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    cache_key: Mapped[str]  # standalone query 정규화 해시 — 조회 키가 아니라 upsert 충돌 대상
    query_text: Mapped[str]  # 저장된 standalone query 원문
    original_query: Mapped[
        str | None
    ]  # 사용자가 친 원문 — 재사용 판정기 입력 전용. NULL 행은 판정기가 '(없음)'으로 받는다
    query_embedding: Mapped[Any] = mapped_column(
        Vector(1024)
    )  # 기준은 query_text다 (original_query는 임베딩·cache_key에 안 섞는다)
    answer: Mapped[str]
    sources: Mapped[Any] = mapped_column(JSONB)  # 인용 청크 메타 [{doc_id, chunk_id, page, ...}]
    source_doc_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))  # 무효화 키
    model: Mapped[str]
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_hit_at: Mapped[datetime] = mapped_column(server_default=func.now())
    hit_count: Mapped[int] = mapped_column(default=0, server_default="0")

    __table_args__ = (UniqueConstraint("tenant_id", "cache_key"),)


class Conversation(Base):
    """멀티턴 대화 세션 (한 상담사-한 통화 단위)."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    title: Mapped[str | None]
    created_by: Mapped[str | None]  # 미전송 시 'test-user'
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    last_used_at: Mapped[datetime] = mapped_column(server_default=func.now())  # 목록 정렬용
    deleted_at: Mapped[datetime | None]  # 소프트 삭제 — NULL=활성. 전 조회 경로가 IS NULL로 거른다


class TurnStatus(StrEnum):
    """messages.status 값 어휘의 정의점 — Document.status 6값과 별개 어휘다.

    str 서브클래스라 문자열 비교·직렬화·DB 저장과 호환된다. 여기는 어휘만 두고,
    파생 의미(이력 격리·미응답·종결 집합)는 rag/turn_state.py가 소유한다.
    밖의 값은 SQLAlchemy Enum이 아니라 쓰기 지점(turn_state.finalize_turn)이 거부한다.
    """

    GENERATING = 'generating'  # assistant 자리표시 — 유일한 비종결 상태
    DONE = 'done'  # 정상 종료. user 메시지는 항상 이 값
    CANCELLED = 'cancelled'  # 사용자 취소
    FAILED = 'failed'  # 생성 실패 또는 고착 회수
    BLOCKED = 'blocked'  # 입력 가드 차단


class Attachment(TypedDict):
    """채팅 첨부 한 건 — Message.attachments JSONB에 이 모양의 목록으로 저장된다."""

    filename: str
    text: str


class Message(Base):
    """대화 내 한 턴. role로 user/assistant 구분."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("conversations.id")
    )  # CASCADE 없음 — 삭제는 소프트(deleted_at), 실수 하드 DELETE에도 이력 보존
    tenant_id: Mapped[str]
    role: Mapped[str]  # 'user' | 'assistant'
    content: Mapped[str]  # user면 원 질문, assistant면 최종 답변
    standalone_query: Mapped[str | None]  # user 메시지의 condense 결과
    sources: Mapped[Any | None] = mapped_column(
        JSONB
    )  # assistant: 실제 인용된 출처(done일 때만 — 그 외 빈 목록, user는 NULL)
    attachments: Mapped[Any | None] = mapped_column(
        JSONB(none_as_null=True)
    )  # user: 첨부 목록(list[Attachment]). none_as_null — IS NOT NULL 필터가 맞게 SQL NULL로 저장
    status: Mapped[str] = mapped_column(
        default="done", server_default="done"
    )  # 값 어휘는 TurnStatus (Document.status와 별개)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # 운영 지표용 — 지표는 소급할 수 없어 화면 반영 전부터 기록한다
    user_id: Mapped[str | None]  # user: 질문한 상담원 (X-User-Id)
    latency_ms: Mapped[int | None]  # assistant: 생성 소요(ms). 캐시/즉시 경로 포함
    cache_kind: Mapped[str | None]  # assistant: 'semantic'=캐시 재생 답변, NULL=신규 생성
    cited_docs: Mapped[Any | None] = mapped_column(
        JSONB
    )  # assistant: 실인용 **문서** 파일명 배열 — 문서 지표는 이 컬럼만 집계. FAQ는 cited_faq_ids로(옛 행의 'FAQ' 원소는 그대로)
    cited_faq_ids: Mapped[Any | None] = mapped_column(
        JSONB
    )  # assistant: 실인용 FAQ id 배열 — FAQ 항목별 ref_count·근거없음 판정이 읽는다
    intent: Mapped[
        str | None
    ]  # assistant: 'KNOWLEDGE'|'OTHER'|'ATTACHMENT'(KB 지표 제외). 답변률 분모는 KNOWLEDGE만 (NULL=차단 턴·옛 행)
    question_message_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("messages.id", ondelete="SET NULL")
    )  # assistant: 답한 user 메시지
    block_reason: Mapped[str | None]  # assistant(status='blocked'): 입력 가드 차단 사유
    # 상담원의 답변 평가 — 값 검증은 Pydantic Literal
    feedback: Mapped[bool | None]  # True=👍 False=👎 NULL=미선택/취소
    feedback_tag: Mapped[str | None]  # 👎 사유 슬러그(wrong_info|wrong_source|outdated_doc|insufficient). 스킵하면 NULL
    feedback_text: Mapped[str | None]  # 👎 자유 서술 (500자)


class TenantQuota(Base):
    """테넌트별 동시 요청 상한 정책(오버라이드 전용 — 행이 없으면 config 기본값).
    실시간 in-flight 카운터는 Redis ZSET(rag/limiter.py)이다."""

    __tablename__ = "tenant_quotas"

    tenant_id: Mapped[str] = mapped_column(primary_key=True)
    concurrency_limit: Mapped[int] = mapped_column(
        default=5, server_default="5"
    )  # 테넌트 동시 in-flight — config 기본값과 같게 유지
    user_concurrency: Mapped[int] = mapped_column(
        default=1, server_default="1"
    )  # 사용자 동시 in-flight — config 기본값과 같게 유지
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())
