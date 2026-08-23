"""SQLAlchemy 2.0 declarative 모델 — 테넌트 테이블의 ORM 매핑.

- folders        : 1단 폴더 (검색 참조 제어 전용 그룹)
- documents      : 업로드된 원본 문서 (filename + version 단위)
- faqs           : FAQ 항목 (검색 편입은 chunks로)
- chunks         : 검색 단위 (dense 임베딩 보유 — dense-only, F99)
- answer_cache   : LLM 응답 영속 캐시 (semantic 매칭 + 문서 단위 무효화)
- conversations  : 멀티턴 대화 세션
- messages       : 대화 내 한 턴 (user/assistant)
- tenant_quotas  : 테넌트별 동시 요청 상한 정책

스키마(DDL)는 schema.sql이 권위. 이 파일은 ORM 매핑만 담당 — 인덱스는 schema.sql 참조.

── 테넌트 격리 규약 ────────────────────────────────────────
이 프로젝트는 **RLS를 사용하지 않는다.** 격리의 유일한 방어선은 쿼리의 WHERE 절이다.

1. 멀티테넌트 테이블을 읽는 모든 쿼리에 `.where(Model.tenant_id == tenant_id)`를 직접 쓴다.
   래퍼 유틸을 두지 않는 이유: 실제 쿼리 상당수가 join·컬럼 지정(`select(Chunk.id, distance)`)·
   집계(`select(func.count())`) 형태라 `select(Model)`을 감싸는 헬퍼가 절반도 덮지 못하고,
   반쯤 적용된 유틸은 "안 썼으니 격리가 빠졌나?"라는 오독을 만든다. 손 WHERE로 일관시킨다.
2. UPDATE/DELETE는 대상 id를 **tenant 스코프 조회로 먼저 확정**한 뒤 그 id로 실행한다
   (예: `delete(Chunk).where(Chunk.document_id.in_(<스코프된 doc_ids>))`).
3. `chunks.tenant_id`·`messages.tenant_id`는 필터 성능용 비정규화 컬럼이다. 부모와의 일치를
   DB가 보장하지 않으므로(FK는 부모 id에만 걸림) 삽입 시 부모의 tenant_id를 그대로 넣는다.
4. 누락 검출은 통합 테스트가 담당한다 — tests/test_tenant_isolation.py(ORM 읽기),
   tests/test_integration_isolation.py(검색 후보·대화·폴더). 표면이 늘면 여기에 케이스를 추가한다.
"""
from datetime import datetime
from typing import Any
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    BigInteger,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# sqlalchemy( 파이썬 ORM )
# ==========================================
# Mapped[T] -> 타입 힌트가 컬럼 타입
# Mapped[T | None] -> NULL 허용
# mapped_column(...) -> PK/FK/default 등 메타
# default=(파이썬), server_default=(DB DDL)


class Base(DeclarativeBase):
    pass


class Folder(Base):
    """1단 폴더 (F2) — 검색 참조 제어 전용 그룹. 트리 없음 (필요 시 parent_id 추가로 확장)."""
    __tablename__ = "folders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    name: Mapped[str]                                                               # 테넌트 내 유일
    description: Mapped[str | None]                                                 # "언제 이 폴더를 참조하나" 자연어 설명 (리랭커 입력·프롬프트에 사용)
    is_searchable: Mapped[bool] = mapped_column(default=True, server_default="true")  # 폴더 단위 참조 on/off
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    __table_args__ = (
        UniqueConstraint("tenant_id", "name"),
    )


class Document(Base):
    """업로드된 원본 문서. 같은 filename 재업로드 = 새 version (내용 무관, 2026-08-05)."""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]                                                            # 테넌트(고객사) 식별자, 격리 키
    filename: Mapped[str]                                                             # 테넌트 내 논리적 문서명 (재업로드 식별 기준)
    mime: Mapped[str]                                                                 # application/pdf | .../wordprocessingml.document
    blob_path: Mapped[str]                                                            # 원본 파일 저장 경로 (웹 업로드는 UUID 이름, CLI는 원본 경로 그대로)
    version: Mapped[int] = mapped_column(default=1, server_default="1")               # 같은 filename 내 일련번호
    is_active: Mapped[bool] = mapped_column(default=False, server_default="false")    # 검색 대상 여부 (filename당 단 하나만 true)
    status: Mapped[str] = mapped_column(default="pending", server_default="pending")  # pending|parsing|embedding|ready|failed|deleted
    status_reason: Mapped[str | None]                                                 # 실패/삭제 사유
    page_count: Mapped[int | None]                                                    # PDF 페이지 수
    char_count: Mapped[int | None]                                                    # 본문 글자 수
    uploaded_by: Mapped[str | None]                                                   # 업로더 식별자
    uploaded_at: Mapped[datetime] = mapped_column(server_default=func.now())          # 업로드 시각
    indexed_at: Mapped[datetime | None]                                               # 인덱싱 완료 시각
    description: Mapped[str | None]                                                    # F1a: 표 설명 (xlsx 검색 보강 — 청크에 병합)
    folder_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("folders.id", ondelete="SET NULL"))  # 폴더 소속 (NULL=미분류)
    is_searchable: Mapped[bool] = mapped_column(default=True, server_default="true")  # 문서 단위 참조 on/off (버전 정책 is_active와 별개)

    __table_args__ = (
        UniqueConstraint("tenant_id", "filename", "version"),
    )


class Faq(Base):
    """FAQ 항목 (F3 전용 저장). 검색 편입은 chunks에 항목당 청크 1개로 (faq_indexing)."""
    __tablename__ = "faqs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    question: Mapped[str]                                                          # 대표 질문
    variants: Mapped[Any] = mapped_column(JSONB, default=list)                     # 유사 질문(구어체 변형) 배열
    answer: Mapped[str]
    is_active: Mapped[bool] = mapped_column(default=True, server_default="true")   # 항목 단위 검색 on/off
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


class Chunk(Base):
    """검색 단위 (검색 인덱스). 출처는 document 또는 faq 정확히 하나 — DDL의 CHECK가 강제."""
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    document_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("documents.id", ondelete="CASCADE"))  # 문서 출처 (F3부터 nullable)
    faq_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("faqs.id", ondelete="CASCADE"))            # FAQ 출처 — 항목당 1청크 (부분 유니크)
    tenant_id: Mapped[str]                                                                                # 비정규화 (필터 성능)
    chunk_index: Mapped[int]                                                                              # 문서 내 청크 순서 (0부터)
    text: Mapped[str]                                                                                     # 청크 본문 원문 (prefix 없음 — 인용·프롬프트가 쓰는 값)
    token_count: Mapped[int | None]                                                                       # 토큰 수
    page: Mapped[int | None]                                                                              # 원 페이지 번호 (PDF만, DOCX는 NULL)
    heading_path: Mapped[list[str]] = mapped_column(ARRAY(Text), server_default="{}")                     # 헤딩 계층 경로 ["3. 배송지연", "3.2 지급기준"]
    meta: Mapped[dict] = mapped_column("metadata", JSONB, server_default="{}")                            # 자유 키-값 확장 영역 (DB 컬럼명은 metadata)
    dense: Mapped[Any] = mapped_column(Vector(1024))                                                      # BGE-M3 dense 임베딩 (1024차원)

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index"),
    )


class AnswerCache(Base):
    """LLM 응답 영속 캐시. semantic 임베딩 매칭 + 문서 단위 무효화."""
    __tablename__ = "answer_cache"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    cache_key: Mapped[str]                                                    # standalone query 정규화 해시 — 조회 키가 아니라 upsert 충돌 대상 (exact 계층 제거 #16)
    query_text: Mapped[str]                                                   # 저장된 standalone query 원문
    query_embedding: Mapped[Any] = mapped_column(Vector(1024))                # 의미 캐시(semantic match)용 임베딩
    answer: Mapped[str]                                                       # 저장된 LLM 답변
    sources: Mapped[Any] = mapped_column(JSONB)                               # 인용 청크 메타 [{doc_id, chunk_id, page, ...}]
    source_doc_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))      # 답변이 의존한 문서 ID들 (무효화 키)
    model: Mapped[str]                                                        # 사용된 LLM 모델명
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())   # 생성 시각
    last_hit_at: Mapped[datetime] = mapped_column(server_default=func.now())  # 마지막 히트 시각 (LRU/통계)
    hit_count: Mapped[int] = mapped_column(default=0, server_default="0")     # 히트 누적 수

    __table_args__ = (
        UniqueConstraint("tenant_id", "cache_key"),
    )


class Conversation(Base):
    """멀티턴 대화 세션 (한 상담사-한 통화 단위)."""
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    tenant_id: Mapped[str]
    title: Mapped[str | None]                                                  # 대화 제목 (옵셔널)
    created_by: Mapped[str | None]                                             # 생성한 사용자 (#10부터 항상 저장 — 미전송 시 'test-user')
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())    # 생성 시각
    last_used_at: Mapped[datetime] = mapped_column(server_default=func.now())  # 최근 사용 시각 (목록 정렬용)
    deleted_at: Mapped[datetime | None]    # 소프트 삭제 시각 (#10) — NULL=활성. 감사 목적 이력 보존, 전 조회 경로가 IS NULL 필터


class Message(Base):
    """대화 내 한 턴. role로 user/assistant 구분."""
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    conversation_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("conversations.id"))  # 부모 대화 FK. CASCADE 해제(#10) — 삭제는 소프트(deleted_at), 실수 하드 DELETE에도 이력 보존
    tenant_id: Mapped[str]
    role: Mapped[str]                                                        # 'user' | 'assistant'
    content: Mapped[str]                                                     # user면 원 질문, assistant면 최종 답변
    standalone_query: Mapped[str | None]                                     # user 메시지의 condense 결과 (assistant는 NULL)
    sources: Mapped[Any | None] = mapped_column(JSONB)                       # assistant 메시지의 인용 메타 (user는 NULL)
    attachments: Mapped[Any | None] = mapped_column(JSONB(none_as_null=True))  # 첨부한 턴의 user 메시지에 저장되는 추출 텍스트 [{filename, text}]. none_as_null: 파이썬 None을 JSON null이 아닌 SQL NULL로 (IS NOT NULL 필터가 정확히 동작하게)
    status: Mapped[str] = mapped_column(default="done", server_default="done")  # assistant 생성 상태: generating|done|failed|blocked|cancelled (user는 'done')
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())  # 생성 시각
    # ── 운영 지표 씨앗 (2026-07-18): 화면 반영 전이라도 기록은 지금부터 (지표는 소급 불가) ──
    user_id: Mapped[str | None]          # user 메시지: 질문한 상담원 (X-User-Id — 인증 전엔 'test-user')
    latency_ms: Mapped[int | None]       # assistant 메시지: 생성 소요(ms). 캐시/즉시 경로 포함
    cache_kind: Mapped[str | None]       # assistant 메시지: 'semantic'=캐시 재생 답변, NULL=신규 생성
    cited_docs: Mapped[Any | None] = mapped_column(JSONB)  # assistant: 실인용 파일명 배열 — 저장 시 확정 (지표는 이 컬럼만 집계)
    intent: Mapped[str | None]           # assistant: 라우팅 결과 'KNOWLEDGE'|'OTHER'|'ATTACHMENT'(#63 — 첨부만 근거, KB 지표에서 제외) — 답변률 분모는 KNOWLEDGE만 (NULL=차단 턴·도입 전 행)
    question_message_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("messages.id", ondelete="SET NULL"))  # assistant: 답한 user 메시지 (짝을 데이터로)
    block_reason: Mapped[str | None]     # assistant(status='blocked'): 입력 가드(classify_and_guard) 차단 사유. 출력 가드는 제거됨(#26)
    # ── 인간 피드백 1단계 (#8): 상담원의 답변 평가. 검증은 Pydantic Literal (태그 4종·별도 테이블은 과설계)
    feedback: Mapped[bool | None]        # assistant: True=👍 False=👎 NULL=미선택/취소
    feedback_tag: Mapped[str | None]     # assistant: 👎 사유 슬러그(wrong_info|wrong_source|outdated_doc|insufficient). 옵셔널 — 👎인데 스킵하면 NULL
    feedback_text: Mapped[str | None]    # assistant: 👎 자유 서술 (옵셔널, 500자) — 태그가 못 잡는 사유 발굴용. 고객 정보 미기입 안내는 FE 몫


class TenantQuota(Base):
    """테넌트별 동시 요청 상한 정책. 행이 없으면 config 기본값이 적용된다 (테이블은 오버라이드 전용).

    실시간 in-flight 카운터는 Redis ZSET(rag/limiter.py) — 이 테이블은 정책값만 보관한다.
    빈도 제한(RPM·일일 한도)·강제 차단·deep 모드 컬럼은 제거했다(#24): 정의만 있고 코드가
    읽지 않아, 값을 넣어도 아무 일이 일어나지 않는 채로 "동작한다"고 읽히는 상태였다.
    빈도 제한이 필요해지면 그때 설계와 함께 다시 도입한다.
    """
    __tablename__ = "tenant_quotas"

    tenant_id: Mapped[str] = mapped_column(primary_key=True)                                # 테넌트당 row 1개
    concurrency_limit: Mapped[int] = mapped_column(default=10, server_default="10")         # 동시 in-flight (테넌트) — config 기본값과 동일하게 유지
    user_concurrency: Mapped[int] = mapped_column(default=10, server_default="10")          # 동시 in-flight (사용자) — config 기본값과 동일하게 유지
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now())                 # 갱신 시각
