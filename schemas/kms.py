"""KMS API 요청/응답 스키마"""
from typing import Literal
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    """응답에 인용된 문서 정보. 내부 retrieval 메타(score 등)는 제외.
    document_id는 원본 다운로드용 — 과거 저장분(메시지·캐시)엔 없으므로 옵셔널."""
    document_id: int | None = None
    filename: str
    version: int

class QueryAttachment(BaseModel):
    """채팅 첨부 문서 — 서버에 저장하지 않고 요청마다 동봉되는 추출 텍스트."""
    filename: str
    text: str

class KmsQueryRequest(BaseModel):
    """KMS 질의 요청 바디.
    conversation_id 가 없으면 새 대화를 생성하고,
    있으면 해당 대화의 이전 메시지를 이용해 멀티턴 질의를 독립 질문으로 변환한다.
    attachments 가 있으면 캐시를 우회하고 첨부 텍스트를 생성 컨텍스트에 주입한다.
    """
    query: str = Field(min_length=1, max_length=4000)   # 빈 질의 방지 + 상한 (P2 스키마 제약)
    conversation_id: int | None = Field(default=None, gt=0)
    attachments: list[QueryAttachment] = []

class KmsQueryResponse(BaseModel):
    """비스트리밍 응답 (?stream=false)
    conversation_id는 다음 턴에서 같은 대화 맥락을 이어가기 위한 식별자.
    sources: 중복 제거된 문서 단위 인용 목록. no_evidence면 빈 리스트.
    reason: ok | no_evidence
    """
    answer: str
    sources: list[SourceCitation]
    conversation_id: int
    reason: Literal["ok", "no_evidence", "blocked_output"]
    cached: bool = False
    cache_kind: Literal["exact", "semantic"] | None = None

class DocumentUploadResponse(BaseModel):
    """문서 업로드/조회 응답."""
    document_id: int
    filename: str
    version: int
    status: str
    status_reason: str | None = None   # failed 사유 (예: 150행 초과) — 화면 표시용
    is_active: bool
    folder_id: int | None = None
    is_searchable: bool = True
    ref_count: int | None = None  # 답변 인용 누적 횟수 (목록 API에서만 집계 — filename 키)


class DocumentUpdateRequest(BaseModel):
    """문서 속성 변경 (F2). 보낸 필드만 반영 — folder_id는 null 전송 시 미분류로 이동."""
    folder_id: int | None = None
    is_searchable: bool | None = None


class FaqResponse(BaseModel):
    """FAQ 항목 (F3 — 전용 저장, 항목 단위 CRUD)."""
    id: int
    question: str
    variants: list[str] = []   # 유사 질문 (구어체 변형)
    answer: str
    is_active: bool = True

class FaqCreateRequest(BaseModel):
    question: str
    variants: list[str] = []
    answer: str

class FaqUpdateRequest(BaseModel):
    """보낸 필드만 반영."""
    question: str | None = None
    variants: list[str] | None = None
    answer: str | None = None
    is_active: bool | None = None


class FolderInfo(BaseModel):
    id: int
    name: str
    is_searchable: bool

class FolderCreateRequest(BaseModel):
    name: str

class FolderUpdateRequest(BaseModel):
    """보낸 필드만 반영."""
    name: str | None = None
    is_searchable: bool | None = None
