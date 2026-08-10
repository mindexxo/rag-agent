"""KMS API 요청/응답 스키마"""
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field

class SourceCitation(BaseModel):
    """응답에 인용된 문서 정보. 내부 retrieval 메타(score 등)는 제외.
    document_id는 원본 다운로드용 — 과거 저장분(메시지·캐시)엔 없으므로 옵셔널."""
    document_id: int | None = None
    filename: str
    version: int

# 첨부 상한 (#22) — 여기가 단일 정의점. routers/documents.py의 extract 엔드포인트도 이 값을 쓴다.
# 이전엔 extract에만 상한이 있어, 클라이언트가 그 헬퍼를 건너뛰고 /kms/query에 직접
# attachments를 실으면 크기·개수 모두 무제한이었다 (컨텍스트 초과·DB 팽창 경로).
ATTACHMENT_MAX_TEXT_CHARS = 6000   # 개당 추출 텍스트 (~3-4페이지, ≈4K토큰)
ATTACHMENT_MAX_ITEMS = 3           # 요청당 첨부 개수 — 주입은 settings.max_attachments(2)개뿐이지만
                                   # 저장은 전량이라 저장 측 상한이 따로 필요하다.
                                   # 3 = 주입분 2 + 여유 1 (주입 상한을 크게 벗어나는 첨부는 어차피 안 쓰인다)
ATTACHMENT_FILENAME_MAX = 255      # 프롬프트의 [첨부: 파일명] 라벨에 그대로 들어감


class QueryAttachment(BaseModel):
    """채팅 첨부 문서 — 서버에 저장하지 않고 요청마다 동봉되는 추출 텍스트."""
    filename: str = Field(max_length=ATTACHMENT_FILENAME_MAX)
    text: str = Field(max_length=ATTACHMENT_MAX_TEXT_CHARS)

DOMAIN_HINT_MAX = 200   # 프롬프트 3곳에 매 요청 주입되므로 길이 제한 (클라이언트발 텍스트 상한)


class KmsQueryRequest(BaseModel):
    """KMS 질의 요청 바디.
    conversation_id 가 없으면 새 대화를 생성하고,
    있으면 해당 대화의 이전 메시지를 이용해 멀티턴 질의를 독립 질문으로 변환한다.
    attachments 가 있으면 캐시를 우회하고 첨부 텍스트를 생성 컨텍스트에 주입한다.
    """
    query: str = Field(min_length=1, max_length=4000)   # 빈 질의 방지 + 상한 (P2 스키마 제약)
    conversation_id: int | None = Field(default=None, gt=0)
    attachments: list[QueryAttachment] = Field(default=[], max_length=ATTACHMENT_MAX_ITEMS)
    # [임시] 테넌트 지식 범위 설명 — 인텐트 분류·생성 프롬프트에 역할 안내로 주입 (#1).
    # ICCS 연동 시 이 필드를 제거하고 서버 측 테넌트 정보 조회로 대체한다.
    domain_hint: str | None = Field(default=None, max_length=DOMAIN_HINT_MAX)

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
    cache_kind: Literal["semantic"] | None = None

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


class DocumentExistsResponse(BaseModel):
    """업로드 전 동일 파일명 존재 확인 응답 (FE가 대체 확인 창을 띄울지 판단)."""
    exists: bool
    document_id: int | None = None
    version: int | None = None       # 존재할 경우 현재 버전 (업로드하면 +1이 된다)
    status: str | None = None
    uploaded_at: datetime | None = None


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


FOLDER_DESC_MAX = 200   # 리랭커 입력에 매 후보마다 붙으므로 길이를 제한 (top_n개 누적 → 지연)


class FolderInfo(BaseModel):
    id: int
    name: str
    description: str | None = None   # "언제 이 폴더를 참조하나" 자연어 설명
    is_searchable: bool

class FolderCreateRequest(BaseModel):
    name: str
    description: str | None = Field(None, max_length=FOLDER_DESC_MAX)

class FolderUpdateRequest(BaseModel):
    """보낸 필드만 반영."""
    name: str | None = None
    description: str | None = Field(None, max_length=FOLDER_DESC_MAX)
    is_searchable: bool | None = None
