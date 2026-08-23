"""KMS API 요청/응답 스키마"""
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from text_norm import normalize_filename

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
ATTACHMENT_MAX_ITEMS = 1           # 요청당 첨부 개수 = 프롬프트 주입 개수(settings.max_attachments)와 동일.
                                   # 주입되지 않을 첨부를 받아 저장만 하는 건 의미가 없다.
                                   # max_attachments를 바꾸면 이 값도 함께 고칠 것 (두 곳이 짝)
ATTACHMENT_FILENAME_MAX = 255      # 프롬프트의 [첨부: 파일명] 라벨에 그대로 들어감


class QueryAttachment(BaseModel):
    """채팅 첨부 문서 — 서버에 저장하지 않고 요청마다 동봉되는 추출 텍스트."""
    filename: str = Field(max_length=ATTACHMENT_FILENAME_MAX)
    text: str = Field(max_length=ATTACHMENT_MAX_TEXT_CHARS)

    @field_validator('filename', mode='before')
    @classmethod
    def _nfc_filename(cls, v):
        """파일명 경계 정규화 (#34). 스키마에 두는 이유 — 이 모델에 도달하는 경로가 둘이다:
        /kms/attachments/extract의 응답 생성과 /kms/query 바디 파싱. 라우터마다 넣으면
        하나를 빠뜨리고, 새 경로가 생기면 또 빠뜨린다. `[첨부: 파일명]` 라벨이 프롬프트에
        들어가므로 문서 파일명과 같은 위험을 갖는다.
        text(본문)는 정규화하지 않는다 — 원문 보존.

        mode='before'인 이유: 기본값 'after'면 max_length가 **정규화 전** 길이로 검사된다.
        NFD 한글은 글자당 최대 3코드포인트라, NFC 기준 상한 이내인 정상 파일명이 분해형으로
        오면 거부될 수 있다 — 검증 기준과 저장 기준이 어긋나는, 이 이슈와 같은 유형의 비대칭.
        """
        return normalize_filename(v) if isinstance(v, str) else v

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
