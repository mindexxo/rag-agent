"""KMS API 요청/응답 스키마"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from schemas.common import KstDatetime
from text_norm import normalize_filename


class SourceCitation(BaseModel):
    """응답에 인용된 문서 정보. 내부 retrieval 메타(score 등)는 제외.
    document_id는 원본 다운로드용 — 과거 저장분(메시지·캐시)엔 없으므로 옵셔널.

    FAQ 인용(#278): document_id=None·faq_id=항목 id, filename은 표시명 'FAQ: 질문'(질문 전체 —
    줄여 보이는 건 FE 몫, rag/turn/citation_labels.faq_display). #278 이전 저장분(메시지·캐시)은 FAQ 전체가
    {filename:'FAQ', faq_id 없음} 하나로 접혀 있다 — 그대로 재생되고 항목 집계에는 안 잡힌다."""

    document_id: int | None = None
    filename: str
    version: int
    faq_id: int | None = None


# 첨부 상한 (#22) — 여기가 단일 정의점. routers/attachments.py의 extract_attachment도 이 값을 쓴다.
# 이전엔 extract에만 상한이 있어, 클라이언트가 그 헬퍼를 건너뛰고 /kms/query에 직접
# attachments를 실으면 크기·개수 모두 무제한이었다 (컨텍스트 초과·DB 팽창 경로).
ATTACHMENT_MAX_TEXT_CHARS = 6000  # 개당 추출 텍스트 (~3-4페이지, ≈4K토큰)
ATTACHMENT_MAX_ITEMS = 1  # 요청당 첨부 개수 = 프롬프트 주입 개수(settings.max_attachments)와 동일.
# 주입되지 않을 첨부를 받아 저장만 하는 건 의미가 없다.
# max_attachments를 바꾸면 이 값도 함께 고칠 것 (두 곳이 짝)
ATTACHMENT_FILENAME_MAX = 255  # 프롬프트의 [첨부: 파일명] 라벨에 그대로 들어감


def strip_text(v: object) -> object:
    """길이를 **공백 제거 후** 잰다 — 라우터는 이 값을 그대로 저장하므로 검증 기준과 저장 기준이 같다(QueryAttachment와 같은 이유)."""
    return v.strip() if isinstance(v, str) else v


def _strip_variants(v: object) -> object:
    """항목을 strip하고 빈 항목과 중복(순서 유지, 첫 것만)을 뺀다 — 개수 상한(FAQ_VARIANTS_MAX)이 실제 항목만 세게.

    문자열이 아닌 항목은 손대지 않고 남겨 뒤의 타입 검증이 422를 내게 한다.
    """
    if not isinstance(v, list):
        return v
    out: list[object] = []
    seen: set[str] = set()
    for x in v:
        if isinstance(x, str):
            x = x.strip()
            if not x or x in seen:
                continue
            seen.add(x)
        out.append(x)
    return out


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


DOMAIN_HINT_MAX = 200  # 프롬프트 3곳에 매 요청 주입되므로 길이 제한 (클라이언트발 텍스트 상한)


class KmsQueryRequest(BaseModel):
    """KMS 질의 요청 바디.
    conversation_id 가 없으면 새 대화를 생성하고,
    있으면 해당 대화의 이전 메시지를 이용해 멀티턴 질의를 독립 질문으로 변환한다.
    attachments 가 있으면 캐시를 우회하고 첨부 텍스트를 생성 컨텍스트에 주입한다.
    """

    query: str = Field(min_length=1, max_length=4000)  # 공백 제거 후 1자 이상
    conversation_id: int | None = Field(default=None, gt=0)
    attachments: list[QueryAttachment] = Field(default=[], max_length=ATTACHMENT_MAX_ITEMS)
    # [임시] 테넌트 지식 범위 설명 — 인텐트 분류·생성 프롬프트에 역할 안내로 주입 (#1).
    # ICCS 연동 시 이 필드를 제거하고 서버 측 테넌트 정보 조회로 대체한다.
    domain_hint: str | None = Field(default=None, max_length=DOMAIN_HINT_MAX)

    _strip = field_validator('query', mode='before')(strip_text)


# 문서 설명(F1a 표 설명)의 길이 상한 (#194). FOLDER_DESC_MAX와 값이 같지만 근거가 다르다 —
# 폴더 설명은 리랭커 입력에 후보마다 붙어 지연이 누적되는 쪽이고, 이 값은 xlsx 청크 **본문**에
# 병합되어 임베딩·어휘 채널에 직접 들어간다. 같은 화면에서 두 설명을 쓰므로 값은 맞춰 둔다.
DOCUMENT_DESCRIPTION_MAX = 200


class DocumentUploadResponse(BaseModel):
    """문서 업로드/조회 응답."""

    document_id: int
    filename: str
    version: int
    status: str
    # failed 사유(예: 엑셀 표 크기 초과)뿐 아니라 **ready인 문서의 경고**도 담는다 —
    # 이미지로 그려진 표·흐름도가 있어 그 내용이 색인되지 않은 경우다 (#137 결함 4).
    # FE가 failed일 때만 표시하면 그 경고가 사용자에게 도달하지 않는다.
    status_reason: str | None = None  # 화면 표시용
    is_active: bool
    folder_id: int | None = None
    is_searchable: bool = True
    uploaded_at: KstDatetime  # 등록일시 — 목록 기본 정렬 키 (해당 **버전**의 업로드 시각). KST 오프셋(#181)
    # 등록자. X-User-Id 미전송이면 null이고, 이 필드가 생기기 전에 올라온 문서도 전부 null이다
    # (소급해 채울 정보가 없다) — 화면은 빈 값 표시를 처리해야 한다.
    uploaded_by: str | None = None
    ref_count: int | None = None  # 답변 인용 누적 횟수 (목록 API에서만 집계 — filename 키)
    # 표 설명 (#194). xlsx만 값이 있다 — 다른 형식은 입력해도 청킹이 무시한다(rag/chunking.chunk_file).
    # 재업로드 화면이 기존 값을 미리 채우려면 돌려받아야 한다(계승 규칙이 있어서 더욱).
    description: str | None = None
    # 색인 완료 시각 (#194). ready 전에는 null — 업로드 직후 폴링 없이 "언제 반영됐나"를 아는 유일한 값.
    indexed_at: KstDatetime | None = None


class DocumentUploadMetadata(BaseModel):
    """업로드 데이터 (#165) — multipart의 `document-data` 파트에 JSON으로 싣는다.

    파트 하나를 `Content-Type: application/json`으로 보내는 꼴이며, 사내 ICCS가
    `@RequestPart("sr-data")`·`@RequestPart("post-data")`로 쓰는 구조와 같다.

    평면 Form 필드로는 표현할 수 없는 값이 있어서 도입했다: multipart는 값이 전부 문자열이라
    **null이 없고**, `int | None = Form(None)`에서는 빈 값과 미전송이 똑같이 None으로 들어온다.
    JSON으로 받으면 `folder_id`의 세 경우가 자연스럽게 갈리고, 판정도 PATCH와 같은
    `model_fields_set`을 쓴다(routers/documents.py의 update_document와 같은 방식):

      {"folder_id": 7}     → 7번 폴더로
      {"folder_id": null}  → 미분류로 (직전 버전의 폴더를 떼라)
      {}                   → 직전 버전에서 계승 (신규 문서면 미분류)

    extra='forbid' — 오타 난 필드를 조용히 삼키면 "보냈는데 안 먹는" 버그가 된다.
    """

    model_config = ConfigDict(extra='forbid')

    folder_id: int | None = Field(None, gt=0)  # null=미분류, 0은 거절(목록 필터의 0=미분류와 다르다)
    # F1a: 표 설명 (xlsx 검색 보강). xlsx는 **필수**다(#194) — 다만 "이번 요청에 보냈는가"가
    # 아니라 "계승까지 따진 최종값이 비는가"로 판정하므로 여기서 required로 만들지 않는다.
    # 그 판정은 `routers/documents.py`의 `upload_document`가 `rag/documents.py`의 `latest_alive`로 끝낸다.
    description: str | None = Field(None, max_length=DOCUMENT_DESCRIPTION_MAX)
    expect_version: int | None = None  # 낙관적 잠금 — 의미는 upload_document 참조


# 한 번에 다룰 수 있는 문서 수 — 일괄 변경(#166)·일괄 삭제(#174) 공통. 화면에서 한 페이지를
# 전체 선택해도 들어가는 크기면서, 사고로 수만 건이 한 요청에 실려 오는 것은 막는 선. 초과는 422.
BULK_MAX_ITEMS = 200

# 문서 목록 페이지 크기 (#176). 상한을 BULK_MAX_ITEMS에 맞춘 건 화면에서 **한 페이지를 전체
# 선택해 일괄 변경·삭제로 넘기는** 흐름 때문이다 — 페이지가 그 상한을 넘으면 전체선택이 422가 된다.
DOC_LIST_DEFAULT_LIMIT = 50
DOC_LIST_MAX_LIMIT = BULK_MAX_ITEMS

# 목록 상태 필터로 받을 수 있는 값 (rag/models.py Document.status의 어휘와 같아야 한다).
DOC_STATUSES = frozenset({'pending', 'ready', 'failed'})  # 목록 필터 허용값 — deleted는 목록에 없다


class DocumentListResponse(BaseModel):
    """문서 목록 (#176) — 필드 이름은 대화 목록(ConversationListResponse)과 맞춘다.

    has_more는 total에서 파생한다(offset + len(items) < total). limit+1 조회 트릭을 쓰지 않는
    것도 대화 목록과 같다 — total이 있으면 산술로 나오고, 마지막 부분 페이지에서도 정확하다.
    """

    items: list[DocumentUploadResponse]
    total: int = 0
    has_more: bool


class DocumentExistsResponse(BaseModel):
    """업로드 전 동일 파일명 존재 확인 응답 (FE가 대체 확인 창을 띄울지 판단)."""

    exists: bool
    document_id: int | None = None
    version: int | None = None  # 존재할 경우 현재 버전 (업로드하면 +1이 된다)
    status: str | None = None
    uploaded_at: KstDatetime | None = None


class DocumentUpdateRequest(BaseModel):
    """문서 속성 변경 (F2). 보낸 필드만 반영 — folder_id는 null 전송 시 미분류로 이동."""

    folder_id: int | None = Field(None, gt=0)  # null=미분류, 0은 거절(목록 필터의 0=미분류와 다르다)
    is_searchable: bool | None = None
    # 표 설명 수정 (#194). 전까지는 파일을 다시 올려야만 고칠 수 있었다(= 새 버전 + 재파싱).
    # folder_id와 달리 model_fields_set 3단 구분을 쓰지 않는다 — description은 None이 "미분류"
    # 같은 유효한 목표값이 아니라서 "미전송"과 "null 전송"이 똑같이 "안 바꿈"이면 충분하다.
    # 지우려면 빈 문자열을 보낸다(routers/folders.py의 FolderUpdateRequest와 같은 2-way).
    # **xlsx는 비울 수 없다** — 업로드에서 필수로 받아놓고 여기서 뚫리면 규칙이 샌다.
    description: str | None = Field(None, max_length=DOCUMENT_DESCRIPTION_MAX)


class DocumentBulkUpdateRequest(BaseModel):
    """문서 여러 건의 속성 변경 (#166) — 폴더 이동·참조 on/off.

    필드 의미는 DocumentUpdateRequest와 같다(folder_id는 null 전송 = 미분류, 미전송 = 변경 없음).
    대상이 하나라도 어긋나면 아무것도 바꾸지 않는다 — 판정은 routers/documents.py.
    """

    document_ids: list[int] = Field(default=[], max_length=BULK_MAX_ITEMS)
    folder_id: int | None = Field(None, gt=0)  # null=미분류, 0은 거절(목록 필터의 0=미분류와 다르다)
    is_searchable: bool | None = None


# FAQ 입력 상한 (#276, 사용자 결정 10/22). 질문+유사 질문+답변이 **청크 하나**로 임베딩되므로
# (rag/search/faq_indexing.py) 길어질수록 임베딩 입력이 잘리거나 검색 정확도가 떨어진다 — 상한의 이유.
# DB 컬럼은 TEXT 그대로다(저장소 관례: 길이는 앱, DB는 TEXT). FE 입력칸이 같은 값을 건다.
FAQ_QUESTION_MAX = 100
FAQ_VARIANT_MAX = 100  # 유사 질문 항목당 — 질문과 같은 성격이라 같은 값
FAQ_VARIANTS_MAX = 10  # FAQ 하나당 유사 질문 개수
FAQ_ANSWER_MAX = 500


class FaqResponse(BaseModel):
    """FAQ 항목 (F3 — 전용 저장, 항목 단위 CRUD). 관리 필드(#278)는 FE 요청 5번 그대로."""

    id: int
    question: str
    variants: list[str] = []  # 유사 질문 (구어체 변형)
    answer: str
    is_active: bool = True
    category_id: int | None = None  # None = 미분류 (관리용 분류 — 검색·생성에 영향 없음)
    created_at: KstDatetime
    updated_at: KstDatetime  # 어떤 UPDATE든 DB가 올린다(is_active 토글 포함) — updated_by도 같은 기준
    created_by: str | None = None  # X-User-Id. 미전송이면 null(문서 uploaded_by와 같은 규칙)
    updated_by: str | None = (
        None  # 마지막으로 손댄 사람 — 헤더 없으면 null로 덮는다(사용자 결정 5, ICCS 연동은 헤더 필수)
    )
    ref_count: int | None = (
        None  # 답변 인용 누적 횟수 — 목록 API에서만 집계(messages.cited_faq_ids). #278 이전 인용은 항목 기록이 없어 안 센다
    )


class FaqCreateRequest(BaseModel):
    question: str = Field(max_length=FAQ_QUESTION_MAX)
    variants: list[Annotated[str, Field(max_length=FAQ_VARIANT_MAX)]] = Field(default=[], max_length=FAQ_VARIANTS_MAX)
    answer: str = Field(max_length=FAQ_ANSWER_MAX)
    category_id: int | None = Field(None, gt=0)  # null=미분류, 0은 거절(목록 필터의 0=미분류와 다르다)

    _strip = field_validator('question', 'answer', mode='before')(strip_text)
    _strip_v = field_validator('variants', mode='before')(_strip_variants)


class FaqUpdateRequest(BaseModel):
    """보낸 필드만 반영. category_id는 문서 PATCH의 folder_id와 같은 3단 — 미전송=유지, null=미분류로 이동(model_fields_set으로 구분)."""

    question: str | None = Field(None, max_length=FAQ_QUESTION_MAX)
    variants: list[Annotated[str, Field(max_length=FAQ_VARIANT_MAX)]] | None = Field(None, max_length=FAQ_VARIANTS_MAX)
    answer: str | None = Field(None, max_length=FAQ_ANSWER_MAX)
    is_active: bool | None = None
    category_id: int | None = Field(None, gt=0)  # null=미분류, 0은 거절(목록 필터의 0=미분류와 다르다)

    _strip = field_validator('question', 'answer', mode='before')(strip_text)
    _strip_v = field_validator('variants', mode='before')(_strip_variants)


class FaqListResponse(BaseModel):
    """FAQ 목록 (#278) — 모양·has_more 산출은 문서 목록(DocumentListResponse)과 같다. 페이지 크기도 같은 상수(DOC_LIST_*)."""

    items: list[FaqResponse]
    total: int = 0
    has_more: bool


# FAQ 카테고리 (#278) — 폴더(F2)와 같은 1단 관리 분류. 다른 점: 검색에 영향 없음(캐시 무효화·fan-out 없음), 이름 상한 있음.
# 폴더 이름에는 상한이 없지만 FAQ 쪽은 #276부터 입력 상한을 두는 쪽이라 맞췄다(사용자 결정 4). DB 컬럼은 TEXT.
FAQ_CATEGORY_NAME_MAX = 50


class FaqCategoryInfo(BaseModel):
    id: int
    name: str
    faq_count: int = 0  # 소속 FAQ 수(활성·비활성 모두) — 삭제 가능 여부를 화면이 미리 알 수 있게


class FaqCategoryCreateRequest(BaseModel):
    name: str = Field(max_length=FAQ_CATEGORY_NAME_MAX)
    _strip = field_validator('name', mode='before')(strip_text)


class FaqCategoryUpdateRequest(BaseModel):
    name: str = Field(max_length=FAQ_CATEGORY_NAME_MAX)
    _strip = field_validator('name', mode='before')(strip_text)


FOLDER_DESC_MAX = 200  # 리랭커 입력에 매 후보마다 붙으므로 길이를 제한 (top_n개 누적 → 지연)
FOLDER_NAME_MAX = 50  # FAQ 카테고리 이름과 같은 상한


class FolderInfo(BaseModel):
    id: int
    name: str
    description: str | None = None  # "언제 이 폴더를 참조하나" 자연어 설명
    is_searchable: bool


class FolderCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=FOLDER_NAME_MAX)  # 공백 제거 후
    description: str | None = Field(None, max_length=FOLDER_DESC_MAX)
    _strip = field_validator('name', mode='before')(strip_text)


class FolderUpdateRequest(BaseModel):
    """보낸 필드만 반영."""

    name: str | None = Field(None, min_length=1, max_length=FOLDER_NAME_MAX)
    description: str | None = Field(None, max_length=FOLDER_DESC_MAX)
    is_searchable: bool | None = None
    _strip = field_validator('name', mode='before')(strip_text)
