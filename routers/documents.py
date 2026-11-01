"""KMS 문서 관리 라우터 — 업로드·목록·변경·삭제·다운로드. 첨부 추출은 routers/attachments.py.

**이 파일이 정의점인 것은 HTTP 경계의 규칙뿐이다.** 무엇이 일어나는지는 아래를 보라:
업로드·supersede·failed 재사용은 rag/documents.py의 handle_upload, 삭제는 같은 파일의
soft_delete_documents, 색인 반영 시점은 rag/outbox.py, 실효 참조의 색인용 판정은
rag/search/os_index.py의 effective_searchable.

문서 식별·버전 정책:
- 식별 기준은 **filename 완전 일치** 하나뿐. 유사 파일명은 별개 문서로 본다.
- 같은 이름 재업로드 = 새 version + 기존 버전 supersede. 내용이 같아도 마찬가지 —
  내용 해시 dedupe는 하지 않는다(규칙을 하나로 유지).
  supersede가 **무엇을 언제** 하는지는 rag/ingestion.py의 index_pending_document가 정의점이다
  (구버전 청크 삭제는 커밋 즉시가 아니라 다음 회차의 DROP 작업이다 — 최대 1분 공존).
- 원본 파일은 테넌트 디렉터리 아래 UUID 이름으로 저장 (업로드 1건 = 파일 1개).
- 버전 롤백은 미지원. 되돌리려면 이전 파일을 다시 업로드한다.
- **예외**: failed 문서는 사용자에게 "없는 문서"다 — 목록에는 보이되(재시도·삭제해야 하므로)
  exists·rag/upload.py의 current_version·handle_upload 셋이 한 기준으로 없는 것처럼 답한다.
  기준 상수는 rag/models.py의 ALIVE_DOCUMENT_STATUSES, 되살리는 규칙은 handle_upload docstring.

동시 업로드:
- FE가 확인창에서 본 버전을 expect_version으로 보내면, 그 사이 DB가 바뀌었을 때 409를 준다.
  조회 후 확인창을 띄우는 흐름은 조회~업로드 사이 창이 원리적으로 남으므로(TOCTOU),
  화면에서 몇 번을 확인해도 서버 검사 없이는 남의 문서를 조용히 대체할 수 있다.
- 파일명 자동 넘버링은 하지 않는다. 이름은 사용자가 확정한다 —
  파일명이 임베딩 입력(rag/search/index_text.py)과 답변 인용 표기(rag/turn/prompts.py)에 그대로 쓰여서,
  '(1)' 같은 무의미한 이름을 서버가 싸게 만들어 주면 KB 품질이 조용히 나빠진다.
"""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.responses import FileResponse, JSONResponse

from database import get_session
from rag import document_admin as admin
from rag import upload
from rag.documents import (
    soft_delete_documents,
)
from rag.folders import require_folder
from rag.models import Document
from routers.deps import get_tenant_id, get_user_id, sanitize_page
from routers.errors import document_http_errors
from routers.uploads import checked_filename, reject_if_oversized
from schemas.kms import (
    BULK_MAX_ITEMS,
    DOC_LIST_DEFAULT_LIMIT,
    DOC_LIST_MAX_LIMIT,
    DOC_STATUSES,
    DOCUMENT_DESCRIPTION_MAX,
    DocumentBulkUpdateRequest,
    DocumentExistsResponse,
    DocumentListResponse,
    DocumentUpdateRequest,
    DocumentUploadMetadata,
    DocumentUploadResponse,
)


def _to_response(doc: Document, ref_count: int | None = None) -> DocumentUploadResponse:
    return DocumentUploadResponse(
        document_id=doc.id,
        filename=doc.filename,
        version=doc.version,
        status=doc.status,
        status_reason=doc.status_reason,
        is_active=doc.is_active,
        folder_id=doc.folder_id,
        is_searchable=doc.is_searchable,
        uploaded_at=doc.uploaded_at,
        uploaded_by=doc.uploaded_by,
        ref_count=ref_count,
        # 다섯 엔드포인트(업로드·목록·단건·일괄 PATCH·단건 PATCH)가 이 함수 하나를
        # 거치므로 여기 한 줄이 전부에 반영된다.
        description=doc.description,
        indexed_at=doc.indexed_at,
    )


router = APIRouter(prefix='/kms')

DOC_MAX_FILE_BYTES = 10 * 1024 * 1024  # 문서 업로드 크기 상한 (첨부와 별개)


def _version_conflict(filename: str, current_version: int) -> JSONResponse:
    """409 — 확인창에서 본 상태와 DB가 달라졌다.

    FE가 확인창을 다시 띄우는 데 필요한 값을 함께 준다(어느 이름이 걸렸는지 + 그 이름의 현재 버전).
    detail은 문자열로 유지 — 공용 에러 토스트가 그대로 쓰기 때문.
    """
    return JSONResponse(
        status_code=409,
        content={
            'detail': f"'{filename}' 문서 상태가 변경되었습니다. 다시 확인해 주세요.",
            'filename': filename,
            'current_version': current_version,
        },
    )


def _validation_detail(e: ValidationError, fallback: str = 'document-data') -> str:
    """pydantic 오류 하나를 "필드: 사유" 한 줄로 — 업로드 데이터의 두 수신 경로가 공유한다.

    메시지를 한 곳에 모아둔 것은 경로마다 다른 문구가 나가지 않게 하려는 것이다.
    """
    first = e.errors()[0]
    where = '.'.join(str(x) for x in first['loc']) or fallback
    return f'{where}: {first["msg"]}'


async def _resolve_upload_data(
    document_data: UploadFile | str | None, description: str | None, expect_version: int | None
) -> tuple[DocumentUploadMetadata, bool]:
    """업로드 데이터를 확정한다. 반환: (데이터, folder_id를 실제로 보냈는지).

    받는 방식이 둘이다 — 새 방식은 `document-data` 파트의 JSON 한 덩어리, 옛 방식은 평면 Form
    필드(description·expect_version). 옛 방식은 FE 전환 전 호환용이고, 그쪽에는 folder_id가 없다.
    둘을 섞어 보내면 **400으로 거절한다** — 한쪽을 조용히 무시하면 "분명히 보냈는데 안 먹는"
    버그가 되고, 전환 중에 섞여 나가는 사고를 늦게 발견한다.

    JSON 파트는 두 모양으로 도착한다: 브라우저가 Blob으로 실으면 filename이 붙어 UploadFile로,
    문자열로 실으면 str로 들어온다. 어느 쪽이든 본문은 같으므로 여기서 하나로 만든다.
    """
    if document_data is None:
        # 옛 방식도 try로 감싼다 — description에 max_length가 있어 이 생성이
        # ValidationError를 낼 수 있다. FastAPI는 **수동 생성**한 pydantic 예외를 422로
        # 바꿔주지 않으므로, 감싸지 않으면 201자를 평면 Form으로 보낼 때 500이 난다.
        try:
            return DocumentUploadMetadata(description=description, expect_version=expect_version), False
        except ValidationError as e:
            raise HTTPException(
                status_code=422, detail=f'입력값이 올바르지 않습니다 ({_validation_detail(e, "description")}).'
            ) from e
    if description is not None or expect_version is not None:
        raise HTTPException(
            status_code=400,
            detail='document-data 파트와 개별 필드(description·expect_version)를 함께 보낼 수 없습니다. 하나만 사용해 주세요.',
        )
    raw = (
        document_data
        if isinstance(document_data, str)
        else (await document_data.read()).decode('utf-8', errors='replace')
    )
    try:
        meta = DocumentUploadMetadata.model_validate_json(raw)
    except ValidationError as e:
        raise HTTPException(
            status_code=422, detail=f'document-data JSON이 올바르지 않습니다 ({_validation_detail(e)}).'
        ) from e
    # null 전송과 미전송을 가르는 지점 — PATCH의 update_document와 같은 방식이다.
    return meta, 'folder_id' in meta.model_fields_set


@router.post('/documents', response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    # 업로드 데이터 JSON 파트 — folder_id·description·expect_version을 한 덩어리로.
    # ICCS의 @RequestPart("sr-data") 관례와 같은 꼴이라 이름도 그 관례를 따른다.
    # Content-Type: application/json을 실은 파트(브라우저 Blob → UploadFile로 들어온다)와
    # 평범한 문자열 필드를 **둘 다** 받는다.
    document_data: Annotated[UploadFile | str | None, Form(alias='document-data')] = None,
    # ── 아래 둘은 옛 방식(평면 Form 필드). document-data 파트가 없을 때만 쓴다 ──
    # 표 설명 (xlsx 검색 보강). xlsx는 필수 — 판정은 계승까지 따진 뒤 rag/upload.py의 _xlsx_precheck가 한다.
    description: str | None = Form(None, max_length=DOCUMENT_DESCRIPTION_MAX),
    # 낙관적 잠금 (선택). exists 응답의 version을 그대로 보낸다 — 없는 이름이면 0.
    #   미전송 → 검사 없음 (기존 호출부 호환)
    #   0      → 아직 아무 버전도 없어야 함 (새 문서로 등록하려는 경우)
    #   N      → 현재 버전이 N이어야 함 (대체하려는 경우)
    expect_version: int | None = Form(None),
    tenant_id: str = Depends(get_tenant_id),
    # 등록자. 헤더 미전송이면 None → uploaded_by=NULL로 남긴다.
    # conversation의 DEFAULT_USER('test-user') 폴백을 여기선 쓰지 않는다 —
    # 대화 스코핑 키와 달리 이 값은 문서 관리 화면의 '등록자' 컬럼에 그대로 뜬다.
    user_id: str | None = Depends(get_user_id),
    session: AsyncSession = Depends(get_session),
) -> DocumentUploadResponse | JSONResponse:
    # 0. 형식·크기 게이트 (인덱싱 전에 명시적으로 막음)
    reject_if_oversized(request, DOC_MAX_FILE_BYTES)  # 거대 본문은 read 전에 차단
    # 이하 전부 정규화된 값을 쓴다(원본 file.filename 금지) — 브라우저가 NFD로 주면 DB에 분해형이 저장되고 LLM이 NFC로 인용해 매칭이 깨진다.
    filename = checked_filename(file)

    # 0-1. 업로드 데이터 확정 + 폴더 검증. **blob을 쓰기 전에** 한다(rag/upload.py 모듈 docstring).
    meta, folder_given = await _resolve_upload_data(document_data, description, expect_version)
    if meta.folder_id is not None:
        with document_http_errors():
            await require_folder(session, tenant_id, meta.folder_id)  # 남의 테넌트 폴더 차단

    # 1. 업로드 바이트 전체를 읽는다
    content = await file.read()
    if len(content) > DOC_MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='파일이 10MB를 초과합니다.')

    # 2. xlsx 사전검증 → blob 저장 → 낙관적 잠금 → 저장·커밋
    try:
        doc = await upload.store_upload(
            session, upload.Upload(tenant_id, user_id, filename, content, meta, folder_given)
        )
    except upload.UploadRejected as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except upload.VersionConflict as e:
        return _version_conflict(e.filename, e.current_version)
    return _to_response(doc)


@router.get('/documents', response_model=DocumentListResponse)
async def list_documents(
    limit: int = DOC_LIST_DEFAULT_LIMIT,
    offset: int = 0,
    q: str | None = None,  # 파일명 부분 일치
    folder_id: int | None = None,  # 0 = 미분류만, 그 외 = 그 폴더
    status: Annotated[list[str] | None, Query()] = None,  # 복수 선택 가능
    is_searchable: bool | None = None,  # 문서 스위치 값 기준 (아래 주석)
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> DocumentListResponse:
    """테넌트의 문서 목록 — 페이징·검색·필터 (#176). 최신 업로드가 위로 온다.

    supersede된 구버전(deleted)은 제외한다 — 죽은 행에 폴더/참조 컨트롤이 노출되는 혼란 방지.
    형태(limit/offset·items/total/has_more)는 대화 목록(routers/conversations.py)과 맞췄다.

    `is_searchable`은 **documents.is_searchable 값**으로만 거른다. 실제로 검색에 쓰이는지는
    폴더 스위치와의 곱(실효 참조 — _effective_ref)이지만, 그 판정을 SQL로 옮기면 사본이 하나 더
    생긴다. 참조 off 폴더를 실제로 운영하기 시작하면 그때 실효 기준으로 올린다 —
    파라미터 이름이 그대로라 FE 계약은 바뀌지 않는다. (#176 결정)
    """
    limit, offset, q = sanitize_page(limit, offset, q, DOC_LIST_MAX_LIMIT)
    if status and not set(status) <= DOC_STATUSES:
        raise HTTPException(status_code=422, detail=f'알 수 없는 status: {sorted(set(status) - DOC_STATUSES)}')

    page = await admin.list_documents(
        session, tenant_id, admin.DocumentFilter(q, folder_id, status, is_searchable), limit, offset
    )
    return DocumentListResponse(
        items=[_to_response(d, page.ref_counts.get(d.filename, 0)) for d in page.docs],
        total=page.total,
        # len(docs)를 쓴다 — limit을 쓰면 마지막 부분 페이지에서 어긋난다(대화 목록과 같은 이유).
        has_more=offset + len(page.docs) < page.total,
    )


# ⚠ 이 라우트는 반드시 '/documents/{document_id}'보다 **위에** 있어야 한다.
#   FastAPI는 등록 순서대로 매칭하므로, 아래에 두면 'exists'가 document_id(int)로 파싱돼 422가 난다.
@router.get('/documents/exists', response_model=DocumentExistsResponse)
async def document_exists(
    filename: str, tenant_id: str = Depends(get_tenant_id), session: AsyncSession = Depends(get_session)
) -> DocumentExistsResponse:
    """업로드 전 동일 파일명 확인 (FE가 대체 확인 창을 띄울지 판단).

    판정 기준은 _current_version·handle_upload의 재사용 판정과 **정확히 같아야** 한다 — tenant +
    filename **완전 일치**(대소문자·공백 구분), status가 살아 있는 것(ALIVE_DOCUMENT_STATUSES).
    기준이 어긋나면 "물어본 것과 다른 문서가 지워지는" 사고가 난다. failed만 있으면 exists=false다
    (#161) — 재업로드가 그 행을 되살리므로 FE는 새 문서처럼(expect_version=0) 보내면 된다.

    이 API는 안내용일 뿐 강제력이 없다. 업로드 API는 확인 없이도 통과하며(2026-08-05 결정),
    그 경우 기존 버전이 그대로 대체된다.
    """
    doc = await admin.find_alive(session, tenant_id, filename)
    if doc is None:
        return DocumentExistsResponse(exists=False)
    return DocumentExistsResponse(
        exists=True,
        document_id=doc.id,
        version=doc.version,
        status=doc.status,
        uploaded_at=doc.uploaded_at,
    )


@router.get('/documents/{document_id}', response_model=DocumentUploadResponse)
async def get_document(
    document_id: int, tenant_id: str = Depends(get_tenant_id), session: AsyncSession = Depends(get_session)
) -> DocumentUploadResponse:
    with document_http_errors():
        return _to_response(await admin.get_document(session, tenant_id, document_id))


@router.patch('/documents', response_model=list[DocumentUploadResponse])
async def bulk_update_documents(
    request: DocumentBulkUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> list[DocumentUploadResponse]:
    """문서 여러 건의 폴더 소속·참조 on/off를 한 번에 바꾼다 (#166).

    단건 PATCH를 N번 부른 것과 결과는 같지만 커밋도, 색인 갱신 대기열 행도 한 번이다.
    대상이 하나라도 어긋나면(없는 id·다른 테넌트·삭제됨) **아무것도 바꾸지 않고 404**를 준다 —
    화면에서 체크한 목록과 결과가 어긋나면 무엇이 반영됐는지 되짚을 방법이 없다.
    """
    with document_http_errors():
        docs = await admin.bulk_update_documents(session, tenant_id, request)
    # ref_count는 비운다 — 목록 API 전용 집계다(_to_response 기본값).
    return [_to_response(d) for d in docs]


@router.patch('/documents/{document_id}', response_model=DocumentUploadResponse)
async def update_document(
    document_id: int,
    request: DocumentUpdateRequest,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> DocumentUploadResponse:
    """문서 속성 변경 (F2): 폴더 소속·참조 on/off·표 설명. 보낸 필드만 반영.

    **경로가 둘이다 (#194).** folder_id·is_searchable은 색인의 비정규화 메타만 바꾸므로
    `META_DOCUMENTS` 부분 갱신으로 끝난다. 반면 `description`은 xlsx 청크 **본문**에 병합되는
    값이라(rag/chunking/xlsx.chunk_xlsx) 메타 갱신으로는 반영되지 않는다 — 재파싱·재임베딩이
    따르는 전체 재색인이 필요하다. 그래서 이 핸들러에 "가벼운 갱신"과 "무거운 재인제스션"이
    공존한다. 둘은 **배타적으로** 적재한다(재색인이 돌면 index_pending_document가 폴더·토글
    최신값을 스스로 다시 읽으므로 META 행은 중복이다).
    """
    with document_http_errors():
        doc = await admin.get_document(session, tenant_id, document_id)
        doc = await admin.update_document(session, doc, request)
    return _to_response(doc)


@router.delete('/documents', status_code=204)
async def bulk_delete_documents(
    ids: Annotated[list[int] | None, Query(max_length=BULK_MAX_ITEMS)] = None,
    tenant_id: str = Depends(get_tenant_id),
    session: AsyncSession = Depends(get_session),
) -> None:
    """문서 여러 건 삭제 (#174). 실제 처리는 단건과 **같은 경로**(soft_delete_documents)다.

    id 목록을 본문이 아니라 쿼리 파라미터로 받는다 — DELETE 요청의 content는 RFC 9110상
    정의된 의미가 없어 중간 장비가 버릴 여지가 있다. 200건이 1.8KB라 URL 길이도 문제없다.

    없는 id·다른 테넌트 id가 하나라도 있으면 **아무것도 지우지 않고 404**(일괄 변경 #166과 같다).
    반면 **이미 삭제된 문서는 통과시킨다** — 단건 DELETE가 원래 그렇게 동작했고(멱등),
    삭제는 "그 문서가 없는 상태"를 만드는 것이라 이미 그 상태면 실패로 볼 이유가 없다.
    """
    if not ids:
        return  # 지울 것이 없는 요청은 조회도 하지 않는다
    with document_http_errors():
        unique = await admin.bulk_delete_targets(session, tenant_id, ids)
    await soft_delete_documents(session, tenant_id, unique)
    await session.commit()


@router.delete('/documents/{document_id}', status_code=204)
async def delete_document(
    document_id: int, tenant_id: str = Depends(get_tenant_id), session: AsyncSession = Depends(get_session)
) -> None:
    """문서 소프트 삭제 (F5). 같은 filename의 전 버전을 비활성 처리한다.

    무엇을 하는지는 rag/documents.py의 soft_delete_documents가 정의점이다 —
    일괄 삭제(#174)와 같은 경로를 탄다. 규칙이 두 벌로 갈리면 한쪽만 고쳐지는 사고가 난다.
    """
    doc_ids = await soft_delete_documents(session, tenant_id, [document_id])
    if not doc_ids:
        raise HTTPException(status_code=404, detail="document not found")
    await session.commit()


@router.get('/documents/{document_id}/download')
async def download_document(
    document_id: int, tenant_id: str = Depends(get_tenant_id), session: AsyncSession = Depends(get_session)
) -> FileResponse:
    """원본 blob 파일 다운로드. 답변 인용(sources)의 document_id에서 연결된다."""
    with document_http_errors():
        # 삭제된 버전도 연다 — 답변 인용에 남은 옛 버전 원본을 받을 수 있어야 한다.
        doc = await admin.get_document(session, tenant_id, document_id, include_deleted=True)
    blob = Path(doc.blob_path)
    if not blob.is_file():
        raise HTTPException(status_code=410, detail="blob file missing")
    return FileResponse(blob, media_type=doc.mime, filename=doc.filename)
