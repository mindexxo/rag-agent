"""KMS 문서 업로드 라우터.

POST /kms/documents (multipart)

문서 식별·버전 정책 (2026-08-05 확정):
- 식별 기준은 **filename 완전 일치** 하나뿐. 유사 파일명은 별개 문서로 본다.
- 같은 이름 재업로드 = 새 version + 기존 버전 supersede(비활성화 + 청크 삭제 + 근거 캐시 무효화).
  내용이 같아도 마찬가지 — 내용 해시 dedupe는 제거했다(규칙을 하나로 유지).
- 원본 파일은 테넌트 디렉터리 아래 UUID 이름으로 저장 (업로드 1건 = 파일 1개).
- 버전 롤백은 미지원. 되돌리려면 이전 파일을 다시 업로드한다.
- **예외(#161)**: failed 문서는 "이미 있는 문서"로 세지 않는다 — 확인창 없이 통과하고, 재업로드는
  새 version을 만들지 않고 그 행을 pending으로 되살린다(version 유지). 상세는 rag/documents.py의
  handle_upload docstring. failed는 목록에는 보인다 — 재시도·삭제할 수 있어야 하므로.

동시 업로드 (2026-08-07 추가):
- FE가 확인창에서 본 버전을 expect_version으로 보내면, 그 사이 DB가 바뀌었을 때 409를 준다.
  조회 후 확인창을 띄우는 흐름은 조회~업로드 사이 창이 원리적으로 남으므로(TOCTOU),
  화면에서 몇 번을 확인해도 서버 검사 없이는 남의 문서를 조용히 대체할 수 있다.
- 파일명 자동 넘버링은 하지 않는다. 이름은 사용자가 확정한다 —
  파일명이 임베딩 입력(rag/index_text.py)과 답변 인용 표기(rag/prompts.py)에 그대로 쓰여서,
  '(1)' 같은 무의미한 이름을 서버가 싸게 만들어 주면 KB 품질이 조용히 나빠진다.
"""
import tempfile
from typing import Annotated
from uuid import uuid4
from pathlib import Path

from fastapi import APIRouter, File, Form, Query, Request, UploadFile, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy import text as sql_text
from pydantic import ValidationError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse

from config import settings
from database import get_session
from rag import cache, outbox
from rag.chunking import extract_text
from rag.documents import FailedReuseConflict, handle_upload, soft_delete_documents
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Folder
from routers.kms import get_tenant_id, get_user_id
from schemas.kms import (ATTACHMENT_FILENAME_MAX, ATTACHMENT_MAX_TEXT_CHARS, BULK_MAX_ITEMS,
                         DOC_LIST_DEFAULT_LIMIT, DOC_LIST_MAX_LIMIT, DOC_STATUSES,
                         DocumentBulkUpdateRequest, DocumentExistsResponse, DocumentListResponse,
                         DocumentUploadMetadata, DocumentUploadResponse, DocumentUpdateRequest,
                         QueryAttachment)
from text_norm import LIKE_ESCAPE_CHAR, like_pattern, normalize_filename


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
    )

router = APIRouter(prefix='/kms')

# F1a: 지원 형식 화이트리스트 (확장자 기준). 그 외는 400.
SUPPORTED_SUFFIXES = {'.pdf', '.docx', '.xlsx', '.txt', '.md'}
DOC_MAX_FILE_BYTES = 10 * 1024 * 1024   # 문서 업로드 크기 상한 (첨부와 별개)


def _reject_if_oversized(request: Request, limit: int) -> None:
    """Content-Length로 명백한 초과 업로드를 body read 전에 차단 (C2-A, 코스 가드).
    조작·multipart 오버헤드로 부정확할 수 있어 정확 경계는 read 후 len() 검사가 담당 —
    여기선 여유(margin)를 두고 '거대 본문을 메모리에 올리기 전에 끊는' 용도. (실전 상한은 nginx도 병행)"""
    cl = request.headers.get('content-length')
    if cl and cl.isdigit() and int(cl) > limit + 8192:
        raise HTTPException(status_code=413, detail='파일이 너무 큽니다.')


async def _current_version(session: AsyncSession, tenant_id: str, filename: str) -> int:
    """해당 파일명의 현재 버전. 없으면 0.

    판정 기준은 exists API·handle_upload의 재사용 판정과 **정확히 같아야** 한다 — tenant + filename
    완전 일치, status가 살아 있는 것(ALIVE_DOCUMENT_STATUSES — 세 곳이 같은 상수를 쓴다). 기준이 어긋나면 "확인창에서 본 것과 다른 문서가
    대체되는" 사고가 난다. failed를 빼는 이유는 #161 — 그 문서는 검색에 없고 재업로드가 그 행을
    되살리므로(version 유지) 사용자에겐 "없는 문서"다.
    """
    return (await session.execute(
        select(func.max(Document.version))
        .where(Document.tenant_id == tenant_id)      # 격리 — WHERE 절 명시
        .where(Document.filename == filename)
        .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
    )).scalar() or 0


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

async def _resolve_upload_data(document_data: UploadFile | str | None, description: str | None,
                               expect_version: int | None) -> tuple[DocumentUploadMetadata, bool]:
    """업로드 데이터를 확정한다 (#165). 반환: (데이터, folder_id를 실제로 보냈는지).

    받는 방식이 둘이다 — 새 방식은 `document-data` 파트의 JSON 한 덩어리, 옛 방식은 평면 Form
    필드(description·expect_version). 옛 방식은 FE 전환 전 호환용이고, 그쪽에는 folder_id가 없다.
    둘을 섞어 보내면 **400으로 거절한다** — 한쪽을 조용히 무시하면 "분명히 보냈는데 안 먹는"
    버그가 되고, 전환 중에 섞여 나가는 사고를 늦게 발견한다.

    JSON 파트는 두 모양으로 도착한다: 브라우저가 Blob으로 실으면 filename이 붙어 UploadFile로,
    문자열로 실으면 str로 들어온다. 어느 쪽이든 본문은 같으므로 여기서 하나로 만든다.
    """
    if document_data is None:
        return DocumentUploadMetadata(description=description,
                                      expect_version=expect_version), False
    if description is not None or expect_version is not None:
        raise HTTPException(
            status_code=400,
            detail='document-data 파트와 개별 필드(description·expect_version)를 함께 보낼 수 없습니다. 하나만 사용해 주세요.')
    raw = (document_data if isinstance(document_data, str)
           else (await document_data.read()).decode('utf-8', errors='replace'))
    try:
        meta = DocumentUploadMetadata.model_validate_json(raw)
    except ValidationError as e:
        first = e.errors()[0]
        where = '.'.join(str(x) for x in first['loc']) or 'document-data'
        raise HTTPException(status_code=422,
                            detail=f'document-data JSON이 올바르지 않습니다 ({where}: {first["msg"]}).')
    # null 전송과 미전송을 가르는 지점 — PATCH의 update_document와 같은 방식이다.
    return meta, 'folder_id' in meta.model_fields_set


@router.post('/documents', response_model=DocumentUploadResponse)
async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        # 업로드 데이터 JSON 파트 (#165) — folder_id·description·expect_version을 한 덩어리로.
        # ICCS의 @RequestPart("sr-data") 관례와 같은 꼴이라 이름도 그 관례를 따른다.
        # Content-Type: application/json을 실은 파트(브라우저 Blob → UploadFile로 들어온다)와
        # 평범한 문자열 필드를 **둘 다** 받는다 — 둘 다 실측했다.
        document_data: Annotated[UploadFile | str | None, Form(alias='document-data')] = None,
        # ── 아래 둘은 옛 방식(평면 Form 필드). document-data 파트가 없을 때만 쓴다 ──
        description: str | None = Form(None),   # F1a: 표 설명 (xlsx 검색 보강). 선택
        # 낙관적 잠금 (선택). exists 응답의 version을 그대로 보낸다 — 없는 이름이면 0.
        #   미전송 → 검사 없음 (기존 호출부 호환)
        #   0      → 아직 아무 버전도 없어야 함 (새 문서로 등록하려는 경우)
        #   N      → 현재 버전이 N이어야 함 (대체하려는 경우)
        expect_version: int | None = Form(None),
        tenant_id: str = Depends(get_tenant_id),
        # 등록자 (#164). 헤더 미전송이면 None → uploaded_by=NULL로 남긴다.
        # conversation의 DEFAULT_USER('test-user') 폴백을 여기선 쓰지 않는다 —
        # 대화 스코핑 키와 달리 이 값은 문서 관리 화면의 '등록자' 컬럼에 그대로 뜬다.
        user_id: str | None = Depends(get_user_id),
        session: AsyncSession = Depends(get_session)
):
    # 0. 형식·크기 게이트 (인덱싱 전에 명시적으로 막음)
    _reject_if_oversized(request, DOC_MAX_FILE_BYTES)   # 거대 본문은 read 전에 차단 (C2-A)
    if not file.filename:   # multipart에 filename 누락 시 Path(None) TypeError→500 방지 (C2)
        raise HTTPException(status_code=400, detail='파일명이 없습니다.')
    # 경계 정규화 (#34) — 브라우저가 macOS 파일명을 NFD로 주면 DB에 분해형이 저장되고,
    # LLM이 NFC로 인용해 매칭이 조용히 깨진다. 이하 전부 이 값을 쓴다 (원본 file.filename 금지).
    filename = normalize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f'지원하지 않는 형식입니다: {suffix or "확장자 없음"}. 지원: PDF/DOCX/XLSX/TXT/MD')

    # 0-1. 업로드 데이터 확정 + 폴더 검증 (#165).
    #      **blob을 쓰기 전에** 한다 — 뒤에 두면 실패 경로마다 unlink를 챙겨야 한다(지금 2곳).
    meta, folder_given = await _resolve_upload_data(document_data, description, expect_version)
    if meta.folder_id is not None:
        folder = (await session.execute(
            select(Folder)
            .where(Folder.tenant_id == tenant_id)   # 다른 테넌트 폴더 지정 차단
            .where(Folder.id == meta.folder_id)
        )).scalars().first()
        if folder is None:
            raise HTTPException(status_code=404, detail="folder not found")

    #1. 업로드 바이트 전체를 읽는다
    content = await file.read()
    if len(content) > DOC_MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='파일이 10MB를 초과합니다.')

    # 2. blob 저장. 테넌트별 디렉터리 아래 UUID 파일명으로 저장한다 (2026-08-05).
    #    업로드 1건 = 파일 1개로 고정 — 내용이 같아도 경로를 공유하지 않는다.
    #    (내용 해시를 쓰면 v1·v2가 같은 파일을 가리켜, 나중에 비활성 문서 blob을
    #     정리할 때 살아 있는 문서의 원본까지 지워질 수 있다. 디스크는 조금 더 쓰지만
    #     '문서 식별은 filename 하나'라는 정책과도 일관된다.)
    blob_dir = Path(settings.blob_storage_dir) / tenant_id
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{uuid4().hex}{suffix}"
    blob_path.write_bytes(content)

    # 3. 낙관적 잠금 — 확인창에서 본 상태와 지금 DB가 같은지 본다.
    #    이 조회만으로는 조회~insert 사이 창이 남는다. 그 창은 아래 IntegrityError가 닫는다.
    if meta.expect_version is not None:
        current = await _current_version(session, tenant_id, filename)
        if current != meta.expect_version:
            blob_path.unlink(missing_ok=True)      # 참조되지 않는 blob 남기지 않기
            return _version_conflict(filename, current)

    # 4. 파일 인덱싱 후 저장 (mime은 handle_upload가 blob_path에서 직접 구한다)
    try:
        doc, stale_blob = await handle_upload(
            session, tenant_id, filename, blob_path, description=meta.description,
            uploaded_by=user_id, folder_id=meta.folder_id, folder_given=folder_given,
        )
        await session.commit()
    except (IntegrityError, FailedReuseConflict):
        # IntegrityError: 같은 이름·같은 version이 방금 먼저 들어왔다 — UNIQUE(tenant_id, filename,
        # version). 위 조회를 두 요청이 함께 통과했을 때의 최종 방어선.
        # (expect_version 미전송 호출도 여기서 409가 된다 — 이전엔 그대로 터져 500이었다)
        # FailedReuseConflict(#161): 같은 failed를 둘이 동시에 되살리려 했고 이쪽이 졌다.
        # 둘 다 새 blob만 지운다 — 옛 blob은 DB가 여전히 가리키고 있다.
        await session.rollback()
        blob_path.unlink(missing_ok=True)
        return _version_conflict(filename, await _current_version(session, tenant_id, filename))
    # failed 행을 되살린 경우 옛 blob은 **커밋 뒤에** 지운다 (#161) — 커밋 전에 지우면 롤백 시
    # DB는 옛 경로를 가리키는데 파일은 없다. failed는 인용된 적이 없어 보존할 이유도 없다.
    if stale_blob:
        Path(stale_blob).unlink(missing_ok=True)

    # 색인 대기열 행은 handle_upload가 문서와 같은 트랜잭션에 등록했다 (#139 outbox) —
    # 워커 cron(1분)이 처리한다. 응답 시점의 status는 pending이고, 검색 반영은 최대 1~5분.
    return _to_response(doc)

@router.get('/documents', response_model=DocumentListResponse)
async def list_documents(
        limit: int = DOC_LIST_DEFAULT_LIMIT,
        offset: int = 0,
        q: str | None = None,                  # 파일명 부분 일치
        folder_id: int | None = None,          # 0 = 미분류만, 그 외 = 그 폴더
        status: Annotated[list[str] | None, Query()] = None,   # 복수 선택 가능
        is_searchable: bool | None = None,     # 문서 스위치 값 기준 (아래 주석)
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """테넌트의 문서 목록 — 페이징·검색·필터 (#176). 최신 업로드가 위로 온다.

    supersede된 구버전(deleted)은 제외한다 — 죽은 행에 폴더/참조 컨트롤이 노출되는 혼란 방지.
    형태(limit/offset·items/total/has_more)는 대화 목록(routers/conversations.py)과 맞췄다.

    `is_searchable`은 **documents.is_searchable 값**으로만 거른다. 실제로 검색에 쓰이는지는
    폴더 스위치와의 곱(실효 참조)이지만, 그 판정을 SQL로 옮기면 같은 규칙의 네 번째 사본이
    된다(rag/os_index.py의 effective_searchable, _folder_is_on, bulk_update_documents의
    _effective). 참조 off 폴더를 실제로 운영하기 시작하면 그때 실효 기준으로 올린다 —
    파라미터 이름이 그대로라 FE 계약은 바뀌지 않는다. (#176 결정)
    """
    # HTTP 파라미터 위생 — 범위 밖 limit은 상한으로, 음수 offset은 0으로 (대화 목록과 같은 처리).
    if not 1 <= limit <= DOC_LIST_MAX_LIMIT:
        limit = DOC_LIST_MAX_LIMIT
    if offset < 0:
        offset = 0
    q = (q or '').strip() or None      # 빈 문자열을 필터로 쓰면 '%%'가 되어 전건 매칭이다
    if status and not set(status) <= DOC_STATUSES:
        raise HTTPException(status_code=422,
                            detail=f'알 수 없는 status: {sorted(set(status) - DOC_STATUSES)}')

    # 조건은 **한 번만** 만들어 count와 페이지 쿼리가 함께 쓴다 — 갈라지면 has_more가 어긋난다.
    where = (Document.tenant_id == tenant_id) & (Document.status != 'deleted')
    if q is not None:
        # 파일명은 경계에서 NFC로 저장된다(#34). 검색어도 맞춰야 한다 — text_norm의 정책은
        # "타이핑 입력은 IME가 NFC를 내므로 위험군이 아니다"지만, 파일명 검색은 사용자가
        # **이름을 복사해 붙여넣는** 경로가 흔하고 macOS에서 복사한 이름은 NFD일 수 있다.
        where &= Document.filename.ilike(like_pattern(normalize_filename(q)),
                                         escape=LIKE_ESCAPE_CHAR)
    if folder_id is not None:
        # 0은 '미분류' 약속 — 폴더 id는 1부터라 유효한 값과 겹치지 않는다. 쿼리 파라미터에는
        # null이 없어서(#165의 multipart와 같은 제약) 값 하나로 표현했다.
        where &= Document.folder_id.is_(None) if folder_id == 0 else Document.folder_id == folder_id
    if status:
        where &= Document.status.in_(status)
    if is_searchable is not None:
        where &= Document.is_searchable == is_searchable

    total = (await session.execute(select(func.count(Document.id)).where(where))).scalar_one()
    docs = (await session.execute(
        select(Document)
        .where(where)
        .order_by(Document.uploaded_at.desc(), Document.id.desc())
        .offset(offset)
        .limit(limit)
    )).scalars().all()

    # 인용 횟수: 저장 시 확정된 실인용 목록(cited_docs)을 filename별 집계 (F5).
    # sources(검색 후보 노출 수)가 아닌 실인용 — stats top_documents와 정의 통일.
    # filename 키라 버전 교체 후에도 카운트가 이어진다.
    # **이번 페이지의 파일명으로 좁힌다** (#176) — 예전엔 테넌트의 messages 전체를 훑었다.
    ref_counts: dict[str, int] = {}
    if docs:
        ref_rows = (await session.execute(sql_text("""
            SELECT d AS filename, count(*) AS cnt
            FROM messages, jsonb_array_elements_text(messages.cited_docs) AS d
            WHERE messages.tenant_id = :tenant_id
              AND messages.role = 'assistant'
              AND jsonb_typeof(messages.cited_docs) = 'array'
              AND d = ANY(:filenames)
            GROUP BY 1
        """), {"tenant_id": tenant_id, "filenames": list({d.filename for d in docs})})).all()
        ref_counts = {r.filename: r.cnt for r in ref_rows}

    return DocumentListResponse(
        items=[_to_response(d, ref_counts.get(d.filename, 0)) for d in docs],
        total=total,
        # len(docs)를 쓴다 — limit을 쓰면 마지막 부분 페이지에서 어긋난다(대화 목록과 같은 이유).
        has_more=offset + len(docs) < total,
    )


# ⚠ 이 라우트는 반드시 '/documents/{document_id}'보다 **위에** 있어야 한다.
#   FastAPI는 등록 순서대로 매칭하므로, 아래에 두면 'exists'가 document_id(int)로 파싱돼 422가 난다.
@router.get('/documents/exists', response_model=DocumentExistsResponse)
async def document_exists(
        filename: str,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """업로드 전 동일 파일명 확인 (FE가 대체 확인 창을 띄울지 판단).

    판정 기준은 _current_version·handle_upload의 재사용 판정과 **정확히 같아야** 한다 — tenant +
    filename **완전 일치**(대소문자·공백 구분), status가 deleted·failed가 아닌 것(#161). 기준이
    어긋나면 "물어본 것과 다른 문서가 지워지는" 사고가 난다. failed만 있으면 exists=false다 —
    재업로드가 그 행을 되살리므로 FE는 새 문서처럼(expect_version=0) 보내면 된다.

    이 API는 안내용일 뿐 강제력이 없다. 업로드 API는 확인 없이도 통과하며(2026-08-05 결정),
    그 경우 기존 버전이 그대로 대체된다.
    """
    # 업로드와 같은 경계 정규화 (#34) — 여기만 빠지면 위 "판정 기준이 정확히 같아야 한다"가
    # 깨져, NFD 이름으로 물어본 클라이언트가 exists=false를 받고 중복 문서를 만든다.
    filename = normalize_filename(filename)
    doc = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)      # 격리 — WHERE 절 명시
        .where(Document.filename == filename)
        .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
        .order_by(Document.version.desc())
        .limit(1)
    )).scalars().first()

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
        document_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    doc = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.id == document_id)
    )).scalars().first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    return _to_response(doc)


async def _folder_is_on(session: AsyncSession, tenant_id: str, folder_id: int | None) -> bool:
    """폴더의 참조 여부. 미분류(None)는 항상 on — retriever 판정과 동일 규칙."""
    if folder_id is None:
        return True
    return bool((await session.execute(
        select(Folder.is_searchable)
        .where(Folder.tenant_id == tenant_id)
        .where(Folder.id == folder_id)
    )).scalar())


@router.patch('/documents', response_model=list[DocumentUploadResponse])
async def bulk_update_documents(
        request: DocumentBulkUpdateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """문서 여러 건의 폴더 소속·참조 on/off를 한 번에 바꾼다 (#166).

    단건 PATCH를 N번 부른 것과 결과는 같지만 커밋도, 색인 갱신 대기열 행도 한 번이다.
    대상이 하나라도 어긋나면(없는 id·다른 테넌트·삭제됨) **아무것도 바꾸지 않고 404**를 준다 —
    화면에서 체크한 목록과 결과가 어긋나면 무엇이 반영됐는지 되짚을 방법이 없다.
    """
    changing_folder = 'folder_id' in request.model_fields_set   # null 전송(미분류)과 미전송 구분
    changing_searchable = request.is_searchable is not None
    if not request.document_ids or not (changing_folder or changing_searchable):
        return []    # 바꿀 것이 없는 요청은 조회도 하지 않는다

    # 1. 대상 확정. 테넌트 격리 규약 2항(UPDATE 전에 스코프 조회로 id를 확정한다)을 이 단계가
    #    겸한다 — 아래 outbox가 부르는 os_index.sync_meta_documents_now는 document_ids에
    #    tenant 필터를 걸지 않고 호출부의 사전 스코프를 전제하기 때문이다.
    ids = list(dict.fromkeys(request.document_ids))    # 중복 제거(순서 유지) — 같은 id를 두 번 세지 않게
    docs = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.status != 'deleted')
        .where(Document.id.in_(ids))
        .order_by(Document.id)
    )).scalars().all()
    missing = [i for i in ids if i not in {d.id for d in docs}]
    if missing:
        # 남의 테넌트 id인지 없는 id인지 구분해 알려주지 않는다 — 요청자가 보낸 값을 되돌려줄 뿐.
        raise HTTPException(status_code=404, detail=f'문서를 찾을 수 없습니다: {missing[:10]}')

    # 2. 폴더 검증은 1회면 된다(대상 전부가 같은 폴더로 간다).
    folder_on: dict[int, bool] = {}
    if changing_folder and request.folder_id is not None:
        folder = (await session.execute(
            select(Folder)
            .where(Folder.tenant_id == tenant_id)   # 다른 테넌트 폴더 지정 차단
            .where(Folder.id == request.folder_id)
        )).scalars().first()
        if folder is None:
            raise HTTPException(status_code=404, detail="folder not found")
        folder_on[folder.id] = folder.is_searchable

    # 3. 변경 **전** 실효 참조를 따지려면 각 문서가 지금 속한 폴더의 상태가 필요하다.
    #    문서마다 폴더가 다르므로 한 번에 모아 온다 — 단건처럼 문서 수만큼 조회하지 않는다.
    current_folder_ids = {d.folder_id for d in docs if d.folder_id is not None} - folder_on.keys()
    if current_folder_ids:
        folder_on.update({r.id: r.is_searchable for r in (await session.execute(
            select(Folder.id, Folder.is_searchable)
            .where(Folder.tenant_id == tenant_id)
            .where(Folder.id.in_(current_folder_ids))
        )).all()})

    def _effective(doc: Document) -> bool:
        """실효 참조 = 문서 on AND (미분류 OR 폴더 on) — _folder_is_on과 같은 규칙."""
        return doc.is_searchable and (doc.folder_id is None or bool(folder_on.get(doc.folder_id)))

    # 4. 캐시 무효화는 **문서별로** 판정한다. 문서마다 현재 폴더·스위치가 달라 on→off 전이도
    #    제각각이다 — 일괄로 판정하면 off된 문서를 근거로 만든 답변이 캐시로 계속 나간다.
    for doc in docs:
        before = _effective(doc)
        if changing_folder:
            doc.folder_id = request.folder_id
        if changing_searchable:
            doc.is_searchable = request.is_searchable
        if before and not _effective(doc):
            await cache.invalidate_source(session, tenant_id, doc.id)

    # 5. 색인의 비정규화 메타 갱신은 **한 행**에 담는다 (#139). 워커는 최종값이 같은 문서끼리
    #    묶어 갱신하므로(os_index.sync_meta_documents_now) 200건이어도 보통 질의 1~2회다.
    outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=[d.id for d in docs])
    await session.commit()
    # ref_count는 비운다 — 목록 API 전용 집계다(_to_response 기본값).
    return [_to_response(d) for d in docs]


@router.patch('/documents/{document_id}', response_model=DocumentUploadResponse)
async def update_document(
        document_id: int,
        request: DocumentUpdateRequest,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """문서 속성 변경 (F2): 폴더 소속·참조 on/off. 보낸 필드만 반영."""
    doc = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        # 삭제된 문서는 대상이 아니다 (#166). 목록·exists 조회는 전부 이 필터를 갖고 있는데
        # 여기만 빠져 있어, 목록에 뜨지도 않는 죽은 행의 폴더·참조 컬럼이 200으로 바뀌었다.
        .where(Document.status != 'deleted')
        .where(Document.id == document_id)
    )).scalars().first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")

    # 변경 전 실효 참조 상태 = 문서 on AND (미분류 OR 폴더 on)
    effective_before = doc.is_searchable and await _folder_is_on(session, tenant_id, doc.folder_id)

    # folder_id는 "null 전송 = 미분류 이동"과 "미전송 = 변경 없음"을 구분해야 함
    if 'folder_id' in request.model_fields_set:
        if request.folder_id is not None:
            folder = (await session.execute(
                select(Folder)
                .where(Folder.tenant_id == tenant_id)   # 다른 테넌트 폴더 지정 차단
                .where(Folder.id == request.folder_id)
            )).scalars().first()
            if folder is None:
                raise HTTPException(status_code=404, detail="folder not found")
        doc.folder_id = request.folder_id
    if request.is_searchable is not None:
        doc.is_searchable = request.is_searchable

    # 실효 참조가 on→off로 바뀌는 모든 경로(문서 off, off 폴더로 이동)에서 캐시 무효화 —
    # off된 문서를 근거로 만든 답변이 exact 캐시로 계속 나가는 것 방지. off→on은 무효화할 캐시가 없음.
    effective_after = doc.is_searchable and await _folder_is_on(session, tenant_id, doc.folder_id)
    if effective_before and not effective_after:
        await cache.invalidate_source(session, tenant_id, doc.id)

    # 외부 색인의 비정규화 메타 갱신을 같은 트랜잭션에 적재 (#139) — 검색가능·폴더 값이
    # 청크 문서마다 복사돼 있어 fan-out이 필요하다. **재색인이 아니라 부분 갱신**이다
    # (재색인은 벡터 없는 구성에서 재임베딩을 부른다 — rag/os_index.py의 _update_meta).
    outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=[doc.id])
    await session.commit()
    return _to_response(doc)


@router.delete('/documents', status_code=204)
async def bulk_delete_documents(
        ids: Annotated[list[int] | None, Query(max_length=BULK_MAX_ITEMS)] = None,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """문서 여러 건 삭제 (#174). 실제 처리는 단건과 **같은 경로**(soft_delete_documents)다.

    id 목록을 본문이 아니라 쿼리 파라미터로 받는다 — DELETE 요청의 content는 RFC 9110상
    정의된 의미가 없어 중간 장비가 버릴 여지가 있다. 200건이 1.8KB라 URL 길이도 문제없다.

    없는 id·다른 테넌트 id가 하나라도 있으면 **아무것도 지우지 않고 404**(일괄 변경 #166과 같다).
    반면 **이미 삭제된 문서는 통과시킨다** — 단건 DELETE가 원래 그렇게 동작했고(멱등),
    삭제는 "그 문서가 없는 상태"를 만드는 것이라 이미 그 상태면 실패로 볼 이유가 없다.
    """
    if not ids:
        return                      # 지울 것이 없는 요청은 조회도 하지 않는다

    # 대상 확정 — status 필터를 두지 않는 것이 위 멱등 규칙의 구현점이다.
    # 테넌트 스코프 조회로 id를 먼저 확정한다(격리 규약 2항) — 아래 outbox가 부르는
    # os_index.drop_documents_now는 document_ids에 tenant 필터를 걸지 않는다.
    unique = list(dict.fromkeys(ids))
    found = set((await session.execute(
        select(Document.id)
        .where(Document.tenant_id == tenant_id)
        .where(Document.id.in_(unique))
    )).scalars().all())
    missing = [i for i in unique if i not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f'문서를 찾을 수 없습니다: {missing[:10]}')

    await soft_delete_documents(session, tenant_id, unique)
    await session.commit()


@router.delete('/documents/{document_id}', status_code=204)
async def delete_document(
        document_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
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
        document_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """원본 blob 파일 다운로드. 답변 인용(sources)의 document_id에서 연결된다."""
    doc = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(Document.id == document_id)
    )).scalars().first()
    if doc is None:
        raise HTTPException(status_code=404, detail="document not found")
    blob = Path(doc.blob_path)
    if not blob.is_file():
        raise HTTPException(status_code=410, detail="blob file missing")
    return FileResponse(blob, media_type=doc.mime, filename=doc.filename)


# 채팅 첨부 크기 게이트 (KMS_UX_FEATURES_PLAN.md — 채팅 내 첨부파일)
# 텍스트 상한은 schemas.kms가 단일 정의점 — /kms/query의 QueryAttachment 제약과 같은 값이어야
# 여기서 통과한 결과가 질의에서 거부되지 않는다 (#22)
ATTACHMENT_MAX_FILE_BYTES = 5 * 1024 * 1024   # 1차: 파일 크기 (문서 업로드 10MB와 별개 — 채팅 첨부는
                                              # 매 턴 프롬프트에 재주입되므로 더 좁게, 10MB→5MB #22)


@router.post('/attachments/extract', response_model=QueryAttachment)
async def extract_attachment(request: Request, file: UploadFile = File(...)):
    """채팅 첨부 파일의 텍스트를 추출해 반환한다. 저장하지 않는다 —
    저장은 이후 질의(attachments 동봉) 시점에 대화에 묶여 이뤄진다.
    크기 초과는 자르지 않고 명시 거절(413)한다.
    """
    _reject_if_oversized(request, ATTACHMENT_MAX_FILE_BYTES)   # 거대 본문은 read 전에 차단 (C2-A)
    # 파일명 상한은 마지막 줄의 QueryAttachment 생성에서도 검증되는데, 그 지점의 ValidationError는
    # 요청 파싱이 아니라 핸들러 내부라 422로 변환되지 않고 500이 된다 — 여기서 명시 거절 (#22)
    # 길이는 정규화 후 값으로 재야 스키마 검증(mode='before')과 같은 기준이 된다 (#34) —
    # NFD 한글은 글자당 최대 3코드포인트라 raw 길이로 재면 여기서만 거부되는 파일명이 생긴다.
    filename = normalize_filename(file.filename or '')
    if len(filename) > ATTACHMENT_FILENAME_MAX:
        raise HTTPException(status_code=413,
                            detail=f'파일명이 너무 깁니다 ({ATTACHMENT_FILENAME_MAX}자 이내로 줄여 주세요).')
    content = await file.read()
    if len(content) > ATTACHMENT_MAX_FILE_BYTES:
        # 문구를 상수에서 파생 — 값만 바꾸고 안내가 그대로 남는 드리프트 방지
        raise HTTPException(status_code=413,
                            detail=f'파일이 {ATTACHMENT_MAX_FILE_BYTES // (1024 * 1024)}MB를 초과합니다.')

    suffix = Path(filename).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        try:
            # 텍스트 추출은 동기 CPU 작업(pdfplumber 등) — 이벤트 루프를 막지 않게 스레드풀에서 실행
            text = await run_in_threadpool(extract_text, tmp_path)
        except Exception:
            raise HTTPException(status_code=422, detail='텍스트를 추출할 수 없는 파일입니다.')
    finally:
        tmp_path.unlink(missing_ok=True)

    if len(text) > ATTACHMENT_MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f'문서가 너무 깁니다 (추출 텍스트 {len(text):,}자 > {ATTACHMENT_MAX_TEXT_CHARS:,}자). 필요한 부분만 잘라 첨부해 주세요.',
        )
    return QueryAttachment(filename=filename, text=text)


