"""KMS 문서 업로드 라우터.

POST /kms/documents (multipart)

문서 식별·버전 정책 (2026-08-05 확정):
- 식별 기준은 **filename 완전 일치** 하나뿐. 유사 파일명은 별개 문서로 본다.
- 같은 이름 재업로드 = 새 version + 기존 버전 supersede(비활성화 + 청크 삭제 + 근거 캐시 무효화).
  내용이 같아도 마찬가지 — 내용 해시 dedupe는 제거했다(규칙을 하나로 유지).
- 원본 파일은 테넌트 디렉터리 아래 UUID 이름으로 저장 (업로드 1건 = 파일 1개).
- 버전 롤백은 미지원. 되돌리려면 이전 파일을 다시 업로드한다.
"""
import tempfile
from uuid import uuid4
from datetime import timedelta
from pathlib import Path

from arq.connections import RedisSettings, create_pool
from fastapi import APIRouter, File, Form, Request, UploadFile, Depends, HTTPException
from sqlalchemy import delete, func, select, update
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse

from config import settings
from database import get_session
from rag.cache import AnswerCache
from rag.chunking import extract_text
from rag.documents import handle_upload
from rag.ingestion import _detect_mime
from rag.models import Chunk, Document, Folder
from routers.kms import get_tenant_id
from schemas.kms import DocumentExistsResponse, DocumentUploadResponse, DocumentUpdateRequest, QueryAttachment


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
        ref_count=ref_count,
    )

router = APIRouter(prefix='/kms')

# F1a: 지원 형식 화이트리스트 (확장자 기준). 그 외는 400.
SUPPORTED_SUFFIXES = {'.pdf', '.docx', '.xlsx', '.txt', '.md'}
DOC_MAX_FILE_BYTES = 10 * 1024 * 1024   # 문서 업로드 크기 상한 (첨부와 별개)
DOC_STALE_SECONDS = 900                 # pending이 이보다 오래면 워커 타임아웃/중단으로 간주 → failed (job_timeout 600보다 길게)


def _reject_if_oversized(request: Request, limit: int) -> None:
    """Content-Length로 명백한 초과 업로드를 body read 전에 차단 (C2-A, 코스 가드).
    조작·multipart 오버헤드로 부정확할 수 있어 정확 경계는 read 후 len() 검사가 담당 —
    여기선 여유(margin)를 두고 '거대 본문을 메모리에 올리기 전에 끊는' 용도. (실전 상한은 nginx도 병행)"""
    cl = request.headers.get('content-length')
    if cl and cl.isdigit() and int(cl) > limit + 8192:
        raise HTTPException(status_code=413, detail='파일이 너무 큽니다.')

@router.post('/documents', response_model=DocumentUploadResponse)
async def upload_document(
        request: Request,
        file: UploadFile = File(...),
        description: str | None = Form(None),   # F1a: 표 설명 (xlsx 검색 보강). 선택
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    # 0. 형식·크기 게이트 (인덱싱 전에 명시적으로 막음)
    _reject_if_oversized(request, DOC_MAX_FILE_BYTES)   # 거대 본문은 read 전에 차단 (C2-A)
    if not file.filename:   # multipart에 filename 누락 시 Path(None) TypeError→500 방지 (C2)
        raise HTTPException(status_code=400, detail='파일명이 없습니다.')
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(status_code=400, detail=f'지원하지 않는 형식입니다: {suffix or "확장자 없음"}. 지원: PDF/DOCX/XLSX/TXT/MD')

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

    # 3. 파일 인덱싱 후 저장
    mime = _detect_mime(blob_path)
    doc = await handle_upload(
        session, tenant_id, file.filename, mime, blob_path, description=description
    )
    await session.commit()

    #4. pending이면 워커에 인덱싱 작업 등록.
    if doc.status == 'pending':
        try:
            pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            try:
                await pool.enqueue_job('index_document', doc.id)
            finally:
                await pool.aclose()
        except Exception as e:
            # enqueue 실패(Redis 순단 등) → 문서가 pending 영구 고착하지 않게 failed 기록 (P1-4)
            doc.status = 'failed'
            doc.status_reason = f'인덱싱 큐 등록 실패: {e}'[:500]
            await session.commit()

    return _to_response(doc)

@router.get('/documents', response_model=list[DocumentUploadResponse])
async def list_documents(
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """테넌트의 문서 목록. 최신 업로드가 위로 오도록 id 내림차순.
    supersede된 구버전(deleted)은 제외 — 죽은 행에 폴더/참조 컨트롤이 노출되는 혼란 방지."""
    # lazy 스윕: 오래 고착된 pending을 failed로 자기치유 (enqueue 실패·워커 타임아웃/크래시 백스톱).
    # 정상 인제스트는 job_timeout(600s) 내 끝나므로 900s 넘게 pending이면 중단으로 간주. 서버측 시간 비교.
    res = await session.execute(
        update(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.status == 'pending')
        .where(Document.uploaded_at < func.now() - timedelta(seconds=DOC_STALE_SECONDS))
        .values(status='failed', status_reason='인덱싱 시간 초과 또는 워커 중단')
    )
    if res.rowcount:
        await session.commit()

    docs = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(Document.status != 'deleted')
        .order_by(Document.id.desc())
    )).scalars().all()

    # 인용 횟수: 저장 시 확정된 실인용 목록(cited_docs)을 filename별 집계 (F5).
    # sources(검색 후보 노출 수)가 아닌 실인용 — stats top_documents와 정의 통일.
    # filename 키라 버전 교체 후에도 카운트가 이어진다.
    ref_rows = (await session.execute(sql_text("""
        SELECT d AS filename, count(*) AS cnt
        FROM messages, jsonb_array_elements_text(messages.cited_docs) AS d
        WHERE messages.tenant_id = :tenant_id
          AND messages.role = 'assistant'
          AND jsonb_typeof(messages.cited_docs) = 'array'
        GROUP BY 1
    """), {"tenant_id": tenant_id})).all()
    ref_counts = {r.filename: r.cnt for r in ref_rows}

    return [_to_response(d, ref_counts.get(d.filename, 0)) for d in docs]


# ⚠ 이 라우트는 반드시 '/documents/{document_id}'보다 **위에** 있어야 한다.
#   FastAPI는 등록 순서대로 매칭하므로, 아래에 두면 'exists'가 document_id(int)로 파싱돼 422가 난다.
@router.get('/documents/exists', response_model=DocumentExistsResponse)
async def document_exists(
        filename: str,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """업로드 전 동일 파일명 확인 (FE가 대체 확인 창을 띄울지 판단).

    판정 기준은 supersede 로직과 **정확히 같아야** 한다 — tenant + filename **완전 일치**
    (대소문자·공백 구분), status != 'deleted'. 기준이 어긋나면 "물어본 것과 다른 문서가
    지워지는" 사고가 난다.

    이 API는 안내용일 뿐 강제력이 없다. 업로드 API는 확인 없이도 통과하며(2026-08-05 결정),
    그 경우 기존 버전이 그대로 대체된다.
    """
    doc = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)      # 격리 — WHERE 절 명시
        .where(Document.filename == filename)
        .where(Document.status != 'deleted')
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
        await AnswerCache().invalidate_document(session, tenant_id, doc.id)

    await session.commit()
    return _to_response(doc)


@router.delete('/documents/{document_id}', status_code=204)
async def delete_document(
        document_id: int,
        tenant_id: str = Depends(get_tenant_id),
        session: AsyncSession = Depends(get_session)
):
    """문서 소프트 삭제 (F5). 같은 filename의 전 버전을 비활성 처리한다.

    - status='deleted' + is_active=False → 검색 즉시 제외 + 목록에서 사라짐 (supersede와 동일 메커니즘)
    - 청크 삭제 → 인덱스에서 제거
    - 캐시 무효화 → 이 문서를 근거로 만든 답변 재사용 방지
    - documents row·blob은 보존 → 과거 대화 인용의 원본 다운로드 유지
    """
    # id → filename은 서브쿼리로 풀어 UPDATE...RETURNING 한 문장으로 처리.
    # 대상 문서가 없으면 서브쿼리가 NULL → 매칭 0건 → 404.
    filename_sq = (
        select(Document.filename)
        .where(Document.tenant_id == tenant_id)   # 격리 — WHERE 절 명시
        .where(Document.id == document_id)
        .scalar_subquery()
    )

    doc_ids = (await session.execute(
        update(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.filename == filename_sq)
        .values(is_active=False, status='deleted', status_reason='user_deleted')
        .returning(Document.id)
    )).scalars().all()

    if not doc_ids:
        raise HTTPException(status_code=404, detail="document not found")

    await session.execute(delete(Chunk).where(Chunk.document_id.in_(doc_ids)))

    cache = AnswerCache()
    for did in doc_ids:
        await cache.invalidate_document(session, tenant_id, did)

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
ATTACHMENT_MAX_FILE_BYTES = 10 * 1024 * 1024   # 1차: 파일 크기
ATTACHMENT_MAX_TEXT_CHARS = 6000               # 개당 추출 텍스트 상한 (~3-4페이지, ≈4K토큰). 누적 max_attachments개까지 주입


@router.post('/attachments/extract', response_model=QueryAttachment)
async def extract_attachment(request: Request, file: UploadFile = File(...)):
    """채팅 첨부 파일의 텍스트를 추출해 반환한다. 저장하지 않는다 —
    저장은 이후 질의(attachments 동봉) 시점에 대화에 묶여 이뤄진다.
    크기 초과는 자르지 않고 명시 거절(413)한다.
    """
    _reject_if_oversized(request, ATTACHMENT_MAX_FILE_BYTES)   # 거대 본문은 read 전에 차단 (C2-A)
    content = await file.read()
    if len(content) > ATTACHMENT_MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail='파일이 10MB를 초과합니다.')

    suffix = Path(file.filename).suffix.lower()
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
    return QueryAttachment(filename=file.filename, text=text)


