"""채팅 첨부 추출 라우터 — 파일에서 텍스트만 뽑아 돌려준다. 저장하지 않는다.

저장은 이후 질의(`/kms/query`의 attachments)가 대화에 묶어 한다. 형식·파일명 규칙은 문서 업로드와 같고
(routers/documents.py의 checked_filename), 텍스트 상한은 schemas.kms의 QueryAttachment와 같은 값이어야 여기서
통과한 결과가 질의에서 거부되지 않는다.
"""

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from rag.chunking import extract_text
from routers.documents import checked_filename, reject_if_oversized
from routers.kms import get_tenant_id
from schemas.kms import ATTACHMENT_MAX_TEXT_CHARS, QueryAttachment

router = APIRouter(prefix='/kms')

# 문서 업로드(10MB)보다 좁다 — 첨부는 매 턴 프롬프트에 재주입된다.
ATTACHMENT_MAX_FILE_BYTES = 5 * 1024 * 1024
UNREADABLE = '텍스트를 추출할 수 없는 파일입니다.'


@router.post('/attachments/extract', response_model=QueryAttachment, dependencies=[Depends(get_tenant_id)])
async def extract_attachment(request: Request, file: UploadFile = File(...)) -> QueryAttachment:
    """크기 초과는 자르지 않고 413, 읽을 수 없거나 텍스트가 없는 파일은 422."""
    reject_if_oversized(request, ATTACHMENT_MAX_FILE_BYTES)
    filename = checked_filename(file)
    content = await file.read()
    if len(content) > ATTACHMENT_MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413, detail=f'파일이 {ATTACHMENT_MAX_FILE_BYTES // (1024 * 1024)}MB를 초과합니다.'
        )

    with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix.lower(), delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        try:
            # 동기 CPU 작업(pdfplumber 등) — 이벤트 루프를 막지 않게 스레드풀에서
            text = await run_in_threadpool(extract_text, tmp_path)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=UNREADABLE) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    if not text:
        raise HTTPException(status_code=422, detail=UNREADABLE)
    if len(text) > ATTACHMENT_MAX_TEXT_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f'문서가 너무 깁니다 (추출 텍스트 {len(text):,}자 > {ATTACHMENT_MAX_TEXT_CHARS:,}자). '
            '필요한 부분만 잘라 첨부해 주세요.',
        )
    return QueryAttachment(filename=filename, text=text)
