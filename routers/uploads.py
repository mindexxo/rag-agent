"""업로드·첨부 공용 입력 검사 — 크기 코스 가드와 파일명(형식·길이·NFC)."""

from pathlib import Path

from fastapi import HTTPException, Request, UploadFile

from schemas.kms import ATTACHMENT_FILENAME_MAX
from text_norm import normalize_filename

# 지원 형식 화이트리스트(확장자 기준). 그 외는 400.
SUPPORTED_SUFFIXES = {'.pdf', '.docx', '.xlsx', '.txt', '.md'}


def reject_if_oversized(request: Request, limit: int) -> None:
    """Content-Length로 명백한 초과 업로드를 body read 전에 차단하는 코스 가드.
    조작·multipart 오버헤드로 부정확할 수 있어 정확 경계는 read 후 len() 검사가 담당 —
    여기선 여유(margin)를 두고 '거대 본문을 메모리에 올리기 전에 끊는' 용도. (실전 상한은 nginx도 병행)"""
    cl = request.headers.get('content-length')
    if cl and cl.isdigit() and int(cl) > limit + 8192:
        raise HTTPException(status_code=413, detail='파일이 너무 큽니다.')


def checked_filename(file: UploadFile) -> str:
    """업로드·첨부 공용 파일명 검사 — 누락·미지원 형식 400, 상한 초과 422. NFC 정규화 값을 돌려준다.

    길이는 정규화 후 값으로 잰다 — NFD 한글은 글자당 최대 3코드포인트라 raw 길이로 재면 스키마(mode='before')와
    기준이 어긋난다. 경로 문자는 정제하지 않는다(식별이 파일명 완전 일치, blob은 UUID 이름).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail='파일명이 없습니다.')
    filename = normalize_filename(file.filename)
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400, detail=f'지원하지 않는 형식입니다: {suffix or "확장자 없음"}. 지원: PDF/DOCX/XLSX/TXT/MD'
        )
    if len(filename) > ATTACHMENT_FILENAME_MAX:
        raise HTTPException(
            status_code=422, detail=f'파일명이 너무 깁니다 ({ATTACHMENT_FILENAME_MAX}자 이내로 줄여 주세요).'
        )
    return filename
