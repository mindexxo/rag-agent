"""문서 업로드 저장 — xlsx 사전검증, blob 저장, 낙관적 잠금, handle_upload·커밋, 충돌·실패 시 blob 정리.

업로드·supersede·failed 재사용의 의미는 rag/documents.py의 handle_upload가 정의점이다. HTTP 입력 해석(파일명·크기·
업로드 데이터 파트)과 응답 변환은 라우터가 한다.

순서가 규칙이다: **폴더·xlsx 사전검증은 blob을 쓰기 전에** 끝낸다 — 뒤에 두면 실패 경로마다 unlink를 챙겨야 하고,
파싱은 워커가 하므로 여기서 막지 않으면 "업로드는 200인데 1분 뒤 failed"가 되어 디스크엔 blob이, 목록엔 failed 행이 남는다.
"""

import functools
from pathlib import Path
from typing import NamedTuple
from uuid import uuid4

import anyio
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.chunking.xlsx import XlsxUploadRejected, validate_xlsx_upload
from rag.documents import FailedReuseConflict, document_versions, handle_upload, latest_alive, resolve_description
from rag.models import ALIVE_DOCUMENT_STATUSES, Document
from schemas.kms import DocumentUploadMetadata


class UploadRejected(Exception):
    """결정적 거절(xlsx 표 설명·크기·손상). `str(e)`가 그대로 사용자에게 나간다."""


class VersionConflict(Exception):
    """확인창에서 본 상태와 DB가 달라졌다 — 같은 이름의 다른 업로드가 먼저 들어왔다."""

    def __init__(self, filename: str, current_version: int) -> None:
        super().__init__(filename)
        self.filename = filename
        self.current_version = current_version


class Upload(NamedTuple):
    tenant_id: str
    user_id: str | None  # 등록자 — 헤더 미전송이면 None(uploaded_by=NULL)
    filename: str  # 경계에서 NFC로 정규화된 값
    content: bytes
    meta: DocumentUploadMetadata
    folder_given: bool  # folder_id를 실제로 보냈는지(null 전송과 미전송 구분)


async def current_version(session: AsyncSession, tenant_id: str, filename: str) -> int:
    """해당 파일명의 현재 버전. 없으면 0.

    판정 기준은 exists API·handle_upload의 재사용 판정과 **정확히 같아야** 한다 — tenant + filename
    완전 일치, status가 살아 있는 것(ALIVE_DOCUMENT_STATUSES — 세 곳이 같은 상수를 쓴다).
    기준이 어긋나면 "확인창에서 본 것과 다른 문서가 대체되는" 사고가 난다. failed를 빼는 이유:
    그 문서는 검색에 없고 재업로드가 그 행을 되살리므로(version 유지) "없는 문서"다.
    """
    return (
        await session.execute(
            select(func.max(Document.version))
            .where(Document.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
            .where(Document.filename == filename)
            .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
        )
    ).scalar() or 0


async def _xlsx_precheck(session: AsyncSession, up: Upload) -> list[Document]:
    """xlsx 표 설명 필수 + 표 크기 상한. 읽은 버전 목록은 handle_upload에 넘겨 재조회를 막는다.

    설명 필수는 "이번 요청에 보냈는가"가 아니라 **계승까지 따진 최종값**으로 판정한다(계승 허용) — 계승 출처·식은
    handle_upload와 같은 latest_alive·resolve_description.
    """
    docs = await document_versions(session, up.tenant_id, up.filename)
    final_description = resolve_description(up.meta.description, latest_alive(docs))
    try:
        # 워크북 열기는 CPU 작업 — 이벤트 루프를 막지 않게 스레드로(첨부 추출과 같은 규율).
        await anyio.to_thread.run_sync(functools.partial(validate_xlsx_upload, up.content, final_description))
    except XlsxUploadRejected as e:
        raise UploadRejected(str(e)) from e
    except Exception as e:
        # openpyxl이 못 여는 파일(손상·확장자 위장) — 안 잡으면 500이 된다. openpyxl은 손상 종류마다
        # 다른 예외를 낸다(BadZipFile·KeyError·ParseError·IOError — 3.1.5 기준). 나열하지 않고 넓게 잡는다.
        raise UploadRejected('xlsx 파일을 열 수 없습니다 (손상되었거나 올바른 xlsx 형식이 아닙니다).') from e
    return docs


def _write_blob(up: Upload, suffix: str) -> Path:
    """테넌트별 디렉터리 아래 UUID 파일명으로 저장한다. 업로드 1건 = 파일 1개 — 내용이 같아도 경로를 공유하지 않는다
    (내용 해시를 쓰면 v1·v2가 같은 파일을 가리켜, 비활성 문서 blob을 정리할 때 살아 있는 문서의 원본까지 지워질 수 있다)."""
    blob_dir = Path(settings.blob_storage_dir) / up.tenant_id
    blob_dir.mkdir(parents=True, exist_ok=True)
    blob_path = blob_dir / f"{uuid4().hex}{suffix}"
    blob_path.write_bytes(up.content)
    return blob_path


async def store_upload(session: AsyncSession, up: Upload) -> Document:
    """사전검증 → blob 저장 → 낙관적 잠금 → handle_upload·커밋. 폴더 검증·크기 상한은 호출부가 이미 했다.

    거절은 UploadRejected, 버전 충돌은 VersionConflict — 둘 다 새 blob을 남기지 않는다. 색인 대기열 행은
    handle_upload가 문서와 같은 트랜잭션에 등록한다(응답 시점 status는 pending, 검색 반영은 최대 1~5분).
    """
    suffix = Path(up.filename).suffix.lower()
    docs = (
        await _xlsx_precheck(session, up) if suffix == '.xlsx' else None
    )  # xlsx가 아니면 추가 쿼리도 워크북 open도 없다
    blob_path = _write_blob(up, suffix)

    # 낙관적 잠금 — 확인창에서 본 상태와 지금 DB가 같은지 본다.
    # 이 조회만으로는 조회~insert 사이 창이 남는다. 그 창은 아래 IntegrityError가 닫는다.
    if up.meta.expect_version is not None:
        current = await current_version(session, up.tenant_id, up.filename)
        if current != up.meta.expect_version:
            blob_path.unlink(missing_ok=True)  # 참조되지 않는 blob 남기지 않기
            raise VersionConflict(up.filename, current)

    try:
        result = await handle_upload(
            session,
            up.tenant_id,
            up.filename,
            blob_path,
            description=up.meta.description,
            uploaded_by=up.user_id,
            folder_id=up.meta.folder_id,
            folder_given=up.folder_given,
            docs=docs,  # xlsx 사전검증이 이미 읽었으면 재사용 — 아니면 None이라 안에서 읽는다
        )
        await session.commit()
    except IntegrityError, FailedReuseConflict:
        # IntegrityError: 같은 이름·같은 version이 방금 먼저 들어왔다 — UNIQUE(tenant_id, filename,
        # version). 위 조회를 두 요청이 함께 통과했을 때의 최종 방어선(expect_version 미전송 호출도 여기서 409).
        # FailedReuseConflict: 같은 failed를 둘이 동시에 되살리려 했고 이쪽이 졌다.
        # 둘 다 새 blob만 지운다 — 옛 blob은 DB가 여전히 가리키고 있다.
        await session.rollback()
        blob_path.unlink(missing_ok=True)
        raise VersionConflict(up.filename, await current_version(session, up.tenant_id, up.filename)) from None
    # failed 행을 되살린 경우 옛 blob은 **커밋 뒤에** 지운다 — 커밋 전에 지우면 롤백 시
    # DB는 옛 경로를 가리키는데 파일은 없다. failed는 인용된 적이 없어 보존할 이유도 없다.
    if result.stale_blob:
        Path(result.stale_blob).unlink(missing_ok=True)
    return result.document
