"""문서 도메인 — 업로드·삭제·버전. 인제스션(pending → ready)은 rag/ingestion.py.

라우터도 워커도 import하므로 starlette를 들이지 않는다 — HTTPException 대신 도메인 예외나 None을
돌려주고 HTTP 변환은 라우터가 한다.
"""

import mimetypes
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag import cache, outbox
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Folder

_MIME_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _detect_mime(blob_path: Path) -> str:
    """확장자로 mime 추론. `mimetypes`가 docx를 못 알아보는 환경이 있어 override를 먼저 본다.

    라우터의 SUPPORTED_SUFFIXES가 전부 여기서 판별돼야 업로드가 500 없이 끝난다.
    """
    suffix = blob_path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(str(blob_path))
    if not mime:
        raise ValueError(f"mime 추론 실패: {blob_path}")
    return mime


async def get_folder(session: AsyncSession, tenant_id: str, folder_id: int) -> Folder | None:
    """테넌트의 폴더 하나. 없거나 **남의 테넌트 것이면 None** — 폴더 조회의 정의점.

    테넌트 WHERE가 곧 격리라 사본을 만들지 말고 이 함수를 쓴다. 404 변환은 호출부가 한다.
    """
    return (
        (
            await session.execute(
                select(Folder)
                .where(Folder.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
                .where(Folder.id == folder_id)
            )
        )
        .scalars()
        .first()
    )


async def retire_documents(
    session: AsyncSession, tenant_id: str, document_ids: list[int], *, status_reason: str | None = None
) -> None:
    """문서들을 내린다 — 상태·캐시·DROP **세 동작이 한 묶음**이다(하나만 빠져도 조용히 어긋난다).

    **커밋하지 않는다** — 호출부 트랜잭션에 얹힌다. flush는 한다: supersede는 옛 active가 먼저
    off로 반영돼야 신버전을 켤 때 uq_docs_one_active_per_name을 안 밟는다.
    """
    if not document_ids:
        return
    values = {'is_active': False, 'status': 'deleted'}
    if status_reason is not None:
        values['status_reason'] = status_reason
    await session.execute(update(Document).where(Document.id.in_(document_ids)).values(**values))
    for did in document_ids:
        await cache.invalidate_source(session, tenant_id, did)
    outbox.enqueue(session, tenant_id, outbox.DROP_DOCUMENTS, document_ids=list(document_ids))
    await session.flush()


async def soft_delete_documents(session: AsyncSession, tenant_id: str, document_ids: list[int]) -> list[int]:
    """문서 소프트 삭제 — 단건·일괄 공용. 같은 filename의 **전 버전**을 상태·캐시·DROP으로 함께 내린다.

    행·blob은 보존한다(과거 인용의 원본 다운로드). 반환은 실제로 내려간 id들(요청보다 많을 수 있다).
    **커밋하지 않는다** — 호출부 트랜잭션에 얹혀야 "지웠는데 색인엔 남는" 상태가 안 생긴다.
    이미 deleted인 문서도 성공으로 본다(status 필터 없음은 의도).
    """
    if not document_ids:
        return []
    # id → filename을 서브쿼리로 풀어 한 문장에 묶는다 — 따로 SELECT하면 그 사이 새 버전이 들어올 창이 생긴다.
    filenames = (
        select(Document.filename)
        .where(Document.tenant_id == tenant_id)  # 격리 — WHERE 절 명시
        .where(Document.id.in_(document_ids))
    )
    doc_ids = (
        (
            await session.execute(
                update(Document)
                .where(Document.tenant_id == tenant_id)  # 격리 — 빠지면 같은 파일명의 남의 테넌트 문서까지 지워진다
                .where(Document.filename.in_(filenames))
                .values(is_active=False, status='deleted', status_reason='user_deleted')
                .returning(Document.id)
            )
        )
        .scalars()
        .all()
    )
    if not doc_ids:
        return []

    for did in doc_ids:
        await cache.invalidate_source(session, tenant_id, did)
    outbox.enqueue(session, tenant_id, outbox.DROP_DOCUMENTS, document_ids=list(doc_ids))
    return list(doc_ids)


class FailedReuseConflict(Exception):
    """failed 행을 되살리려는 순간 다른 요청이 먼저 되살렸다. 라우터가 409로 바꾼다."""


async def _reuse_failed_row(session: AsyncSession, target_id: int, **values) -> int | None:
    """failed 행 하나를 pending으로 되살리는 조건부 UPDATE — 동시 재업로드에 진 쪽은 0행이라 None.

    별도 함수로 둘 것 — 테스트가 이 경합을 monkeypatch로 재현한다.
    """
    return (
        await session.execute(
            update(Document)
            .where(Document.id == target_id)
            .where(Document.status == 'failed')
            .values(**values)
            .returning(Document.id)
        )
    ).scalar()


class UploadResult(NamedTuple):
    document: Document
    stale_blob: str | None  # 되살린 failed 행의 옛 blob — 호출부가 커밋 뒤에 지운다


async def _reuse_failed_document(
    session: AsyncSession,
    tenant_id: str,
    filename: str,
    blob_path: Path,
    failed: list[Document],
    *,
    description: str | None,
    uploaded_by: str | None,
    folder_id: int | None,
    folder_given: bool,
) -> UploadResult:
    """failed만 있는 파일명의 재업로드 — 최신 failed를 되살리고 나머지는 내린다(이유는 handle_upload).

    반환은 handle_upload와 같은 (문서, 지워야 할 옛 blob) — 옛 blob은 호출부가 **커밋 뒤에** 지운다.
    """
    target = max(failed, key=lambda d: d.version)
    stale_blob = target.blob_path
    reused = await _reuse_failed_row(
        session,
        target.id,
        status='pending',
        is_active=False,
        # 워커 ③은 status_reason을 이미지 경고일 때만 덮어쓰므로 옛 실패 사유를 여기서 비운다.
        status_reason=None,
        char_count=None,
        indexed_at=None,
        blob_path=str(blob_path),
        mime=_detect_mime(blob_path),
        # server_default는 INSERT에만 적용된다 — 갱신하지 않으면 목록(uploaded_at 내림차순)에서 위로 안 온다.
        uploaded_at=func.now(),
        uploaded_by=uploaded_by,
        # 설정은 그 행의 값을 유지(= 계승), 보냈으면 그 값.
        folder_id=(folder_id if folder_given else target.folder_id),
        description=resolve_description(description, target),
    )
    if reused is None:
        raise FailedReuseConflict(filename)
    # 같은 id를 다른 내용으로 되살리므로 자기 캐시도 지운다 — 워커 ③은 구버전만 무효화한다.
    await cache.invalidate_source(session, tenant_id, target.id)
    # 나머지 failed는 내린다 — soft_delete_documents는 filename 기준이라 target까지 지우므로 쓰지 않는다.
    # failed라도 엔진에 청크가 남을 수 있어 DROP이 필요하다(failed 확정 DROP의 이중 안전판).
    others = [d.id for d in failed if d.id != target.id]
    await retire_documents(session, tenant_id, others, status_reason='failed_superseded')
    await session.refresh(target)  # UPDATE로 바뀐 값을 ORM 객체에 반영(응답에 쓴다)
    outbox.enqueue(session, tenant_id, outbox.INDEX_DOCUMENT, document_id=target.id)
    return UploadResult(target, stale_blob)


async def document_versions(session: AsyncSession, tenant_id: str, filename: str) -> list[Document]:
    """같은 filename의 **모든** 버전 행(deleted·failed 포함) — handle_upload와 라우터 사전검증이 공유한다.

    status를 거르지 않는 것은 의도다 — next_version은 전체 최대값+1이어야 UNIQUE 위반이 없다.
    계승 출처는 별개 질문이고 `latest_alive`가 답한다.
    """
    return list(
        (
            await session.execute(
                select(Document).where(Document.tenant_id == tenant_id).where(Document.filename == filename)
            )
        )
        .scalars()
        .all()
    )


def latest_alive(docs: list[Document]) -> Document | None:
    """`docs` 중 살아 있는 최신 버전 — **설정 계승 출처의 정의점**.

    ALIVE만 본다 — deleted·failed는 사용자에게 "없는 문서"라, 그 위의 업로드는 신규 등록이다.
    `next_version`(전체 최대값+1)과 혼동하지 마라.
    """
    alive = [d for d in docs if d.status in ALIVE_DOCUMENT_STATUSES]
    return max(alive, key=lambda d: d.version) if alive else None


def resolve_description(description: str | None, source: Document | None) -> str | None:
    """표 설명의 최종값 — 보낸 값, 없으면 `source`에서 계승, 그것도 없으면 None.

    업로드·failed 재사용·라우터 xlsx 사전판정이 이 식 하나를 쓴다 — 갈리면 사전판정과 저장값이 어긋난다.
    """
    return description if description is not None else (source.description if source else None)


async def handle_upload(
    session: AsyncSession,
    tenant_id: str,
    filename: str,
    blob_path: Path,
    description: str | None = None,
    uploaded_by: str | None = None,
    folder_id: int | None = None,
    folder_given: bool = False,
    docs: list[Document] | None = None,
) -> UploadResult:
    """업로드 시점 처리: pending 행 + INDEX_DOCUMENT 대기열 행을 **같은 트랜잭션**에 등록한다(커밋은 호출부).

    반환은 (문서, 지워야 할 옛 blob 또는 None) — 옛 blob은 **호출부가 커밋 뒤에** 지운다(롤백 시 파일 소실 방지).
    folder_given=False(미전송)면 폴더를 계승하고, True면 folder_id를 그대로 쓴다(None = 미분류).
    docs는 호출부가 이미 읽은 전 버전 목록(선택) — 같은 쿼리를 두 번 돌지 않게.

    문서 식별은 filename 완전 일치 하나뿐이다(내용이 같아도 새 version). 예외로 정상 행 없이 failed만
    있으면 새 version 대신 그 행을 되살린다 — 실패한 시도가 개정 번호를 소비하지 않게.
    """
    if docs is None:
        docs = await document_versions(session, tenant_id, filename)

    # failed 재사용 — 정상 행이 없고 failed만 있을 때(deleted 이력이 섞여 있어도 된다).
    prev = latest_alive(docs)
    failed = [d for d in docs if d.status == 'failed']
    if prev is None and failed:
        return await _reuse_failed_document(
            session,
            tenant_id,
            filename,
            blob_path,
            failed,
            description=description,
            uploaded_by=uploaded_by,
            folder_id=folder_id,
            folder_given=folder_given,
        )

    # 설정(폴더·검색 on/off·표 설명)은 살아 있는 직전 버전에서 계승하고, 번호는 전체 최대값+1이다.
    next_version = max((d.version for d in docs), default=0) + 1
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        mime=_detect_mime(blob_path),
        blob_path=str(blob_path),
        version=next_version,
        is_active=False,
        status="pending",
        folder_id=(folder_id if folder_given else (prev.folder_id if prev else None)),
        is_searchable=prev.is_searchable if prev else True,
        description=resolve_description(description, prev),
        # 등록자는 계승하지 않는다 — 문서 설정이 아니라 "이 버전을 올린 사람"이다.
        uploaded_by=uploaded_by,
    )
    session.add(doc)
    await session.flush()  # doc.id 확보(대기열 payload에 필요)
    outbox.enqueue(session, tenant_id, outbox.INDEX_DOCUMENT, document_id=doc.id)
    return UploadResult(doc, None)
