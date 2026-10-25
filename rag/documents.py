"""문서 도메인 — 업로드·인제스션·삭제. 라우터(routers/documents.py)와 워커가 함께 쓴다.

라우터도 워커도 import하므로 starlette를 들이지 않는다 — HTTPException 대신 도메인 예외나 None을
돌려주고 HTTP 변환은 라우터가 한다. 스레드 오프로딩도 stdlib `asyncio.to_thread`를 쓴다.
"""

import asyncio
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from rag import cache, os_index, outbox
from rag.chunking import abort_conversion, chunk_file, count_picture_placeholders, pdf_image_area_ratio
from rag.embeddings import embed_texts  # 이름으로 들일 것 — tests/conftest.py가 rag.documents.embed_texts를 패치
from rag.index_text import build_index_text
from rag.metrics import (
    INDEX_DURATION_SECONDS,
    INDEX_PICTURE_PLACEHOLDER_TOTAL,
    INDEX_TOTAL,
    ext_label,
)
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Folder

# 이미지 면적(%)이 이 값 이상이면 문서에 경고를 남긴다 — 작은 도표 하나에도 핵심 내용이 들 수 있어 낮게 잡는다.
IMAGE_WARN_RATIO = 0.5

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


async def _retire_documents(
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


async def index_pending_document(document_id: int, *, outbox_row_id: int | None = None) -> None:
    """INDEX_DOCUMENT 핸들러 — 인제스션 전체. 실패는 예외로 올린다(재시도·failed 판정은 outbox.drain).

    커밋은 ③의 한 번뿐이고 ready·supersede·캐시 무효화·outbox done이 함께 들어가 `ready ≡ 색인됨`이
    커밋 단위로 성립한다. 어디서 죽어도 다음 회차가 처음부터 다시 한다(엔진 색인은 같은 _id upsert라 멱등).
    **파괴적 엔진 변경은 커밋 뒤 outbox로만** — 구버전 청크 삭제는 ③에 DROP 행으로 얹어, ③이 실패하면
    구버전이 PG·엔진 모두 그대로 남게 한다.

    세션을 셋으로 나눠 무거운 파싱·임베딩 동안 DB 커넥션을 물지 않는다.
      ① 짧은 읽기   처리 대상 확인 + 파싱·색인에 필요한 값만
      ② DB 없이     파싱·청킹·임베딩 → 엔진 색인(삭제 없음)
      ③ 짧은 쓰기   유일한 커밋
    """
    # ── ① 짧은 읽기 ──
    async with AsyncSessionLocal() as session:
        row = (
            await session.execute(
                select(Document, Folder.name, Folder.description, Folder.is_searchable)
                .outerjoin(Folder, Document.folder_id == Folder.id)
                .where(Document.id == document_id)
            )
        ).first()
        if row is None or row[0].status != 'pending':  # 이미 처리됨·삭제됨 — 할 일 없음
            if outbox_row_id is not None:
                await outbox.mark_done(session, outbox_row_id)
                await session.commit()
            return
        doc, folder_name, folder_desc, folder_searchable = row
        tenant_id, filename, version = doc.tenant_id, doc.filename, doc.version
        blob_path, description = doc.blob_path, doc.description or ''
        folder_id, doc_searchable = doc.folder_id, doc.is_searchable
        # supersede 대상 — 같은 filename의 **낮은 version** 중 살아 있는 것(pending 포함). outbox 처리 순서는
        # 보장되지 않으므로 늦게 도는 구버전이 신버전을 내리지 못하게 version 조건을 건다.
        old_ids = list(
            (
                await session.execute(
                    select(Document.id)
                    .where(Document.tenant_id == tenant_id)
                    .where(Document.filename == filename)
                    .where(Document.version < version)
                    .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
                )
            )
            .scalars()
            .all()
        )

    # ── ② 무거운 계산 — DB 세션 없음 ──
    # 단계별 소요 지표는 성공한 건만 기록한다(실패는 outbox.drain이 센다).
    _ext = ext_label(filename)
    _t_start = time.monotonic()
    # 청킹은 동기 CPU 작업이라 스레드로 보낸다. 취소되면 변환 프로세스를 즉시 죽인다 — to_thread는 취소돼도
    # 스레드가 계속 돌며 변환 잠금을 쥐어, 다음 회차까지 잠금에 막혀 연쇄로 취소된다.
    try:
        chunks = await asyncio.to_thread(chunk_file, blob_path, description=description)
    except asyncio.CancelledError:
        abort_conversion()
        raise
    image_ratio = await asyncio.to_thread(pdf_image_area_ratio, blob_path)
    if not chunks:  # 빈 파일·텍스트레이어 없는 PDF → ready 승격 대신 실패 (유령 ready 방지)
        raise ValueError('추출된 텍스트가 없습니다 (빈 파일이거나 파싱 결과가 비어 있음)')
    # 단계별 지표가 같은 분모(성공한 문서)를 갖도록 실패 분기 뒤에서 기록한다.
    INDEX_DURATION_SECONDS.labels(stage='parse', ext=_ext).observe(time.monotonic() - _t_start)
    # 캡션이 안 붙은 그림 수(면적 하한 미만 등) — 캡션 실패는 변환에서 예외로 끝나 여기까지 오지 않는다.
    if placeholders := count_picture_placeholders(chunks):
        INDEX_PICTURE_PLACEHOLDER_TOTAL.labels(ext=_ext).inc(placeholders)
    _t_parse_done = time.monotonic()
    # 임베딩 입력에만 '파일명 > 헤딩' 컨텍스트를 얹는다(저장되는 chunk.text는 원문 유지).
    index_texts = [build_index_text(c.text, filename, c.heading_path) for c in chunks]
    embeddings = await embed_texts(index_texts)
    INDEX_DURATION_SECONDS.labels(stage='embed', ext=_ext).observe(time.monotonic() - _t_parse_done)
    _t_embed_done = time.monotonic()

    # searchable은 PG의 현재 상태(pending·inactive)가 아니라 ③이 만들 상태(ready·active)로 계산한다.
    indexed_searchable = os_index.effective_searchable(
        is_faq=False,
        doc_is_active=True,
        doc_status='ready',
        doc_is_searchable=doc_searchable,
        folder_is_searchable=folder_searchable,
    )
    await os_index.index_parsed_document(
        document_id=document_id,
        tenant_id=tenant_id,
        filename=filename,
        version=version,
        folder_id=folder_id,
        folder_name=folder_name,
        folder_description=folder_desc,
        searchable=indexed_searchable,
        chunks=chunks,
        embeddings=embeddings,
    )
    # 구버전 청크는 여기서 지우지 않는다 — ③의 DROP 행이 지운다.
    INDEX_DURATION_SECONDS.labels(stage='index', ext=_ext).observe(time.monotonic() - _t_embed_done)

    # ── ③ 유일한 커밋 ──
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None or doc.status != 'pending':
            # ②가 도는 사이 삭제됐다. 엔진에 넣은 청크는 그 삭제가 만든 DROP 행이 다음 회차에 지운다.
            if outbox_row_id is not None:
                await outbox.mark_done(session, outbox_row_id)
                await session.commit()
            return

        # 청크는 PG에 쓰지 않는다(엔진이 정본). 구버전 DROP 행은 이 커밋에 얹혀 상태와 함께 확정·롤백된다.
        await _retire_documents(session, tenant_id, old_ids)

        doc.status = 'ready'
        doc.is_active = True
        doc.char_count = sum(len(c.text) for c in chunks)
        # 이미지 비중이 큰 문서는 캡션·OCR 오류 가능성만 알린다 — 검색·인용은 정상이라 ready 그대로.
        if image_ratio >= IMAGE_WARN_RATIO:
            doc.status_reason = (
                f'이미지가 쪽 면적의 {image_ratio:.0f}%를 차지합니다. '
                f'그림은 AI가 설명을 만들고 표 이미지는 글자를 인식해 검색에 반영합니다. '
                f'생성된 설명이나 인식된 값이 부정확할 수 있어 원문 확인이 필요합니다.'
            )[:500]
        doc.indexed_at = datetime.now(timezone.utc)

        # ② 도중 토글·폴더 이동이 있었으면 ① 시점 값으로 넣은 메타가 낡는다(그때의 META 행은 이 문서를
        # 못 봤다) — 최신 PG 값과 다르면 META 행을 이 커밋에 얹는다.
        f_on = None
        if doc.folder_id is not None:
            f_on = (await session.execute(select(Folder.is_searchable).where(Folder.id == doc.folder_id))).scalar()
        now_searchable = os_index.effective_searchable(
            is_faq=False,
            doc_is_active=True,
            doc_status='ready',
            doc_is_searchable=doc.is_searchable,
            folder_is_searchable=f_on,
        )
        if now_searchable != indexed_searchable or doc.folder_id != folder_id:
            outbox.enqueue(session, tenant_id, outbox.META_DOCUMENTS, document_ids=[document_id])

        if outbox_row_id is not None:
            await outbox.mark_done(session, outbox_row_id)  # ready와 같은 커밋
        await session.commit()
    # total은 ③ 커밋까지 포함한다(사용자 체감 지연에는 cron 대기 최대 1분이 더 붙는다).
    INDEX_DURATION_SECONDS.labels(stage='total', ext=_ext).observe(time.monotonic() - _t_start)
    INDEX_TOTAL.labels(ext=_ext, result='ok').inc()  # 실패 쪽은 outbox.drain이 센다(재시도/확정 구분)


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
) -> tuple[Document, str | None]:
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
        page_count=None,
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
    await _retire_documents(session, tenant_id, others, status_reason='failed_superseded')
    await session.refresh(target)  # UPDATE로 바뀐 값을 ORM 객체에 반영(응답에 쓴다)
    outbox.enqueue(session, tenant_id, outbox.INDEX_DOCUMENT, document_id=target.id)
    return target, stale_blob


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
) -> tuple[Document, str | None]:
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
    return doc, None
