"""인제스션 — pending 문서를 파싱·임베딩·색인해 ready로 올리는 outbox INDEX_DOCUMENT 핸들러.

세션 셋(짧은 읽기 → DB 없이 무거운 계산 → 유일한 커밋)으로 나뉘며, `ready ≡ 색인됨`이 커밋 단위로 성립한다.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import NamedTuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from rag import outbox
from rag.chunking import abort_conversion, chunk_file, count_picture_placeholders, pdf_image_area_ratio
from rag.documents import retire_documents
from rag.metrics import INDEX_DURATION_SECONDS, INDEX_PICTURE_PLACEHOLDER_TOTAL, INDEX_TOTAL, ext_label
from rag.models import ALIVE_DOCUMENT_STATUSES, Document, Folder
from rag.search import os_index
from rag.search.embeddings import embed_texts  # 이름으로 들일 것 — tests/conftest.py가 rag.ingestion.embed_texts를 패치
from rag.search.index_text import build_index_text

# 이미지 면적(%)이 이 값 이상이면 문서에 경고를 남긴다 — 작은 도표 하나에도 핵심 내용이 들 수 있어 낮게 잡는다.
IMAGE_WARN_PERCENT = 0.5


class _IndexTarget(NamedTuple):
    """① 짧은 읽기가 ②·③에 넘기는 값 — 세션을 닫고 들고 가므로 ORM 객체가 아니라 값이다."""

    tenant_id: str
    filename: str
    version: int
    blob_path: str
    description: str
    folder_id: int | None
    folder_name: str | None
    folder_description: str | None
    folder_searchable: bool | None
    doc_searchable: bool
    old_ids: list[int]  # supersede 대상 — 같은 filename의 낮은 version 중 살아 있는 것


class _Parsed(NamedTuple):
    chunks: list
    image_ratio: float
    embeddings: list


class _Indexed(NamedTuple):
    """②가 엔진에 넣은 결과 중 ③이 쓰는 것."""

    chunks: list
    image_ratio: float
    searchable: bool  # 엔진에 넣은 searchable — ③에서 최신 PG 값과 비교해 META 보정


async def index_pending_document(document_id: int, *, outbox_row_id: int | None = None) -> None:
    """INDEX_DOCUMENT 핸들러 — 인제스션 전체. 실패는 예외로 올린다(재시도·failed 판정은 outbox.drain).

    커밋은 ③의 한 번뿐이고 ready·supersede·캐시 무효화·outbox done이 함께 들어가 `ready ≡ 색인됨`이
    커밋 단위로 성립한다. 어디서 죽어도 다음 회차가 처음부터 다시 한다(엔진 색인은 같은 _id upsert라 멱등).
    **파괴적 엔진 변경은 커밋 뒤 outbox로만** — 구버전 청크 삭제는 ③에 DROP 행으로 얹어, ③이 실패하면
    구버전이 PG·엔진 모두 그대로 남게 한다.

    세션을 셋으로 나눠 무거운 파싱·임베딩 동안 DB 커넥션을 물지 않는다.
      ① 짧은 읽기   처리 대상 확인 + 파싱·색인에 필요한 값만        (_load_index_target)
      ② DB 없이     파싱·청킹·임베딩 → 엔진 색인(삭제 없음)          (_parse_and_embed + index_parsed_document)
      ③ 짧은 쓰기   유일한 커밋                                       (_commit_ready)
    """
    async with AsyncSessionLocal() as session:
        target = await _load_index_target(session, document_id)
        if target is None:  # 이미 처리됨·삭제됨 — 할 일 없음
            await _close_row(session, outbox_row_id)
            return

    # 단계별 소요 지표는 성공한 건만 기록한다(실패는 outbox.drain이 센다).
    ext = ext_label(target.filename)
    t_start = time.monotonic()
    parsed = await _parse_and_embed(target, ext)
    t_embed_done = time.monotonic()

    # searchable은 PG의 현재 상태(pending·inactive)가 아니라 ③이 만들 상태(ready·active)로 계산한다.
    searchable = os_index.effective_searchable(
        is_faq=False,
        doc_is_active=True,
        doc_status='ready',
        doc_is_searchable=target.doc_searchable,
        folder_is_searchable=target.folder_searchable,
    )
    await os_index.index_parsed_document(
        document_id=document_id,
        tenant_id=target.tenant_id,
        filename=target.filename,
        version=target.version,
        folder_id=target.folder_id,
        folder_name=target.folder_name,
        folder_description=target.folder_description,
        searchable=searchable,
        chunks=parsed.chunks,
        embeddings=parsed.embeddings,
    )
    # 구버전 청크는 여기서 지우지 않는다 — ③의 DROP 행이 지운다.
    INDEX_DURATION_SECONDS.labels(stage='index', ext=ext).observe(time.monotonic() - t_embed_done)

    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None or doc.status != 'pending':
            # ②가 도는 사이 삭제됐다. 엔진에 넣은 청크는 그 삭제가 만든 DROP 행이 다음 회차에 지운다.
            await _close_row(session, outbox_row_id)
            return
        await _commit_ready(
            session, doc, target, _Indexed(parsed.chunks, parsed.image_ratio, searchable), outbox_row_id
        )
    # total은 ③ 커밋까지 포함한다(사용자 체감 지연에는 cron 대기 최대 1분이 더 붙는다).
    INDEX_DURATION_SECONDS.labels(stage='total', ext=ext).observe(time.monotonic() - t_start)
    INDEX_TOTAL.labels(ext=ext, result='ok').inc()  # 실패 쪽은 outbox.drain이 센다(재시도/확정 구분)


async def _load_index_target(session: AsyncSession, document_id: int) -> _IndexTarget | None:
    """① — pending 문서의 색인 재료. pending이 아니면(이미 처리·삭제) None."""
    row = (
        await session.execute(
            select(Document, Folder.name, Folder.description, Folder.is_searchable)
            .outerjoin(Folder, Document.folder_id == Folder.id)
            .where(Document.id == document_id)
        )
    ).first()
    if row is None or row[0].status != 'pending':
        return None
    doc, folder_name, folder_desc, folder_searchable = row
    # outbox 처리 순서는 보장되지 않으므로 늦게 도는 구버전이 신버전을 내리지 못하게 version 조건을 건다.
    old_ids = list(
        (
            await session.execute(
                select(Document.id)
                .where(Document.tenant_id == doc.tenant_id)
                .where(Document.filename == doc.filename)
                .where(Document.version < doc.version)
                .where(Document.status.in_(ALIVE_DOCUMENT_STATUSES))
            )
        )
        .scalars()
        .all()
    )
    return _IndexTarget(
        tenant_id=doc.tenant_id,
        filename=doc.filename,
        version=doc.version,
        blob_path=doc.blob_path,
        description=doc.description or '',
        folder_id=doc.folder_id,
        folder_name=folder_name,
        folder_description=folder_desc,
        folder_searchable=folder_searchable,
        doc_searchable=doc.is_searchable,
        old_ids=old_ids,
    )


async def _close_row(session: AsyncSession, outbox_row_id: int | None) -> None:
    """처리할 문서가 없을 때 outbox 행만 닫는다."""
    if outbox_row_id is not None:
        await outbox.mark_done(session, outbox_row_id)
        await session.commit()


async def _parse_and_embed(target: _IndexTarget, ext: str) -> _Parsed:
    """② — DB 세션 없이 청킹·임베딩. 빈 결과는 예외(유령 ready 방지), parse·embed 지표는 성공 분기에서만."""
    t_start = time.monotonic()
    # 청킹은 동기 CPU 작업이라 스레드로 보낸다. 취소되면 변환 프로세스를 즉시 죽인다 — to_thread는 취소돼도
    # 스레드가 계속 돌며 변환 잠금을 쥐어, 다음 회차까지 잠금에 막혀 연쇄로 취소된다.
    try:
        chunks = await asyncio.to_thread(chunk_file, target.blob_path, description=target.description)
    except asyncio.CancelledError:
        abort_conversion()
        raise
    image_ratio = await asyncio.to_thread(pdf_image_area_ratio, target.blob_path)
    if not chunks:  # 빈 파일·텍스트레이어 없는 PDF → ready 승격 대신 실패
        raise ValueError('추출된 텍스트가 없습니다 (빈 파일이거나 파싱 결과가 비어 있음)')
    INDEX_DURATION_SECONDS.labels(stage='parse', ext=ext).observe(time.monotonic() - t_start)
    # 캡션이 안 붙은 그림 수(면적 하한 미만 등) — 캡션 실패는 변환에서 예외로 끝나 여기까지 오지 않는다.
    if placeholders := count_picture_placeholders(chunks):
        INDEX_PICTURE_PLACEHOLDER_TOTAL.labels(ext=ext).inc(placeholders)
    t_parse_done = time.monotonic()
    # 임베딩 입력에만 '파일명 > 헤딩' 컨텍스트를 얹는다(저장되는 chunk.text는 원문 유지).
    index_texts = [build_index_text(c.text, target.filename, c.heading_path) for c in chunks]
    embeddings = await embed_texts(index_texts)
    INDEX_DURATION_SECONDS.labels(stage='embed', ext=ext).observe(time.monotonic() - t_parse_done)
    return _Parsed(chunks, image_ratio, embeddings)


async def _commit_ready(
    session: AsyncSession, doc: Document, target: _IndexTarget, indexed: _Indexed, outbox_row_id: int | None
) -> None:
    """③ — 유일한 커밋. 구버전 retire(DROP 행 포함)·ready 승격·메타 보정·outbox done이 함께 확정·롤백된다."""
    await retire_documents(session, target.tenant_id, target.old_ids)

    doc.status = 'ready'
    doc.is_active = True
    doc.char_count = sum(len(c.text) for c in indexed.chunks)
    # 이미지 비중이 큰 문서는 캡션·OCR 오류 가능성만 알린다 — 검색·인용은 정상이라 ready 그대로.
    if indexed.image_ratio >= IMAGE_WARN_PERCENT:
        doc.status_reason = (
            f'이미지가 쪽 면적의 {indexed.image_ratio:.0f}%를 차지합니다. '
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
    if now_searchable != indexed.searchable or doc.folder_id != target.folder_id:
        outbox.enqueue(session, target.tenant_id, outbox.META_DOCUMENTS, document_ids=[doc.id])

    if outbox_row_id is not None:
        await outbox.mark_done(session, outbox_row_id)  # ready와 같은 커밋
    await session.commit()
