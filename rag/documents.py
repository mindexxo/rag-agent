"""문서 업로드 서비스
업로드 파일을 dedupe/버전 정책(supersede)에 따라 처리한다.
"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from rag.cache import AnswerCache
from rag.chunking import chunk_file, chunk_txt
from rag.xlsx_chunking import chunk_xlsx
from rag.embeddings import embed_texts
from rag.index_text import build_index_text
from rag.models import Chunk, Document

async def _mark_failed(document_id: int, reason: str) -> None:
    """인덱싱 실패 시 문서를 failed로 기록 (짧은 세션). status=='pending'일 때만."""
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is not None and doc.status == 'pending':
            doc.status = "failed"
            doc.status_reason = reason[:500]
            await session.commit()


async def index_pending_document(document_id: int) -> None:
    """pending 문서를 청킹/임베딩해 ready로 만든다. 워커가 호출.

    트랜잭션 경계 3분할 — 무거운 청킹·임베딩은 트랜잭션 밖에서 수행해
    DB 커넥션을 오래 물지 않는다 (대형 문서·크롤링 대비). 세션은 함수가 관리.
      1) 짧은 읽기: 처리 대상 확인 + 파싱에 필요한 정보만
      2) 트랜잭션 밖: 청킹 + 임베딩 (무거움)
      3) 짧은 쓰기: 청크 저장 + supersede + ready 승격 + 캐시 무효화
    stage 2·3 어느 단계 예외든 failed로 기록(P1-5a). 타임아웃(CancelledError)·워커 크래시는
    여기서 못 잡으므로 GET /documents의 lazy 스윕이 백스톱. 매 단계 status=='pending' 재확인.
    """
    # ── 1) 짧은 읽기 — 커넥션 즉시 반납 ──
    async with AsyncSessionLocal() as session:
        doc = await session.get(Document, document_id)
        if doc is None or doc.status != 'pending':
            return
        blob_path = doc.blob_path
        description = doc.description or ''
        filename = doc.filename          # 임베딩 입력 앞에 붙일 문서 컨텍스트 (index_text)

    try:
        # ── 2) 무거운 계산 — 트랜잭션 밖 (DB 커넥션 안 물고 청킹·임베딩) ──
        #    형식별 분기: xlsx는 openpyxl(헤더 확실히), txt는 평문, 그 외 pdfplumber/python-docx
        blob_lower = blob_path.lower()
        if blob_lower.endswith('.xlsx'):
            chunks = chunk_xlsx(blob_path, description=description)
        elif blob_lower.endswith('.txt'):
            chunks = chunk_txt(blob_path)
        else:
            chunks = chunk_file(blob_path)   # pdf/docx/md
        # 빈 파일·텍스트레이어 없는 PDF 등 → 청크 0개면 ready 승격 대신 failed (C2 유령 ready 방지)
        if not chunks:
            raise ValueError('추출된 텍스트가 없습니다 (빈 파일이거나 파싱 결과가 비어 있음)')
        # 임베딩 입력에만 '파일명 > 헤딩' 컨텍스트를 얹는다 (저장되는 chunk.text는 원문 유지).
        # 리랭커도 rag/reranker.py에서 같은 조립을 쓴다 — 두 단계가 같은 형태를 보게.
        embeddings = await embed_texts([
            build_index_text(c.text, filename, c.heading_path) for c in chunks
        ])

        # ── 3) 짧은 쓰기 — 청크 저장 + supersede + ready 승격 + 캐시 무효화 ──
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, document_id)
            if doc is None or doc.status != 'pending':   # 그새 상태 바뀌면(중복 실행 등) 스킵
                return

            # 청크 insert (xlsx 청크는 meta에 is_table·sheet — retriever '한 시트만' 필터용)
            for chunk, embedding in zip(chunks, embeddings):
                session.add(Chunk(
                    document_id=doc.id,
                    tenant_id=doc.tenant_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page=chunk.page,
                    heading_path=chunk.heading_path,
                    meta=chunk.meta or {},
                    dense=embedding.dense,
                    # sparse=SparseVector(embedding.sparse, 250002),   # [dense-only, F99]
                  ))

            # supersede: 같은 filename의 다른 active 버전 내리기 + 청크 삭제
            others = (await session.execute(
                select(Document)
                .where(Document.tenant_id == doc.tenant_id)
                .where(Document.filename == doc.filename)
                .where(Document.is_active.is_(True))
                .where(Document.id != doc.id)
            )).scalars().all()

            old_active_ids = [o.id for o in others]
            for o in others:
                o.is_active = False
                o.status = "deleted"
            if old_active_ids:
                await session.execute(
                    delete(Chunk).where(Chunk.document_id.in_(old_active_ids))
                )

            # 옛 active off를 먼저 반영 -> 유니크 위반 방지
            await session.flush()

            # 이 문서를 ready + active로 승격
            doc.status = "ready"
            doc.is_active = True
            doc.char_count = sum(len(c.text) for c in chunks)
            doc.indexed_at = datetime.now(timezone.utc).replace(tzinfo=None)   # naive 컬럼 — UTC 유지

            # 옛 문서 근거 캐시 무효화
            cache = AnswerCache()
            for old_id in old_active_ids:
                await cache.invalidate_document(session, doc.tenant_id, old_id)

            await session.commit()
    except Exception as e:
        await _mark_failed(document_id, str(e))


async def handle_upload(
        session: AsyncSession,
        tenant_id: str,
        filename: str,
        mime: str,
        blob_path: Path,
        description: str | None = None,
) -> Document:
    """업로드 시점 처리: pending row 등록까지만.
    실제 청킹/임베딩/supersede는 워커(index_document)가 수행한다.
    description은 표 설명(xlsx 검색 보강) — 워커가 청킹 시 병합한다.

    문서 식별은 **filename 완전 일치** 하나뿐 (2026-08-05 정책 확정).
    내용 해시(sha) dedupe는 제거 — 같은 이름이면 내용이 같아도 새 version이 된다.
    "같은 이름이면 물어보고, 확인하면 대체"라는 단일 규칙을 유지하기 위함
    (내용 동일 여부로 확인 창을 띄울지 말지 분기하면 규칙이 둘이 된다).
    """
    # 같은 filename의 모든 버전 조회 (다음 version 계산 + 설정 계승용)
    docs = (await session.execute(
        select(Document)
        .where(Document.tenant_id == tenant_id)
        .where(Document.filename == filename)
    )).scalars().all()

    # pending 버전 row만 insert. is_active=False(ready 전엔 검색 제외).
    # 폴더 소속·참조 on/off(F2)는 직전 버전에서 계승 — 개정판 업로드로 설정이 풀리지 않게.
    next_version = max((d.version for d in docs), default=0) + 1
    prev = max(docs, key=lambda d: d.version) if docs else None
    doc = Document(
        tenant_id=tenant_id,
        filename=filename,
        mime=mime,
        blob_path=str(blob_path),
        version=next_version,
        is_active=False,
        status="pending",
        folder_id=prev.folder_id if prev else None,
        is_searchable=prev.is_searchable if prev else True,
        description=description if description is not None else (prev.description if prev else None),
    )
    session.add(doc)
    await session.flush()  # doc.id 확보 (enqueue에 필요)
    return doc


