"""문서 업로드 서비스
업로드 파일을 dedupe/버전 정책(supersede)에 따라 처리한다.
"""
import asyncio
import mimetypes
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import ARRAY, Text, cast, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from database import AsyncSessionLocal
from rag import cache, lexical, opensearch, outbox
from rag.chunking import chunk_file
from rag.embeddings import embed_texts
from rag.index_text import build_index_text
from rag.models import Chunk, Document, Folder

_MIME_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _detect_mime(blob_path: Path) -> str:
    """확장자로 mime 추론 — 업로드 시점에 handle_upload가 부른다.

    `mimetypes`가 docx를 못 알아보는 환경이 있어 override를 먼저 본다.
    라우터의 SUPPORTED_SUFFIXES 5종이 전부 여기서 판별돼야 업로드가 500 없이 끝난다
    (tests/test_small_utils.py가 그 조건을 고정한다).
    """
    suffix = blob_path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(str(blob_path))
    if not mime:
        raise ValueError(f"mime 추론 실패: {blob_path}")
    return mime


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
        # 청킹은 동기 CPU 작업(pdfplumber·python-docx·openpyxl)이라 스레드로 보낸다.
        # 이벤트 루프에서 그대로 돌리면 PDF 하나에 100~200ms(실문서는 초 단위) 동안
        # 워커 전체가 멈춰, max_jobs=10이 청킹 구간에선 사실상 1이 된다.
        # to_thread(stdlib)를 쓴다 — 이 모듈은 워커와 라우터가 함께 import하므로
        # starlette(run_in_threadpool)를 들이면 도메인 계층이 웹 프레임워크에 묶인다.
        # 형식 분기는 chunk_file 안에 하나뿐이다 (#42 — 두 곳이던 게 xlsx 버그의 원인).
        chunks = await asyncio.to_thread(chunk_file, blob_path, description=description)
        # 빈 파일·텍스트레이어 없는 PDF 등 → 청크 0개면 ready 승격 대신 failed (C2 유령 ready 방지)
        if not chunks:
            raise ValueError('추출된 텍스트가 없습니다 (빈 파일이거나 파싱 결과가 비어 있음)')
        # 임베딩 입력에만 '파일명 > 헤딩' 컨텍스트를 얹는다 (저장되는 chunk.text는 원문 유지).
        # 리랭커도 rag/reranker.py에서 같은 조립을 쓴다 — 두 단계가 같은 형태를 보게.
        index_texts = [build_index_text(c.text, filename, c.heading_path) for c in chunks]
        embeddings = await embed_texts(index_texts)
        # 어휘 채널(#135)도 같은 조립을 토큰화한다 — 세 소비자(임베딩·리랭커·BM25)가 같은 형태.
        lex_tokens = [lexical.bigrams(t) for t in index_texts]

        # ── 3) 짧은 쓰기 — 청크 저장 + supersede + ready 승격 + 캐시 무효화 ──
        async with AsyncSessionLocal() as session:
            doc = await session.get(Document, document_id)
            if doc is None or doc.status != 'pending':   # 그새 상태 바뀌면(중복 실행 등) 스킵
                return

            # 청크 insert (xlsx 청크는 meta에 is_table·sheet — retriever '한 시트만' 필터용)
            #
            # **엔진이 서빙 정본이면 PG에 청크를 넣지 않는다** (#139). 서빙 경로가 한 번도
            # 읽지 않는 행을 쓰는 것이라 순수 낭비다 — 아래 flush_document_index가 파싱 결과와
            # 임베딩을 그대로 색인한다. 판단 근거·되돌리기는 opensearch.pg_stores_chunks 주석.
            # 청크 행을 두는 구성에서는 파생 컬럼(dense·lex_tsv·lex_len)을 다시 갈라
            # 판단한다(config의 pg_vector_columns).
            if opensearch.pg_stores_chunks():
                keep_pg_vectors = opensearch.pg_stores_vectors()
                for chunk, embedding, toks in zip(chunks, embeddings, lex_tokens):
                    derived = {
                        'dense': embedding.dense,
                        # 어휘 채널(#135) — 청크와 같은 트랜잭션이라 별도 정합 관리가 없다
                        'lex_tsv': func.array_to_tsvector(
                            cast(lexical.tsvector_lexemes(toks), ARRAY(Text))),
                        'lex_len': len(toks),
                    } if keep_pg_vectors else {}
                    session.add(Chunk(
                        document_id=doc.id,
                        tenant_id=doc.tenant_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        page=chunk.page,
                        heading_path=chunk.heading_path,
                        meta=chunk.meta or {},
                        **derived,
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
            if old_active_ids and opensearch.pg_stores_chunks():
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
            for old_id in old_active_ids:
                await cache.invalidate_source(session, doc.tenant_id, old_id)

            # 외부 색인 반영을 **같은 트랜잭션에** 적재한다 (#139 outbox). 커밋 후 직접
            # 호출하면 그 사이 프로세스가 죽었을 때 색인이 조용히 낡는다 — 커밋이 성공하면
            # 할 일이 반드시 남아 있게 만드는 것이 이 패턴의 요점이다. pg면 no-op.
            outbox.enqueue(session, doc.tenant_id, outbox.INDEX_DOCUMENT,
                           document_id=document_id)
            if old_active_ids:
                outbox.enqueue(session, doc.tenant_id, outbox.DROP_DOCUMENTS,
                               document_ids=old_active_ids)

            await session.commit()

        # 빠른 길 — 방금 계산한 임베딩으로 바로 색인하고 그 대기열 행을 지운다.
        # 느린 길(워커 cron)은 PG에서 벡터를 읽는데, PG가 벡터를 안 드는 구성에선 그게
        # 재임베딩이다. 실패해도 대기열 행이 남아 워커가 반영을 보장하므로 여기선 삼킨다
        # (try 안이라 예외가 새면 이미 커밋된 인제스션이 _mark_failed로 뒤집힌다).
        await outbox.flush_document_index(document_id, old_active_ids, embeddings,
                                         parsed_chunks=chunks)
    except Exception as e:
        await _mark_failed(document_id, str(e))


async def rederive_and_index(document_id: int) -> int:
    """원본 blob에서 다시 파싱·임베딩해 색인한다 — PG에 청크 행이 없을 때의 재시도 경로 (#139).

    엔진이 서빙 정본이고 PG가 청크를 안 드는 구성에서는 재색인의 원천이 **원본 파일**이다.
    outbox의 INDEX_DOCUMENT 재시도가 이 길로 온다(빠른 길이 실패했거나 프로세스가 죽은 경우).

    비싸다 — 파싱 + TEI 임베딩 전량이다. 그래서 정상 경로는 인제스션이 손에 든 결과를 바로
    색인하고(flush_document_index) 이 함수는 보험으로만 쓴다.

    문서가 없거나 blob을 못 읽으면 예외를 올린다 — outbox가 백오프로 재시도하고, 계속 실패하면
    attempts·last_error에 남아 사람이 본다.
    """
    from rag import opensearch
    async with AsyncSessionLocal() as session:
        row = (await session.execute(
            select(Document.tenant_id, Document.filename, Document.version,
                   Document.blob_path, Document.description,
                   Document.is_active, Document.status, Document.is_searchable,
                   Folder.id, Folder.name, Folder.description, Folder.is_searchable)
            .outerjoin(Folder, Document.folder_id == Folder.id)
            .where(Document.id == document_id))).first()
    if row is None:
        return 0
    (tenant_id, filename, version, blob_path, desc, d_active, d_status, d_searchable,
     f_id, f_name, f_desc, f_searchable) = row

    chunks = await asyncio.to_thread(chunk_file, blob_path, description=desc or '')
    if not chunks:
        raise ValueError(f'문서 {document_id}: 재파싱 결과가 비었다')
    index_texts = [build_index_text(c.text, filename, c.heading_path) for c in chunks]
    embeddings = await embed_texts(index_texts)
    return await opensearch.index_parsed_document(
        document_id=document_id, tenant_id=tenant_id, filename=filename,
        version=version or 1, folder_id=f_id, folder_name=f_name, folder_description=f_desc,
        searchable=opensearch.effective_searchable(
            is_faq=False, doc_is_active=d_active, doc_status=d_status,
            doc_is_searchable=d_searchable, folder_is_searchable=f_searchable),
        chunks=chunks, embeddings=embeddings)


async def handle_upload(
        session: AsyncSession,
        tenant_id: str,
        filename: str,
        blob_path: Path,
        description: str | None = None,
) -> Document:
    """업로드 시점 처리: pending row 등록까지만.
    실제 청킹/임베딩/supersede는 워커(index_document)가 수행한다.
    description은 표 설명(xlsx 검색 보강) — 워커가 청킹 시 병합한다.
    mime은 blob_path에서 직접 구한다 — 호출부가 계산해 넘길 이유가 없다.

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
        mime=_detect_mime(blob_path),
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


