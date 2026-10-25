"""OpenSearch 색인 쓰기 — 청크 문서 조립·색인, 삭제, 메타 부분 갱신의 **정의점**.

부르는 곳은 `rag/outbox.py`의 drain과 복구 명령 `rag/os_reconcile.py`(삭제)다. 모든 연산은 멱등이고(결정적 `_id` upsert·없는 것 삭제·같은 값 덮어쓰기),
실패는 삼키지 않고 올려 호출부가 재시도·failed 확정을 하게 한다.
"""

import json
from collections import defaultdict

from sqlalchemy import select

from config import settings
from rag.embeddings import embed_texts
from rag.faq_indexing import build_faq_chunk_text
from rag.identifiers import chunk_identifiers, effective_headings
from rag.index_text import build_index_text
from rag.lexical import bigrams
from rag.models import Document, Faq, Folder
from rag.os_client import LEX_BIGRAM_FIELD, NORI_FIELD, client


def lex_text(text: str, filename: str | None, heading_path) -> str:
    """어휘 필드 입력 — 문서는 '파일명>헤딩' 프리픽스, FAQ(filename=None)는 원문 그대로. 임베딩 입력과 같은 조립이다."""
    if filename is None:
        return text
    return build_index_text(text, filename, list(heading_path or []))


def chunk_os_id(*, document_id: int | None = None, faq_id: int | None = None, chunk_index: int = 0) -> int:
    """(부모 id, chunk_index)로 계산한 결정적 chunk_id — `_id` upsert가 멱등이 된다.

    chunk_index는 20비트까지. FAQ는 음수 네임스페이스를 쓴다(캐시의 -faq_id 표기와 같은 관례).
    """
    if not 0 <= chunk_index < (1 << 20):
        # 넘으면 다른 문서의 청크와 id가 겹쳐 조용히 덮어쓴다.
        raise ValueError(f'chunk_index {chunk_index}가 20비트 상한(1,048,575)을 넘는다')
    if faq_id is not None:
        return -((faq_id << 20) | chunk_index)
    return (document_id << 20) | chunk_index


def effective_searchable(
    *,
    is_faq: bool,
    doc_is_active=None,
    doc_status=None,
    doc_is_searchable=None,
    folder_is_searchable=None,
    faq_is_active=None,
) -> bool:
    """검색 가능 판정의 **정의점** — 색인 시점에 계산해 `searchable` 플래그로 넣는다.

    판정 조건을 바꾸면 기존 청크 플래그가 낡으므로 전량 META 갱신(sync_meta_*)이 따라야 한다.
    """
    if is_faq:
        return bool(faq_is_active)
    return bool(
        doc_is_active
        and doc_status == 'ready'
        and doc_is_searchable
        and (folder_is_searchable is None or folder_is_searchable)
    )


def build_doc(
    chunk,
    filename: str | None,
    version: int | None,
    *,
    folder_id: int | None = None,
    folder_name: str | None = None,
    folder_description: str | None = None,
    searchable: bool = True,
) -> dict:
    """OpenSearch 문서 본문의 **정의점**. filename=None은 FAQ 청크다('FAQ'로 적는다)."""
    lex = lex_text(chunk.text, filename, chunk.heading_path)
    # 헤딩 줄 = 청킹이 기록한 묶음 헤딩 ∪ 본문의 헤딩 꼴 줄 — 식별자와 오리기 경계의 재료
    headings = effective_headings(list((chunk.meta or {}).get("headings") or []), chunk.text)
    vec = getattr(chunk, 'dense', None)
    if vec is None:
        # 벡터 없이 색인하면 kNN에서 안 잡히는 유령이 된다.
        raise ValueError(f'chunk {getattr(chunk, "id", "?")}: 벡터가 없다 — 색인 불가')
    return {
        "chunk_id": chunk.id,
        "tenant_id": chunk.tenant_id,
        "document_id": chunk.document_id,
        "faq_id": chunk.faq_id,
        "page": chunk.page,
        "version": version or 1,
        "is_table": bool((chunk.meta or {}).get("is_table")),
        "filename": filename or "FAQ",
        "heading_path": list(chunk.heading_path or []),
        "headings": headings,
        "identifiers": chunk_identifiers(list(chunk.heading_path or []), headings),
        "searchable": bool(searchable),
        "folder_id": folder_id,
        "folder_name": folder_name,
        "folder_description": folder_description,
        "text": chunk.text,
        "table_key": (chunk.meta or {}).get("table_key"),  # 표 묶음 청크만 — 그 밖은 None
        "table_text": (chunk.meta or {}).get("table_text"),
        NORI_FIELD: lex,
        LEX_BIGRAM_FIELD: " ".join(bigrams(lex)),
        "dense": [float(x) for x in vec],  # float32 → float: json 직렬화 때문
    }


async def bulk_index(docs: list[dict]) -> int:
    """_id=chunk_id upsert. _bulk은 건별로 성공/실패하므로 부분 실패를 삼키지 않고 첫 오류를 올린다."""
    if not docs:
        return 0
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": settings.opensearch_index, "_id": str(d["chunk_id"])}}))
        lines.append(json.dumps(d, ensure_ascii=False))
    resp = await client().bulk(body="\n".join(lines) + "\n", refresh="wait_for")
    if resp.get("errors"):
        first = next(i["index"] for i in resp["items"] if i["index"].get("error"))
        raise RuntimeError(f"bulk 색인 실패: {first['error']}")
    return len(docs)


async def _delete_by_terms(field: str, values: list) -> int:
    """문서·FAQ 단위 청크 제거 — chunk_id를 모르는 삭제 경로용."""
    if not values:
        return 0
    resp = await client().delete_by_query(
        index=settings.opensearch_index, refresh=True, body={"query": {"terms": {field: list(values)}}}
    )
    return int(resp.get("deleted", 0))


async def _update_meta(field: str, values: list, meta: dict) -> int:
    """메타 필드만 부분 갱신한다 — 재색인하지 않는다(벡터가 엔진에만 있어 재색인은 재임베딩이다).

    conflicts=proceed: 동시 재색인과의 버전 충돌은 넘기고 나머지 청크를 갱신한다 — 놓친 것은 outbox 재시도와
    `rag.os_reconcile --apply`가 잡는다. None 값은 null로 반영된다.
    """
    if not values:
        return 0
    script = "; ".join(f"ctx._source.{k} = params.{k}" for k in meta)
    resp = await client().update_by_query(
        index=settings.opensearch_index,
        refresh=True,
        conflicts='proceed',
        body={
            "query": {"terms": {field: list(values)}},
            "script": {"source": script, "params": meta, "lang": "painless"},
        },
    )
    return int(resp.get("updated", 0))


class _ParsedChunk:
    """build_doc에 넘기는 청크 형태 — 파싱 결과·FAQ 행의 어댑터. id는 chunk_os_id 값이다."""

    __slots__ = ('id', 'tenant_id', 'document_id', 'faq_id', 'text', 'heading_path', 'page', 'meta', 'dense')

    def __init__(self, *, cid, tenant_id, document_id, faq_id, text, heading_path, page, meta, dense):
        self.id, self.tenant_id = cid, tenant_id
        self.document_id, self.faq_id = document_id, faq_id
        self.text, self.heading_path, self.page, self.meta = text, heading_path, page, meta
        self.dense = dense


async def index_parsed_document(
    *,
    document_id: int,
    tenant_id: str,
    filename: str,
    version: int,
    folder_id,
    folder_name,
    folder_description,
    searchable: bool,
    chunks,
    embeddings,
) -> int:
    """파싱 결과를 색인한다. 옛 청크를 먼저 지운다 — 청크 수가 줄면 큰 chunk_index의 옛 청크가 꼬리로 남는다."""
    await _delete_by_terms('document_id', [document_id])
    docs = []
    for c, e in zip(chunks, embeddings):
        docs.append(
            build_doc(
                _ParsedChunk(
                    cid=chunk_os_id(document_id=document_id, chunk_index=c.chunk_index),
                    tenant_id=tenant_id,
                    document_id=document_id,
                    faq_id=None,
                    text=c.text,
                    heading_path=c.heading_path,
                    page=c.page,
                    meta=c.meta or {},
                    dense=list(e.dense),
                ),
                filename,
                version,
                folder_id=folder_id,
                folder_name=folder_name,
                folder_description=folder_description,
                searchable=searchable,
            )
        )
    return await bulk_index(docs)


async def index_parsed_faq(
    *, faq_id: int, tenant_id: str, text: str, question: str, searchable: bool, embedding
) -> int:
    """FAQ 청크 색인. 항목당 청크 1개라 chunk_index는 0 고정이다."""
    await _delete_by_terms('faq_id', [faq_id])
    doc = build_doc(
        _ParsedChunk(
            cid=chunk_os_id(faq_id=faq_id),
            tenant_id=tenant_id,
            document_id=None,
            faq_id=faq_id,
            text=text,
            heading_path=[question],
            page=None,
            meta={},
            dense=list(embedding.dense),
        ),
        None,
        None,
        searchable=searchable,
    )
    return await bulk_index([doc])


async def drop_documents_now(document_ids) -> int:
    return await _delete_by_terms('document_id', list(document_ids))


async def drop_faqs_now(faq_ids) -> int:
    return await _delete_by_terms('faq_id', list(faq_ids))


async def index_faq_chunks(session, faq_id: int) -> int:
    """INDEX_FAQ 핸들러 — FAQ 행에서 텍스트를 조립해 임베딩·색인한다. 행이 없으면(삭제됨) 0."""
    faq = (await session.execute(select(Faq).where(Faq.id == faq_id))).scalars().first()
    if faq is None:
        return 0
    text = build_faq_chunk_text(faq.question, faq.variants or [], faq.answer)
    embs = await embed_texts([text])  # FAQ는 프리픽스 없이 원문
    return await index_parsed_faq(
        faq_id=faq_id,
        tenant_id=faq.tenant_id,
        text=text,
        question=faq.question,
        searchable=effective_searchable(is_faq=True, faq_is_active=faq.is_active),
        embedding=embs[0],
    )


async def sync_meta_documents_now(session, document_ids) -> int:
    """문서 메타(검색가능·폴더) 부분 갱신. 폴더 변경이면 호출부가 그 폴더의 document_ids를 넘긴다.

    같은 메타 값을 갖는 문서끼리 묶어 갱신 호출 수를 줄인다.
    """
    rows = (
        await session.execute(
            select(
                Document.id,
                Document.is_active,
                Document.status,
                Document.is_searchable,
                Folder.id,
                Folder.name,
                Folder.description,
                Folder.is_searchable,
            )
            .outerjoin(Folder, Document.folder_id == Folder.id)
            .where(Document.id.in_(list(document_ids)))
        )
    ).all()

    groups = defaultdict(list)
    for did, d_active, d_status, d_searchable, f_id, f_name, f_desc, f_searchable in rows:
        key = (
            effective_searchable(
                is_faq=False,
                doc_is_active=d_active,
                doc_status=d_status,
                doc_is_searchable=d_searchable,
                folder_is_searchable=f_searchable,
            ),
            f_id,
            f_name,
            f_desc,
        )
        groups[key].append(did)

    n = 0
    for (searchable, f_id, f_name, f_desc), dids in groups.items():
        n += await _update_meta(
            'document_id',
            dids,
            {'searchable': searchable, 'folder_id': f_id, 'folder_name': f_name, 'folder_description': f_desc},
        )
    return n


async def sync_meta_faqs_now(session, faq_ids) -> int:
    """FAQ 활성 토글 부분 갱신."""
    rows = (await session.execute(select(Faq.id, Faq.is_active).where(Faq.id.in_(list(faq_ids))))).all()
    groups = defaultdict(list)
    for fid, active in rows:
        groups[bool(active)].append(fid)
    n = 0
    for active, fids in groups.items():
        n += await _update_meta('faq_id', fids, {'searchable': active})
    return n
