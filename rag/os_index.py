"""OpenSearch 쓰기 경로 — 색인·삭제·메타 부분 갱신, 그리고 정합 관리. #146에서 분리.

청크가 PG 트랜잭션 밖(엔진)에 있으므로 "PG 변경 ↔ 색인 반영"의 정합은 따로 관리해야 한다.
그 방법이 이 절이고, 정의점은 rag/outbox.py다. 이 모듈의 함수들은 outbox 처리기(_apply)와 인제스션 핸들러가 부른다.

## 전제 — 색인이 서빙의 정본이다

실무 표준 구성: 검색 결과의 텍스트·메타를 엔진이 돌려준다(`fetch_chunk_map`). PG로 되묻어
검증하지 않는다. 그래서 "색인이 낡으면 낡은 답이 나간다"가 구조적으로 가능하고, 그 자리를
아래가 메운다.

## 1) 필터를 엔진에 비정규화

`searchable`(effective_searchable)·`tenant_id`를 청크 문서마다 넣고 엔진에서 필터한다.
검색 후 PG로 걸러내면 상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다.
비검색 청크도 색인한다 — 토글 on/off가 문서 추가·삭제가 아니라 플래그 부분 갱신
(`_update_meta`)이 되고 재임베딩이 없다.

## 2) 모든 변경은 트랜잭셔널 outbox로 — 처리는 단일 워커 cron 1분

업로드·삭제·토글·FAQ 변경은 라우터가 PG 변경과 **같은 트랜잭션**에 대기열 행을 남긴다
(rag/outbox.enqueue). 워커 cron이 1분마다 id 순으로 처리한다. 커밋 후 직접 호출·인라인
처리는 **없다** — 아래 원시 연산들은 drain만 부른다.

인제스션(INDEX_DOCUMENT)은 rag/documents.index_pending_document가 파싱·임베딩·색인을 한
뒤 **마지막 커밋 하나**에 ready·supersede·done을 넣는다 — `ready ≡ 색인됨`이 커밋 단위로
성립한다. 색인 시 `searchable`은 그 커밋이 만들 상태(ready·active)로 미리 계산해 켜진 채로
넣고, 구버전은 그 뒤에 지운다 — 빈 창이 없다.

**제품 결정 — 반영 지연을 받아들인다.** 삭제·비검색·FAQ 수정도 cron까지 최대 1분(재시도
포함 1~5분) 검색에 반영되지 않는다: "문서 변경은 검색에 최대 1~5분 뒤 반영될 수 있다"가
가이드다. 답변 캐시는 라우터가 즉시 무효화하므로 창은 새 검색에만 열린다. 좁히려면 워커
폴링 루프(10초)나 "지금 반영" 수동 트리거를 얹으면 되고 둘 다 이 구조 위에 그대로 붙는다.

## 3) 재동기화 — 안전판

`python -m eval.os_reconcile`이 PG↔OS를 **문서 단위**로 대조해 복구한다(`reconcile`) — PG에
청크가 없으므로 청크 일부 누락은 못 잡고 outbox의 원자성(문서 단위 색인 후 한 커밋)에 의존한다.

아래 원시 연산들은 **실패를 삼키지 않는다** — 예외를 올려 outbox가 횟수를 세고 MAX_ATTEMPTS에
failed로 확정하게 한다. 전부 멱등이다: 색인은 _id=chunk_id upsert(결정적 id), 삭제는 없는 것을
지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이라 재시도가 겹쳐도 결과가 같다.
"""
import json
from collections import defaultdict

from sqlalchemy import select

from config import settings
from rag.embeddings import embed_texts
from rag.faq_indexing import build_faq_chunk_text
from rag.index_text import build_index_text
from rag.lexical import bigrams
from rag.models import Document, Faq, Folder
from rag.os_client import LEX_BIGRAM_FIELD, NORI_FIELD, client

def lex_text(text: str, filename: str | None, heading_path) -> str:
    """어휘·Nori 필드의 입력 텍스트 — 인제스션·운영 어휘 채널과 **동일 조립**.

    문서는 '파일명>헤딩' 프리픽스(build_index_text), FAQ는 원문 그대로다. 이 비대칭은
    임베딩 입력과 같다(rag/index_text.py docstring) — 임베딩·Nori·bigram 세 필드가 같은
    텍스트를 본다. 폴더 설명은 넣지 않는다 — 리랭커 전용이다(임베딩·어휘엔 미포함).
    """
    if filename is None:
        return text
    return build_index_text(text, filename, list(heading_path or []))


def chunk_os_id(*, document_id: int | None = None, faq_id: int | None = None,
                chunk_index: int = 0) -> int:
    """**결정적** chunk_id — PG 시퀀스가 없으므로 (부모 id, chunk_index)로 계산한다.

    같은 입력이면 같은 id다 —
    같은 문서를 다시 색인해도 같은 id가 나오므로 `_id` upsert가 그대로 멱등이고, 재색인이
    유령을 남기지 않는다(옛 청크가 더 적어졌을 때 남는 꼬리는 재색인 전 drop이 치운다).

    chunk_index는 20비트(약 100만)까지 — 한 문서의 청크 수 상한으로 충분하다. FAQ는 음수
    네임스페이스를 쓴다(캐시가 FAQ 출처를 -faq_id로 표기하는 기존 관례와 같은 방식).
    """
    if not 0 <= chunk_index < (1 << 20):
        # 20비트를 넘으면 상위 비트(부모 id 몫)를 침범해 **다른 문서의 청크와 같은 id**가 된다 —
        # 조용히 덮어쓰는 것보다 크게 실패한다.
        raise ValueError(f'chunk_index {chunk_index}가 20비트 상한(1,048,575)을 넘는다')
    if faq_id is not None:
        return -((faq_id << 20) | chunk_index)
    return (document_id << 20) | chunk_index


def effective_searchable(*, is_faq: bool, doc_is_active=None, doc_status=None,
                         doc_is_searchable=None, folder_is_searchable=None,
                         faq_is_active=None) -> bool:
    """검색 가능 판정의 **정의점** — 색인 시점에 계산해 `searchable` 플래그로 넣는다.

    문서: 활성 + ready + 검색가능 + (폴더 없음 또는 폴더 검색가능) / FAQ: 활성.
    조건이 바뀌면 플래그가 낡은 청크가 생기므로 전량 META 갱신(sync_meta_*)이 따라야 한다.
    """
    if is_faq:
        return bool(faq_is_active)
    return bool(doc_is_active and doc_status == 'ready' and doc_is_searchable
                and (folder_is_searchable is None or folder_is_searchable))


def build_doc(chunk, filename: str | None, version: int | None, *,
              folder_id: int | None = None, folder_name: str | None = None,
              folder_description: str | None = None, searchable: bool = True) -> dict:
    """OpenSearch 문서 본문 — **정의점**. 인제스션(index_parsed_document·index_parsed_faq)이 쓴다.

    chunk는 `_ParsedChunk`(id·tenant_id·document_id·faq_id·text·heading_path·page·meta·dense).
    filename=None은 FAQ 청크다 — 'FAQ'로 적는다(fetch_chunk_map과 동일 규약).
    벡터는 chunk.dense에 실려 와야 한다 — 인제스션이 방금 계산한 임베딩이다.
    """
    lex = lex_text(chunk.text, filename, chunk.heading_path)
    vec = getattr(chunk, 'dense', None)
    if vec is None:
        # 조용히 벡터 없는 문서를 색인하면 kNN에서 안 잡히는 유령이 된다 — 크게 실패한다.
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
        "searchable": bool(searchable),
        "folder_id": folder_id,
        "folder_name": folder_name,
        "folder_description": folder_description,
        "text": chunk.text,
        NORI_FIELD: lex,
        LEX_BIGRAM_FIELD: " ".join(bigrams(lex)),
        "dense": [float(x) for x in vec],      # float32 → float: json 직렬화 때문
    }


async def bulk_index(docs: list[dict]) -> int:
    """_bulk 색인. _id=chunk_id라 재실행이 중복을 만들지 않고 upsert가 된다.

    **_bulk은 원자적이지 않다** — 건별로 성공/실패한다. 부분 실패를 삼키면 색인이 조용히
    비므로 첫 오류를 예외로 올린다(호출부가 잡아 지표로 기록한다).
    """
    if not docs:
        return 0
    lines = []
    for d in docs:
        lines.append(json.dumps({"index": {"_index": settings.opensearch_index,
                                           "_id": str(d["chunk_id"])}}))
        lines.append(json.dumps(d, ensure_ascii=False))
    resp = await client().bulk(body="\n".join(lines) + "\n", refresh="wait_for")
    if resp.get("errors"):
        first = next(i["index"] for i in resp["items"] if i["index"].get("error"))
        raise RuntimeError(f"bulk 색인 실패: {first['error']}")
    return len(docs)


async def _delete_by_terms(field: str, values: list) -> int:
    """_delete_by_query — 문서·FAQ 단위 청크 제거. chunk_id를 모르는 삭제 경로용."""
    if not values:
        return 0
    resp = await client().delete_by_query(
        index=settings.opensearch_index, refresh=True,
        body={"query": {"terms": {field: list(values)}}})
    return int(resp.get("deleted", 0))


# 아래 연산들은 **실패를 삼키지 않는다** — 예외를 올려 outbox가 재시도·백오프를 걸게 한다
# (rag/outbox.py). 삼킴과 지표는 그쪽 한 곳에 모여 있다. 전부 멱등이다: 색인은
# _id=chunk_id upsert, 삭제는 없는 것을 지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이라
# 재시도가 겹쳐 두 번 처리해도 결과가 같다(단일 워커 cron만 부른다 — 인라인 없음).


async def _update_meta(field: str, values: list, meta: dict) -> int:
    """_update_by_query로 메타 필드만 갱신한다 — **재색인하지 않는다.**

    재색인하면 재임베딩이 따라온다(벡터는 엔진에만 있다). 메타 하나 바꾸는데 GPU를 태우는 건
    뒤바뀐 설계다. 그래서 부분 갱신이 이 경로의 유일한 선택이다.

    conflicts=proceed: 같은 문서를 동시에 재색인 중이면 버전 충돌이 날 수 있다. 그때 전체를
    실패시키기보다 넘긴다 — 여기서 멈추면 나머지 청크가 옛 메타로 남는 쪽이 더 나쁘다.
    놓친 것은 outbox 재시도와 `eval.os_reconcile --apply`가 잡는다.

    params가 None인 필드(미분류 이동의 folder_id 등)는 painless `ctx._source.x = params.x`로
    **null이 반영되고** `exists` 필터에서도 없는 것으로 본다 — 2.18.0에서 실측(2026-09-11).
    """
    if not values:
        return 0
    script = "; ".join(f"ctx._source.{k} = params.{k}" for k in meta)
    resp = await client().update_by_query(
        index=settings.opensearch_index, refresh=True, conflicts='proceed',
        body={"query": {"terms": {field: list(values)}},
              "script": {"source": script, "params": meta, "lang": "painless"}})
    return int(resp.get("updated", 0))


class _ParsedChunk:
    """build_doc에 넘기는 청크 형태 — 파싱 결과(chunk_file 산출물)·FAQ 행의 어댑터.

    id는 chunk_os_id로 계산한 결정적 값이다.
    """
    __slots__ = ('id', 'tenant_id', 'document_id', 'faq_id',
                 'text', 'heading_path', 'page', 'meta', 'dense')

    def __init__(self, *, cid, tenant_id, document_id, faq_id, text,
                 heading_path, page, meta, dense):
        self.id, self.tenant_id = cid, tenant_id
        self.document_id, self.faq_id = document_id, faq_id
        self.text, self.heading_path, self.page, self.meta = text, heading_path, page, meta
        self.dense = dense


async def index_parsed_document(*, document_id: int, tenant_id: str, filename: str,
                                version: int, folder_id, folder_name, folder_description,
                                searchable: bool, chunks, embeddings) -> int:
    """파싱 결과를 곧장 색인한다 (#139) — 인제스션이 손에 든 청크·임베딩 그대로.

    같은 문서를 다시 색인할 때 **옛 청크를 먼저 지운다** — 청크 수가 줄면 chunk_index가 큰
    옛 문서가 꼬리로 남기 때문이다(id가 결정적이라 겹치는 것들은 upsert로 덮인다).
    """
    await _delete_by_terms('document_id', [document_id])
    docs = []
    for c, e in zip(chunks, embeddings):
        docs.append(build_doc(
            _ParsedChunk(cid=chunk_os_id(document_id=document_id, chunk_index=c.chunk_index),
                         tenant_id=tenant_id, document_id=document_id, faq_id=None,
                         text=c.text, heading_path=c.heading_path, page=c.page,
                         meta=c.meta or {}, dense=list(e.dense)),
            filename, version,
            folder_id=folder_id, folder_name=folder_name,
            folder_description=folder_description, searchable=searchable))
    return await bulk_index(docs)


async def index_parsed_faq(*, faq_id: int, tenant_id: str, text: str, question: str,
                           searchable: bool, embedding) -> int:
    """FAQ 청크 색인. 항목당 청크 1개라 chunk_index는 0 고정이다."""
    await _delete_by_terms('faq_id', [faq_id])
    doc = build_doc(
        _ParsedChunk(cid=chunk_os_id(faq_id=faq_id), tenant_id=tenant_id,
                     document_id=None, faq_id=faq_id, text=text,
                     heading_path=[question], page=None, meta={},
                     dense=list(embedding.dense)),
        None, None, searchable=searchable)
    return await bulk_index([doc])


async def drop_documents_now(document_ids) -> int:
    return await _delete_by_terms('document_id', list(document_ids))


async def drop_faqs_now(faq_ids) -> int:
    return await _delete_by_terms('faq_id', list(faq_ids))


async def index_faq_chunks(session, faq_id: int) -> int:
    """FAQ 청크 (재)색인 — INDEX_FAQ 핸들러. FAQ 행(질문·유사질문·답변)에서 텍스트를 조립해
    임베딩하고 색인한다. FAQ는 파싱이 없어 원천이 그 행 자체다. 행이 없으면(삭제됨) 0.
    """
    faq = (await session.execute(select(Faq).where(Faq.id == faq_id))).scalars().first()
    if faq is None:
        return 0
    text = build_faq_chunk_text(faq.question, faq.variants or [], faq.answer)
    embs = await embed_texts([text])          # FAQ는 프리픽스 없이 원문 (임베딩 입력 규약)
    return await index_parsed_faq(
        faq_id=faq_id, tenant_id=faq.tenant_id, text=text, question=faq.question,
        searchable=effective_searchable(is_faq=True, faq_is_active=faq.is_active),
        embedding=embs[0])


async def sync_meta_documents_now(session, document_ids) -> int:
    """문서 메타(검색가능·폴더) 부분 갱신. 문서 토글·폴더 이동, 그리고 폴더 자체의 변경
    (이름·설명·검색토글 → 호출부가 그 폴더의 document_ids를 넘긴다)에서 쓴다.

    같은 메타 값을 갖는 문서끼리 묶어 호출 수를 줄인다 — 폴더 변경이면 대개 1~2번이다.
    """
    rows = (await session.execute(
        select(Document.id, Document.is_active, Document.status, Document.is_searchable,
               Folder.id, Folder.name, Folder.description, Folder.is_searchable)
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .where(Document.id.in_(list(document_ids)))
    )).all()

    groups = defaultdict(list)
    for (did, d_active, d_status, d_searchable,
         f_id, f_name, f_desc, f_searchable) in rows:
        key = (effective_searchable(
            is_faq=False, doc_is_active=d_active, doc_status=d_status,
            doc_is_searchable=d_searchable, folder_is_searchable=f_searchable),
            f_id, f_name, f_desc)
        groups[key].append(did)

    n = 0
    for (searchable, f_id, f_name, f_desc), dids in groups.items():
        n += await _update_meta('document_id', dids, {
            'searchable': searchable, 'folder_id': f_id,
            'folder_name': f_name, 'folder_description': f_desc})
    return n


async def sync_meta_faqs_now(session, faq_ids) -> int:
    """FAQ 활성 토글 부분 갱신."""
    rows = (await session.execute(
        select(Faq.id, Faq.is_active).where(Faq.id.in_(list(faq_ids))))).all()
    groups = defaultdict(list)
    for fid, active in rows:
        groups[bool(active)].append(fid)
    n = 0
    for active, fids in groups.items():
        n += await _update_meta('faq_id', fids, {'searchable': active})
    return n
