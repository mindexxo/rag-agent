"""OpenSearch — 질의 조립·재동기화. 연결·매핑은 `rag/os_client.py`, 색인 쓰기는 `rag/os_index.py`가
정의점이다 (#146 분리 진행 중 — 이 파일은 남은 관심사가 옮겨가면 사라진다).

청크(본문·메타·벡터·어휘 필드)는 여기에만 있다. PG는 문서·FAQ·폴더·대화·캐시·outbox의 정본이고
청크 행을 쓰지 않는다. `rag/retriever.py`는 후보 회수 두 개(`search_dense_per_query`·
`search_lexical`)와 본문 조회(`fetch_chunk_map`)를 이 모듈에서 하고, 그 위(멀티쿼리 RRF·리랭커·
표 필터·top_n·게이트)는 엔진과 무관하다.

## 경계

- `embed_texts`는 `retriever.py` 안에 남는다 — 옮기면 `tests/conftest.py`의 monkeypatch가
  안 먹고 테스트가 실제 TEI를 때린다(그 파일 주석이 명시).
- 융합은 우리가 한다(union 주입, #135). 엔진 `hybrid` 쿼리로 바꾸지 않은 이유: A/B 실측에서
  결과가 같았고, 바꾸면 멀티쿼리 경로를 다시 설계해야 한다. RRF 프로세서는 2.19+다.
- **청크 본문·메타는 엔진에서 읽는다**(`fetch_chunk_map`) — 실무 표준 구성. 왕복 한 번으로
  끝나고 필터도 엔진에서 걸린다. 대가는 **색인이 서빙의 정본**이라는 것 — 색인이 낡으면
  낡은 텍스트·메타가 인용된다. 그 대가를 쓰기 경로(`rag/os_index.py`·`rag/outbox.py`)가 갚는다.
"""
from config import settings
from rag.os_client import NORI_FIELD, client
from rag.os_index import _delete_by_terms


def score_to_cosine_distance(score: float) -> float:
    """OpenSearch cosinesimil 점수 → pgvector cosine_distance.

    `cosinesimil` 점수가 (1 + cos)/2이므로 distance = 1 - cos = 2 - 2*score다.
    실측으로 확인했다: 같은 질의·같은 청크에서 pgvector `cosine_distance`와 소수점 6자리까지
    일치(2026-09-08, 5건). 게이트 신호가 이 값이므로(apply_gate) 환산이 틀리면 no_evidence
    판정이 조용히 어긋난다.
    """
    return 2.0 - 2.0 * score


def tenant_filter(tenant_id: str) -> dict:
    """테넌트 격리 + 검색가능 필터. **둘 다 엔진에서 걸어야 한다** — 검색 후 걸러내면
    상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다(매핑의 searchable 주석)."""
    return {"bool": {"filter": [{"term": {"tenant_id": tenant_id}},
                                {"term": {"searchable": True}}]}}


def knn_clause(tenant_id: str, vector: list[float], n: int) -> dict:
    """kNN 절. filter를 kNN **안에** 넣는다 — lucene 엔진은 이걸 탐색 단계에서 처리하므로
    상위 k를 뽑은 뒤 걸러내는 post-filter와 달리 리콜이 조용히 깎이지 않는다.

    (구 PG 경로는 HNSW LIMIT 뒤에 WHERE절을 거는 post-filter였다 — 구판·비활성 문서 비중이
    커질수록 리콜이 조용히 깎이는 함정. 엔진 필터는 그 함정이 없다.)
    """
    return {"knn": {"dense": {"vector": list(vector), "k": n,
                              "filter": tenant_filter(tenant_id)}}}


def lex_clause(field: str, query_text: str) -> dict:
    """어휘 match 절. 운영 경로와 eval variant가 **같은 절**을 쓰게 한 곳으로 모은다.

    `auto_generate_synonyms_phrase_query: False`가 왜 필요한가 — **2.18 hybrid 쿼리의 실측
    결함 회피**다. nori는 같은 위치에 대체 토큰을 낸다('주나요'→'주','줏' /
    '빨대'→'빨대','빨','대'). 기본값(True)에서는 match가 그래프·구문 쿼리로 컴파일되고,
    hybrid가 그것을 처리하다 500으로 죽는다:
        read past EOF (pos=2147483647) ... [slice=_0.nvd]
    (2147483647 = Lucene NO_MORE_DOCS — 소진된 norms 이터레이터에서 읽는 패턴.)

    실측(gold v2 450문항, 2026-09-08 · OpenSearch 2.18.0/Lucene 9.12.0):
      hybrid + nori   실패 47건 10.4%   ← 기본값(True)
      hybrid + bigram 실패  0건  0.0%   ← whitespace analyzer라 같은 위치 토큰이 없다
      bm25   + nori   실패  0건  0.0%   ← 단독 BM25 경로는 무해
    즉 트리거는 hybrid × 형태소 분석기다. 이 플래그를 걸면 다섯 경로(hybrid×2·bm25×2·knn)가
    450/450 통과한다. 질의 term 집합은 그대로이므로 리콜 손실은 없다.

    시도했다 **버린** 우회: 검색 analyzer에 flatten_graph. posLen>1(복합어 '적립금')은
    사라지지만 같은 위치 토큰('주'/'줏')이 남아 47건이 그대로 실패했다. 이 플래그를 넣으면
    flatten_graph는 불필요해진다(없이도 0/450 — 실측). 재시도하지 마라.

    운영 경로는 hybrid 쿼리를 쓰지 않아(융합은 우리가 한다) 이 결함에 노출되지 않지만,
    플래그는 그대로 둔다 — 질의 term 집합이 같아 무해하고, 실측 기록을 코드에 남긴다.
    """
    return {"match": {field: {"query": query_text,
                              "auto_generate_synonyms_phrase_query": False}}}


def _ids(resp) -> list[int]:
    return [int(h["_source"]["chunk_id"]) for h in resp["hits"]["hits"]]


# ===== 운영 경로 (rag/retriever.py가 호출) =================================

async def search_dense_per_query(
    tenant_id: str,
    q_embs: list,
    candidates_per_branch: int,
) -> tuple[list[list[int]], list[tuple[int, float]]]:
    """쿼리별 kNN top-N. (쿼리별 id 리스트, 원본 쿼리의 (id, cosine_distance)) 반환.

    두 번째 값은 게이트 신호 전용이고 원본 쿼리(index 0) 결과만 담는다 — 변형 쿼리가 의미를
    이탈해도 게이트 판정이 흔들리지 않게. 거리는 `score_to_cosine_distance`로 환산한다.

    검색 가능 여부는 `tenant_filter()`의 `searchable` 플래그가 kNN 탐색 단계에서 건다 —
    그 플래그의 정합은 쓰기 경로(outbox)가 책임진다. 쿼리 수만큼 순차 호출한다.
    """
    os_client = client()
    per_query_ids: list[list[int]] = []
    dense_results: list[tuple[int, float]] = []
    for i, q_emb in enumerate(q_embs):
        resp = await os_client.search(index=settings.opensearch_index, body={
            "size": candidates_per_branch,
            "_source": ["chunk_id"],
            "query": knn_clause(tenant_id, q_emb.dense, candidates_per_branch),
        })
        hits = resp["hits"]["hits"]
        if i == 0:
            dense_results = [(int(h["_source"]["chunk_id"]),
                              score_to_cosine_distance(h["_score"])) for h in hits]
        per_query_ids.append([int(h["_source"]["chunk_id"]) for h in hits])
    return per_query_ids, dense_results


async def fetch_chunk_map(ids: list[int]) -> dict:
    """id → RetrievedChunk. **엔진의 _source에서 직접 만든다 — PG를 되묻지 않는다.**

    실무 표준 구성이다: 검색 엔진이 텍스트·메타를 함께 들고 있으므로 왕복 한 번으로 끝난다.
    대가는 색인이 정본으로 쓰인다는 것 — 색인이 낡으면 낡은 텍스트·메타가 인용된다. 그래서
    쓰기 경로가 정합을 책임진다(모듈 상단 "정합 관리 3층"): 삭제·메타 변경은 즉시 부분
    갱신하고, 놓친 것은 `eval.os_reconcile`이 줍는다.

    검색가능 필터는 이미 엔진에서 걸렸다(tenant_filter) — 여기서 다시 걸지 않는다.
    mget으로 한 번에 읽는다. 색인에 없는 id는 조용히 빠진다(호출부가 순서대로 재조립한다).
    """
    # 자료형은 0층 leaf(rag/retrieval_types.py)라 순환이 없다 — 이 모듈을 0층으로 유지하려는
    # 규율 때문에 아직 함수 안에 있을 뿐이다(#146 분리에서 톱레벨로 올라간다).
    from rag.retrieval_types import RetrievedChunk
    if not ids:
        return {}
    resp = await client().mget(index=settings.opensearch_index,
                               body={"ids": [str(i) for i in ids]})
    out = {}
    for d in resp["docs"]:
        if not d.get("found"):
            continue
        src = d["_source"]
        out[int(src["chunk_id"])] = RetrievedChunk(
            chunk_id=int(src["chunk_id"]),
            document_id=src.get("document_id"),
            text=src.get("text") or '',
            heading_path=list(src.get("heading_path") or []),
            page=src.get("page"),
            filename=src.get("filename") or 'FAQ',
            version=src.get("version") or 1,
            faq_id=src.get("faq_id"),
            is_table=bool(src.get("is_table")),
            folder_name=src.get("folder_name"),
            folder_description=src.get("folder_description"),
        )
    return out


async def search_lexical(tenant_id: str, query: str, limit: int) -> list[int]:
    """어휘 채널 — BM25(Nori) 상위 limit개의 id.

    엔진의 형태소 분석기(Nori)로 채점한다. bigram 필드로도 질의할 수 있지만 A/B 실측에서
    앱 BM25와 결과가 같았다 — 엔진을 들인 값을 하려면 엔진의 분석기를 쓴다.

    **단서**: Nori 기본 사전은 코퍼스의 고유명을 제대로 못 자른다 — '아더에러'가 단독으로는
    ['에러'], 문장 안에서는 ['아더','에러']로 갈려 색인문↔질의문이 어긋난다(실측). 사용자
    사전 운영 절차가 후속 이슈다.
    """
    if not query.strip():
        return []
    os_client = client()
    resp = await os_client.search(index=settings.opensearch_index, body={
        "size": limit,
        "_source": ["chunk_id"],
        "query": {"bool": {
            "filter": [tenant_filter(tenant_id)],
            "must": [lex_clause(NORI_FIELD, query)],
        }},
    })
    return _ids(resp)


def _tenant_term(tenant_id):
    return {"term": {"tenant_id": tenant_id}} if tenant_id else {"match_all": {}}


async def _indexed_parent_ids(field: str, tenant_id) -> set[int]:
    """색인에 있는 부모 id 집합(document_id 또는 faq_id) — terms 집계. 반드시 PG보다 **먼저** 읽는다."""
    agg = await client().search(index=settings.opensearch_index, body={
        "size": 0, "query": _tenant_term(tenant_id),
        "aggs": {"p": {"terms": {"field": field, "size": 65536}}}})
    return {int(b["key"]) for b in agg["aggregations"]["p"]["buckets"]}


async def reconcile(session, tenant_id: str | None = None) -> dict:
    """PG↔OS **문서** 단위 재동기화 — 안전판. `python -m eval.os_reconcile`이 부른다.

    tenant_id를 주면 그 테넌트만 본다 — 공유 DB에서 한 테넌트를 손볼 때, 그리고 테스트가
    남의 문서를 pending으로 되돌리지 않게. 없으면 전체.
    반환: {'pg': ready 문서 수, 'os': 색인 문서 수, 'indexed': 재등재 수, 'deleted': 삭제 청크 수}.
    PG에 청크가 없으므로 "ready 문서가 색인에도 있는가"만 본다 — 청크 수준 드리프트(일부 누락)는
    못 잡고 outbox의 문서 단위 원자성에 의존한다.

    **읽는 순서가 계약이다: OpenSearch 먼저, PG 나중.** 두 스냅샷 사이에 인제스션이 끼면
    새 문서는 pg_docs에만 있어 'missing'으로 판정돼 재등재된다 — 이미 있는 것을 다시 넣는
    무해한 방향이다. 반대 순서면 os_docs에만 있어 'extra'로 판정돼 **방금 색인된 청크를 지운다.**
    """
    from sqlalchemy import select, update

    from rag import outbox
    from rag.models import Document

    os_docs = await _indexed_parent_ids("document_id", tenant_id)
    stmt = select(Document.id, Document.tenant_id).where(Document.status == 'ready')
    if tenant_id:
        stmt = stmt.where(Document.tenant_id == tenant_id)
    pg_rows = (await session.execute(stmt)).all()
    pg_docs = {did for did, _ in pg_rows}
    tenant_of = dict(pg_rows)

    missing, extra = pg_docs - os_docs, os_docs - pg_docs
    # 색인에 없는 ready 문서는 pending으로 되돌려 대기열에 넣는다 — 재파싱·재임베딩·ready 승격은
    # 인제스션 핸들러 한 곳(rag/documents.index_pending_document)이 맡는다. 두 벌을 두지 않는다.
    if missing:
        await session.execute(update(Document).where(Document.id.in_(list(missing))).values(status='pending'))
        for did in sorted(missing):
            outbox.enqueue(session, tenant_of[did], outbox.INDEX_DOCUMENT, document_id=did)
        await session.commit()
    deleted = await _delete_by_terms('document_id', sorted(extra)) if extra else 0
    return {'pg': len(pg_docs), 'os': len(os_docs), 'indexed': len(missing), 'deleted': deleted,
            'unit': 'document'}


async def reconcile_faqs(session, tenant_id: str | None = None) -> dict:
    """PG↔OS **FAQ** 단위 재동기화 — reconcile()의 FAQ 짝. 순서 계약 동일(OS 먼저).

    FAQ는 활성 여부와 무관하게 전부 색인 대상이다(비활성은 searchable=False로 들어간다 —
    effective_searchable). 그래서 PG의 모든 FAQ 행과 대조한다. 누락은 INDEX_FAQ 행으로 등재
    (워커가 재임베딩·색인), 잉여(PG에 없는 faq_id)는 색인에서 지운다.
    반환: {'pg': FAQ 수, 'os': 색인 FAQ 수, 'indexed': 재등재 수, 'deleted': 삭제 청크 수}.
    """
    from sqlalchemy import select

    from rag import outbox
    from rag.models import Faq

    os_faqs = await _indexed_parent_ids("faq_id", tenant_id)
    stmt = select(Faq.id, Faq.tenant_id)
    if tenant_id:
        stmt = stmt.where(Faq.tenant_id == tenant_id)
    pg_rows = (await session.execute(stmt)).all()
    pg_faqs = {fid for fid, _ in pg_rows}
    tenant_of = dict(pg_rows)

    missing, extra = pg_faqs - os_faqs, os_faqs - pg_faqs
    if missing:
        for fid in sorted(missing):
            outbox.enqueue(session, tenant_of[fid], outbox.INDEX_FAQ, faq_id=fid)
        await session.commit()
    deleted = await _delete_by_terms('faq_id', sorted(extra)) if extra else 0
    return {'pg': len(pg_faqs), 'os': len(os_faqs), 'indexed': len(missing), 'deleted': deleted,
            'unit': 'faq'}
