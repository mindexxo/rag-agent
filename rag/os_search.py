"""OpenSearch 질의 조립·후보 회수 — 검색 경로의 **정의점**.

왕복은 dense kNN · BM25(Nori) 어휘 · 식별자 정확 일치 · 본문 mget이고, 그 위의 융합·리랭크는 `rag/retriever.py`가 한다.
본문·메타는 엔진의 _source에서 읽는다 — 색인이 서빙의 정본이라 정합은 쓰기 경로(`rag/os_index.py`·`rag/outbox.py`)가 책임진다.
`embed_texts`는 `rag/retriever.py`에 둔다 — 옮기면 `tests/conftest.py`의 monkeypatch가 안 먹는다.
"""

from config import settings
from rag.os_client import NORI_FIELD, client
from rag.retrieval_types import RetrievedChunk


def score_to_cosine_distance(score: float) -> float:
    """cosinesimil 점수 (1 + cos)/2 → cosine distance(1 - cos). 게이트(apply_gate) 신호라 틀리면 no_evidence가 어긋난다."""
    return 2.0 - 2.0 * score


def tenant_filter(tenant_id: str) -> dict:
    """테넌트 격리 + 검색가능 필터. 엔진에서 걸어야 한다 — 검색 후 걸러내면 post-filter로 후보가 조용히 깎인다."""
    return {"bool": {"filter": [{"term": {"tenant_id": tenant_id}}, {"term": {"searchable": True}}]}}


def knn_clause(tenant_id: str, vector: list[float], n: int) -> dict:
    """kNN 절. filter를 kNN 안에 넣어 lucene이 탐색 단계에서 거르게 한다 — post-filter의 리콜 손실이 없다."""
    return {"knn": {"dense": {"vector": list(vector), "k": n, "filter": tenant_filter(tenant_id)}}}


def lex_clause(field: str, query_text: str) -> dict:
    """어휘 match 절 — 운영 경로와 eval이 같은 절을 쓰게 한 곳으로 모은다.

    `auto_generate_synonyms_phrase_query: False`: nori가 같은 위치에 대체 토큰을 내면 기본값(True)의 구문 쿼리를
    2.18 hybrid 쿼리가 처리하다 500(read past EOF)으로 죽는다. 질의 term 집합은 같아 리콜 손실은 없다.
    """
    return {"match": {field: {"query": query_text, "auto_generate_synonyms_phrase_query": False}}}


def _ids(resp) -> list[int]:
    return [int(h["_source"]["chunk_id"]) for h in resp["hits"]["hits"]]


# ===== 운영 경로 (rag/retriever.py가 호출) =================================


async def search_dense_per_query(
    tenant_id: str,
    q_embs: list,
    candidates_per_branch: int,
) -> tuple[list[list[int]], list[tuple[int, float]]]:
    """쿼리별 kNN top-N. (쿼리별 id 리스트, 원본 쿼리의 (id, cosine_distance)) 반환.

    두 번째 값은 게이트 신호 전용이라 원본 쿼리(index 0) 결과만 담는다 — 변형 쿼리가 게이트를 흔들지 않게.
    """
    os_client = client()
    per_query_ids: list[list[int]] = []
    dense_results: list[tuple[int, float]] = []
    for i, q_emb in enumerate(q_embs):
        resp = await os_client.search(
            index=settings.opensearch_index,
            body={
                "size": candidates_per_branch,
                "_source": ["chunk_id"],
                "query": knn_clause(tenant_id, q_emb.dense, candidates_per_branch),
            },
        )
        hits = resp["hits"]["hits"]
        if i == 0:
            dense_results = [(int(h["_source"]["chunk_id"]), score_to_cosine_distance(h["_score"])) for h in hits]
        per_query_ids.append([int(h["_source"]["chunk_id"]) for h in hits])
    return per_query_ids, dense_results


async def fetch_chunk_map(ids: list[int]) -> dict:
    """id → RetrievedChunk. 엔진의 _source에서 만든다(PG를 되묻지 않는다) — 색인이 낡으면 낡은 본문이 인용된다.

    검색가능 필터는 다시 걸지 않는다(검색에서 이미 걸렸다). 색인에 없는 id는 조용히 빠진다.
    """
    if not ids:
        return {}
    resp = await client().mget(index=settings.opensearch_index, body={"ids": [str(i) for i in ids]})
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
            table_key=src.get("table_key"),
            table_text=src.get("table_text"),
            folder_name=src.get("folder_name"),
            folder_description=src.get("folder_description"),
            identifiers=list(src.get("identifiers") or []),
            headings=list(src.get("headings") or []),
        )
    return out


async def search_identifiers(tenant_id: str, tokens: list[str], limit: int) -> list[int]:
    """식별자 채널 — `identifiers`에 토큰이 정확히 있는 조각 id. 다른 채널이 놓친 조 번호를 리랭커 풀에 넣는다.

    토큰은 rag.identifiers.normalize_identifiers(질문)로 만든다.
    """
    if not tokens:
        return []
    resp = await client().search(
        index=settings.opensearch_index,
        body={
            "size": limit,
            "_source": ["chunk_id"],
            "query": {
                "bool": {
                    "filter": [tenant_filter(tenant_id)],
                    "must": [{"terms": {"identifiers": tokens}}],
                }
            },
        },
    )
    return _ids(resp)


async def search_lexical(tenant_id: str, query: str, limit: int) -> list[int]:
    """어휘 채널 — BM25(Nori) 상위 limit개의 id.

    Nori 기본 사전은 고유명을 문맥에 따라 다르게 잘라 색인문과 질의문이 어긋날 수 있다(사용자 사전 없음).
    """
    if not query.strip():
        return []
    os_client = client()
    resp = await os_client.search(
        index=settings.opensearch_index,
        body={
            "size": limit,
            "_source": ["chunk_id"],
            "query": {
                "bool": {
                    "filter": [tenant_filter(tenant_id)],
                    "must": [lex_clause(NORI_FIELD, query)],
                }
            },
        },
    )
    return _ids(resp)
