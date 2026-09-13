"""OpenSearch 질의 조립·후보 회수 — 검색 경로의 **정의점** (#146, 구 rag/opensearch.py에서 분리).

왕복은 셋뿐이다: dense kNN(`search_dense_per_query`) · BM25(Nori) 어휘 채널(`search_lexical`) ·
본문 mget(`fetch_chunk_map`). 그 위(멀티쿼리 RRF·리랭커·표 필터·top_n·게이트)는 엔진과 무관한
조합이고 `rag/retriever.py`가 맡는다.

## 경계

- **청크 본문·메타는 엔진에서 읽는다**(`fetch_chunk_map`) — 실무 표준 구성. 왕복 한 번으로
  끝나고 필터도 엔진에서 걸린다. 대가는 **색인이 서빙의 정본**이라는 것 — 색인이 낡으면 낡은
  텍스트·메타가 인용된다. 그 대가는 쓰기 경로(`rag/os_index.py`·`rag/outbox.py`)가 갚는다.
- 융합은 우리가 한다(union 주입, #135). 엔진 `hybrid` 쿼리로 바꾸지 않은 이유: A/B 실측에서
  결과가 같았고, 바꾸면 멀티쿼리 경로를 다시 설계해야 한다. RRF 프로세서는 2.19+다.
- `embed_texts`는 `rag/retriever.py` 안에 남는다 — 옮기면 `tests/conftest.py`의 monkeypatch가
  안 먹고 테스트가 실제 TEI를 때린다(그 파일 주석이 명시).
- 검색가능·테넌트 필터는 **엔진에서** 건다(`tenant_filter`). 그 플래그의 정합은 쓰기 경로가
  책임진다 — 매핑·필드 정의는 `rag/os_client.py`.
"""
from config import settings
from rag.os_client import NORI_FIELD, client
from rag.retrieval_types import RetrievedChunk


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
    갱신하고, 놓친 것은 `rag.os_reconcile`이 줍는다.

    검색가능 필터는 이미 엔진에서 걸렸다(tenant_filter) — 여기서 다시 걸지 않는다.
    mget으로 한 번에 읽는다. 색인에 없는 id는 조용히 빠진다(호출부가 순서대로 재조립한다).
    """
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
