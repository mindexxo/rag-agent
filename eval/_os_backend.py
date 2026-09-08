"""OpenSearch 검색 백엔드 — #139 A/B 실험 전용. 매핑·질의 조립의 **정의점**.

운영 코드(`rag/`)는 이 모듈을 쓰지 않는다. 이번 스콥은 "검색 계층만 교체했을 때 품질이
달라지는가"를 재는 것이고, 기각될 수도 있는 실험을 위해 운영 검색 경로를 고치지 않는다.
승격되면 이 모듈의 매핑·질의를 `rag/`로 옮긴다.

## 변인 격리 — 이 파일의 존재 이유

PG(pgvector + FTS bigram + 앱 BM25)와 OpenSearch를 비교할 때, 조건이 하나라도 어긋나면
"엔진 차이"가 아니라 "설정 차이"를 재게 된다. 그래서 아래를 PG와 **값으로** 맞춘다:

- **벡터**: `chunks.dense`를 재계산 없이 그대로 복사. 차원 1024, 코사인.
- **HNSW**: `m=16`, `ef_construction=64` — `schema.sql:111-113`의 pgvector 인덱스와 동일.
  엔진은 lucene을 쓴다. `cosinesimil`을 지원하고(faiss는 l2/innerproduct 계열),
  필터를 kNN 탐색 단계에서 처리한다.
- **BM25 파라미터**: `k1=1.5`, `b=0.75` — `rag/lexical.bm25_rank` 기본값과 동일.
  Lucene 기본 k1은 1.2라 그대로 두면 파라미터 차이가 구현 차이로 오독된다.
- **bigram 토큰**: analyzer로 흉내내지 않고 `rag.lexical.bigrams()`가 자른 토큰을 공백으로
  이어 붙여 색인하고, 필드 analyzer는 `whitespace`로 둔다. 이유는 `LEX_BIGRAM_FIELD` 주석.

## 알려진 잔차 — 측정 결과를 읽을 때 감안할 것

1. **df 스코프**: 우리 앱 BM25는 통계를 테넌트 단위로 잡는다(`_search_lexical`의 N·avgdl이
   `WHERE tenant_id`). Lucene의 df는 **인덱스 전체**다. 단일 인덱스 + 테넌트 필터 구성은
   운영에서 실제로 쓸 모양이지만, 이 때문에 `os_bigram`과 `hyb_bigram`은 토큰이 같아도
   idf가 다르다. 잔차가 크면 테넌트별 인덱스 구성을 후속 variant로 잰다.
2. **길이 노름**: Lucene은 필드 길이를 양자화해 저장하므로 `dl`이 정확값이 아니다.
3. 위 둘 때문에 `os_bigram` ≈ `hyb_bigram`은 **근사 일치**가 기대값이고, 정확 일치는 아니다.
   반면 `os_knn` ≈ `baseline`은 근거가 더 강하다(같은 벡터·같은 거리·같은 파라미터·같은 색인
   집합) — 그래서 이쪽을 배선 게이트로 쓴다(`_os_index.py`의 게이트 3).

접속: 환경변수 `OS_URL`(기본 http://localhost:9200) · `OS_INDEX`(기본 kms_chunks_v1).
"""
import os

INDEX = os.getenv("OS_INDEX", "kms_chunks_v1")
URL = os.getenv("OS_URL", "http://localhost:9200")

DIM = 1024              # chunks.dense와 동일 (schema.sql:102)
HNSW_M = 16             # schema.sql:111-113의 pgvector HNSW와 동일
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40     # pgvector 세션 기본값과 동일 (OpenSearch 기본은 100 — 맞춰야 대조가 성립)
BM25_K1 = 1.5           # rag/lexical.bm25_rank 기본값과 동일 (Lucene 기본 1.2 아님)
BM25_B = 0.75

NORI_FIELD = "index_text_nori"
LEX_BIGRAM_FIELD = "lex_bigram"

# 정규화 융합 파이프라인 (#139). RRF(score-ranker-processor)는 2.19에 들어왔고 2.18엔 없다 —
# 2.18의 normalization-processor를 쓴다. 순위만 쓰고 점수를 버리는 RRF와 달리 채널 가중치를
# 직접 줄 수 있어 #135의 주입형 설계와 대응시키기 좋다.
PIPELINE = "kms_norm_hybrid"

MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,      # 실험 규모(수백~수천 청크). 샤드를 늘리면 df가 샤드별로
            "number_of_replicas": 0,    # 갈려 BM25 점수가 흔들린다 — 변인 격리를 위해 1샤드 고정.
            "knn": True,
            # pgvector 세션 기본값(40)에 맞춘다 — OpenSearch 기본은 100이라 그대로 두면
            # 탐색 폭이 넓은 쪽이 유리해져 "엔진 차이"가 아니라 "파라미터 차이"를 재게 된다.
            #
            # **단, 현 코퍼스 규모에서는 이 값이 쓰이지도 않는다.** OpenSearch는
            # index.knn.advanced.approximate_threshold(기본 15000) 미만이면 HNSW를 건너뛰고
            # 정확 탐색(brute force)을 한다 — 우리 코퍼스는 602청크다(실측). 그래서 게이트 3의
            # 후보 겹침 1.000은 "HNSW ≡ HNSW"의 근거가 아니라 "정확 탐색 대 (거의) 정확한
            # pgvector HNSW"의 결과로 읽어야 한다. 배선 검증(space_type·정규화·필터)으로는
            # 여전히 유효하고 오히려 근사 잡음이 없어 더 깔끔하다.
            # 청크가 1만5천을 넘는 규모(#138 실문서 또는 그 이후)에서 비로소 두 엔진의 ANN이
            # 맞붙고, 그때 이 값이 실제로 대조를 성립시킨다. 강제로 ANN을 켜서 비교하고 싶으면
            # approximate_threshold를 0으로 내리면 되지만, 실제로 운영할 구성은 기본값이다.
            "knn.algo_param.ef_search": HNSW_EF_SEARCH,
            "similarity": {
                "bm25_kms": {"type": "BM25", "k1": BM25_K1, "b": BM25_B},
            },
        },
        "analysis": {
            "tokenizer": {
                # mixed — 복합어를 통째로도, 조각으로도 남긴다. 고유명·상품코드 리콜을 위해.
                "nori_kms": {"type": "nori_tokenizer", "decompound_mode": "mixed"},
            },
            "analyzer": {
                # 조사·어미·구두점을 떨어내는 nori_part_of_speech를 포함한다 — 기존 kiwi 채널이
                # 내용어만 남기는 것(_hybrid_ablation._kiwi_tokens)과 같은 성격으로 맞춘 것.
                "nori_ko": {
                    "type": "custom",
                    "tokenizer": "nori_kms",
                    "filter": ["nori_part_of_speech", "lowercase"],
                },
                # 색인·질의 모두 이미 토큰화된 문자열이 들어온다 — 공백으로만 자른다.
                "pretokenized": {"type": "custom", "tokenizer": "whitespace"},
            },
        },
    },
    "mappings": {
        "properties": {
            # PG chunks.id 그대로. 채점(eval/retrieval.score_one)이 chunk id 매칭이라 필수다.
            "chunk_id": {"type": "long"},
            # 테넌트 격리 — PG는 RLS 없이 WHERE절이 유일한 방어선이고(rag/models.py:14-26),
            # OpenSearch에서는 그 보장을 filter로 처음부터 다시 세운다.
            "tenant_id": {"type": "keyword"},
            "document_id": {"type": "long"},
            "faq_id": {"type": "long"},
            "page": {"type": "integer"},
            "version": {"type": "integer"},
            "is_table": {"type": "boolean"},      # _keep_single_table이 쓴다
            "filename": {"type": "keyword"},
            "heading_path": {"type": "keyword"},
            # 원문은 확인·디버깅용으로만 둔다. 색인하면 어휘 점수에 이중으로 관여한다.
            "text": {"type": "text", "index": False},
            # 이 필드를 질의할 때는 반드시 _lex_clause()를 쓴다 — 기본 match로 질의하면
            # 2.18 hybrid가 500으로 죽는다(실측 47/450). 사유·실측치는 _lex_clause docstring.
            NORI_FIELD: {"type": "text", "analyzer": "nori_ko", "similarity": "bm25_kms"},
            LEX_BIGRAM_FIELD: {
                # rag.lexical.bigrams()의 출력을 공백으로 이어 붙인 문자열이 들어온다.
                # analyzer로 재현하지 않는 이유: bigrams()는 어절 내 문자 bigram이면서
                # **1글자 어절을 그대로 보존**하는데(rag/lexical.py:24-38), OpenSearch의 ngram
                # 토큰 필터는 min_gram 미달 토큰을 버려 그 동작이 재현되지 않는다. 흉내내면
                # bigram 대조군이 대조군이 아니게 되므로, 정의점 함수를 직접 호출해 토큰을
                # 만들고 여기서는 공백으로만 자른다 — 토큰이 바이트 단위로 동일해져
                # os_bigram vs hyb_bigram의 차이가 BM25 구현 차이로 좁혀진다.
                "type": "text",
                "analyzer": "pretokenized",
                "similarity": "bm25_kms",
            },
            "dense": {
                "type": "knn_vector",
                "dimension": DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",
                    "space_type": "cosinesimil",
                    "parameters": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
                },
            },
        },
    },
}


def client():
    """OpenSearch 클라이언트. 로컬 실험용이라 인증 없음(docker-compose.opensearch.yml 참조)."""
    from opensearchpy import OpenSearch
    return OpenSearch([URL], timeout=60, retry_on_timeout=True, max_retries=2)


def ensure_pipeline(os_client, weights: tuple[float, float] = (0.7, 0.3)) -> None:
    """정규화 융합 검색 파이프라인 등록 (dense, 어휘) 가중치.

    min_max 정규화 + arithmetic_mean 결합. weights 합은 1.0이어야 하고 hybrid 쿼리의
    서브쿼리 개수와 길이가 같아야 한다(2.18 normalization-processor 계약).
    기본 0.7/0.3은 dense 우세인 현행 체계(#128에서 어휘 채널 주입 상방 0/450)를 반영한
    출발점일 뿐 — 측정에서 흔들어 볼 값이다.
    """
    assert abs(sum(weights) - 1.0) < 1e-9, f"weights 합이 1.0이 아니다: {weights}"
    os_client.transport.perform_request("PUT", f"/_search/pipeline/{PIPELINE}", body={
        "description": "#139 dense+lexical 정규화 융합 (RRF는 2.19+, 2.18엔 없음)",
        "phase_results_processors": [{
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {"weights": list(weights)},
                },
            },
        }],
    })


def _tenant_filter(tenant_id: str) -> dict:
    return {"term": {"tenant_id": tenant_id}}


def _knn_clause(tenant_id: str, vector: list[float], n: int) -> dict:
    """kNN 절. filter를 kNN 안에 넣는다 — lucene 엔진은 이걸 탐색 단계에서 처리하므로
    상위 k를 뽑은 뒤 걸러내는 post-filter와 달리 리콜이 조용히 깎이지 않는다.
    (PG는 HNSW LIMIT 이후 WHERE절로 거는 post-filter다 — rag/retriever.py:57-67 주석.
    이번엔 searchable 청크만 색인해 이 차이를 변인에서 뺐다. #138에서 따로 잰다.)"""
    return {"knn": {"dense": {"vector": list(vector), "k": n, "filter": _tenant_filter(tenant_id)}}}


def lex_text(text: str, filename: str | None, heading_path) -> str:
    """어휘·Nori 필드의 입력 텍스트 — 운영과 **동일 조립**.

    문서는 '파일명>헤딩' 프리픽스(build_index_text), FAQ는 원문 그대로다. 이 비대칭은
    인제스션이 그렇게 넣기 때문이고(rag/index_text.py docstring), 운영 어휘 채널
    (rag/retriever.py:245-251)과 기존 어블레이션(_hybrid_ablation._load_corpus)이 같은 규약을
    쓴다. 폴더 설명은 넣지 않는다 — 리랭커 전용이다(임베딩·어휘엔 미포함).

    두 벌로 갈라지면 어블레이션 눈금이 운영을 예언하지 못한다 — 그래서 매핑과 같은 파일에 둔다.
    """
    from rag.index_text import build_index_text
    if filename is None:
        return text
    return build_index_text(text, filename, list(heading_path or []))


def _lex_clause(field: str, query_text: str) -> dict:
    """어휘 match 절. `os_nori`와 `os_hybrid`가 **같은 절**을 쓰게 한 곳으로 모은다 —
    둘이 다른 절을 쓰면 "융합의 상방"을 재는 대조가 성립하지 않는다.

    `auto_generate_synonyms_phrase_query: False`가 왜 필요한가 — **2.18 hybrid 쿼리의 실측
    결함 회피**다. nori는 같은 위치에 대체 토큰을 낸다('주나요'→'주','줏' / '빨대'→'빨대','빨','대').
    기본값(True)에서는 match가 그래프·구문 쿼리로 컴파일되고, hybrid가 그것을 처리하다
    500으로 죽는다:  read past EOF (pos=2147483647) ... [slice=_0.nvd]
    (2147483647 = Lucene NO_MORE_DOCS — 소진된 norms 이터레이터에서 읽는 패턴).

    실측(gold v2 450문항, 2026-09-08 · 로컬 OpenSearch 2.18.0/Lucene 9.12.0):
      hybrid + nori   실패 47건 10.4%   ← 기본값
      hybrid + bigram 실패  0건  0.0%   ← whitespace analyzer라 같은 위치 토큰이 없다
      bm25   + nori   실패  0건  0.0%   ← 단독 BM25 경로는 무해
    즉 트리거는 hybrid × 형태소 분석기다. flatten_graph를 검색 analyzer에 걸어도(posLen>1은
    사라진다) 같은 위치 토큰은 남아 해결되지 않았다 — 이 플래그가 그것까지 덮는다.
    질의 term 집합은 그대로이므로 리콜 손실은 없다. 이 플래그를 걸면 다섯 경로(hybrid×2·
    bm25×2·knn)가 450/450 통과한다.

    시도했다 **버린** 우회: 검색 analyzer에 flatten_graph를 걸어 그래프를 없애는 방법.
    posLen>1(복합어 '적립금')은 사라지지만 같은 위치의 대체 토큰('주'/'줏')이 남아 47건이
    그대로 실패했다. 이 플래그를 넣으면 flatten_graph는 불필요해진다(없이도 0/450 — 실측).

    **이 우회가 필요하다는 사실 자체가 운영 비교표의 항목이다** — 한국어에서 같은 위치 토큰은
    예외가 아니라 상식적 빈도이고, 엔진의 하이브리드 융합을 형태소 분석기와 함께 쓰는 것이
    2.18에서 곧바로 되지 않는다는 뜻이다.
    """
    return {"match": {field: {"query": query_text,
                              "auto_generate_synonyms_phrase_query": False}}}


def _ids(resp) -> list[int]:
    return [int(h["_source"]["chunk_id"]) for h in resp["hits"]["hits"]]


def knn_ids(os_client, tenant_id: str, vector: list[float], n: int) -> list[int]:
    """dense 채널만 — PG _search_dense_per_query의 대응물."""
    resp = os_client.search(index=INDEX, body={
        "size": n,
        "_source": ["chunk_id"],
        "query": _knn_clause(tenant_id, vector, n),
    })
    return _ids(resp)


def bm25_ids(os_client, tenant_id: str, field: str, query_text: str, n: int) -> list[int]:
    """어휘 채널만 — Lucene BM25. field에 따라 Nori/bigram 두 대조군이 갈린다.

    query_text는 필드에 맞춰 호출부가 준비한다: NORI_FIELD면 원문 질의,
    LEX_BIGRAM_FIELD면 ' '.join(rag.lexical.bigrams(질의)).
    """
    if not query_text.strip():
        return []
    resp = os_client.search(index=INDEX, body={
        "size": n,
        "_source": ["chunk_id"],
        "query": {"bool": {
            "filter": [_tenant_filter(tenant_id)],
            "must": [_lex_clause(field, query_text)],
        }},
    })
    return _ids(resp)


def hybrid_ids(os_client, tenant_id: str, vector: list[float], query_text: str,
               field: str, n: int) -> list[int]:
    """엔진 융합 — hybrid 쿼리 + normalization-processor. 파이프라인은 ensure_pipeline이 먼저."""
    resp = os_client.search(index=INDEX, params={"search_pipeline": PIPELINE}, body={
        "size": n,
        "_source": ["chunk_id"],
        "query": {"hybrid": {"queries": [
            _knn_clause(tenant_id, vector, n),
            {"bool": {"filter": [_tenant_filter(tenant_id)],
                      "must": [_lex_clause(field, query_text)]}},
        ]}},
    })
    return _ids(resp)


def analyze(os_client, analyzer: str, text: str) -> list[str]:
    """_analyze API — 게이트 4(토큰 값 대조)가 쓴다."""
    resp = os_client.indices.analyze(index=INDEX, body={"analyzer": analyzer, "text": text})
    return [t["token"] for t in resp["tokens"]]
