"""OpenSearch 검색 백엔드 (#139) — 매핑·질의 조립의 **정의점**.

`settings.search_backend == 'opensearch'`일 때 `rag/retriever.py`가 dense·어휘 두 채널을
이 모듈로 갈아끼운다. **기본값은 'pg'이고, 운영 전환 선결 조건이 남아 있다** — 무엇이
없는지는 `config.py`의 `search_backend` 주석이 정본이다(색인 동기화, BM25 통계 스코프).

측정 정본은 `eval/report_os_ablation_v1.md`다. 현행 근거로는 이득이 0이다 — kNN 구현 차이 0,
BM25 구현 차이 0, 토크나이저 차이 잡음 수준, 엔진 융합 차이 0. 그래서 이 모듈은 "전환이
가능한가"를 코드로 증명하고 실문서 규모에서 다시 재기 위한 도구이지, 권고안이 아니다.

## 경계 — 무엇을 갈아끼우고 무엇을 남기는가

갈아끼우는 것은 **후보 회수 두 개**뿐이다(`search_dense_per_query`·`search_lexical`).
그 위아래는 백엔드와 무관하므로 손대지 않는다:

- `embed_texts`는 `retriever.py` 안에 남는다 — 옮기면 `tests/conftest.py`의 monkeypatch가
  안 먹고 테스트가 실제 TEI를 때린다(그 파일 주석이 명시).
- 멀티쿼리 RRF 융합(#5)·리랭커·표 필터·top_n 슬라이스·게이트 판정은 그대로다. 엔진 융합
  (`hybrid` 쿼리 + normalization-processor)으로 바꾸지 않은 이유: 실측에서 우리 union 주입과
  결과가 동일했고(`os_hybrid` ≡ `hyb_os_nori`), 바꾸면 멀티쿼리 경로를 다시 설계해야 한다.
- **청크 본문·메타는 PG에서 읽는다**(`_fetch_chunk_map`). OpenSearch에도 넣어두지만 검색
  결과로 쓰지 않는다 — PG가 정본이라, 색인이 낡아도 인용 파일명·버전·페이지가 틀리지 않는다.

## PG와 값으로 맞춘 것 — 변인 격리

어긋나면 "엔진 차이"가 아니라 "설정 차이"를 재게 되어 비교 자체가 무효다.

- **벡터**: `chunks.dense`를 재계산 없이 복사한다. TEI는 호출마다 비결정적이다(실측 1.4e-4).
- **HNSW**: `m=16`·`ef_construction=64`(`schema.sql:111-113`과 동일), `ef_search=40`
  (pgvector 세션 기본값. OpenSearch 기본은 100이라 명시하지 않으면 탐색 폭이 넓은 쪽이 유리).
- **BM25**: `k1=1.5`·`b=0.75`(`rag/lexical.bm25_rank` 기본값. Lucene 기본은 1.2다).
- **거리 환산**: `cosine_distance = 2 - 2*score`. OpenSearch `cosinesimil` 점수가 (1+cos)/2라서다.
  실측으로 확인했다 — 같은 질의·같은 청크에서 pgvector `cosine_distance`와 **소수점 6자리까지
  일치**(2026-09-08, 5건). 게이트 신호가 이 값이므로 환산이 틀리면 `no_evidence` 판정이
  조용히 어긋난다(운영은 게이트가 꺼져 있어 실질 영향은 eval sweep에 국한된다 — `apply_gate`).
- **bigram 토큰**: analyzer로 재현하지 않는다. 사유는 `LEX_BIGRAM_FIELD` 주석.

## 알려진 잔차 — 결과를 읽을 때 감안할 것

1. **df 스코프**: 앱 BM25는 통계를 테넌트 단위로 잡지만(`_search_lexical`의 N·avgdl이
   `WHERE tenant_id`) Lucene의 df는 인덱스 전체다. 단일 인덱스는 운영에서 쓸 모양이지만,
   다른 테넌트의 데이터가 우리 idf를 움직인다 — 실측으로 겪었다(다른 세션이 실문서 211청크를
   넣자 색인 대상이 602→813). 테넌트별 인덱스가 대안이고, 그러면 이 성질이 사라진다.
2. **길이 노름**: Lucene은 필드 길이를 양자화해 저장하므로 `dl`이 정확값이 아니다.
3. 위 둘 때문에 어휘 채널은 **근사 일치**가 기대값이다. dense 쪽은 근거가 더 강하다
   (같은 벡터·같은 거리·같은 파라미터).
"""
from config import settings

DIM = 1024              # chunks.dense와 동일 (schema.sql:102)
HNSW_M = 16             # schema.sql:111-113의 pgvector HNSW와 동일
HNSW_EF_CONSTRUCTION = 64
HNSW_EF_SEARCH = 40     # pgvector 세션 기본값 (OpenSearch 기본은 100)
BM25_K1 = 1.5           # rag/lexical.bm25_rank 기본값 (Lucene 기본 1.2 아님)
BM25_B = 0.75

NORI_FIELD = "index_text_nori"
LEX_BIGRAM_FIELD = "lex_bigram"

# 정규화 융합 파이프라인. RRF(score-ranker-processor)는 2.19에 들어왔고 2.18엔 없다.
# 2.18의 normalization-processor는 채널 가중치를 직접 줄 수 있어 #135의 주입형 설계와
# 대응시키기 좋다. **운영 경로는 이 파이프라인을 쓰지 않는다** — 융합은 우리가 한다(모듈 docstring).
PIPELINE = "kms_norm_hybrid"

MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,      # 샤드를 늘리면 df가 샤드별로 갈려 BM25 점수가 흔들린다
            "number_of_replicas": 0,    # — 변인 격리를 위해 1샤드 고정.
            "knn": True,
            # pgvector 세션 기본값(40)에 맞춘다 — OpenSearch 기본은 100이다.
            #
            # **단, 현 규모에서는 이 값이 쓰이지도 않는다.** OpenSearch는
            # index.knn.advanced.approximate_threshold(기본 15000) 미만이면 HNSW를 건너뛰고
            # 정확 탐색(brute force)을 한다 — 실측 당시 코퍼스는 602청크였다. 그래서 어블레이션
            # 게이트 3의 후보 겹침 1.000은 "HNSW ≡ HNSW"의 근거가 아니라 "정확 탐색 대 (이
            # 규모에선 거의 정확한) pgvector HNSW"의 결과다. 배선 검증으로는 여전히 유효하고,
            # 오히려 근사 잡음이 없어 더 깔끔하다.
            # 두 엔진의 ANN이 실제로 맞붙는 것은 청크가 1만5천을 넘는 규모부터다.
            "knn.algo_param.ef_search": HNSW_EF_SEARCH,
            "similarity": {
                "bm25_kms": {"type": "BM25", "k1": BM25_K1, "b": BM25_B},
            },
        },
        "analysis": {
            "tokenizer": {
                # mixed — 복합어를 통째로도, 조각으로도 남긴다(고유명·상품코드 리콜).
                "nori_kms": {"type": "nori_tokenizer", "decompound_mode": "mixed"},
            },
            "analyzer": {
                # 내장 nori analyzer의 표준 구성을 그대로 쓴다(조사·어미·구두점 제거).
                # 손으로 조정한 구성은 "실제로 운영할 설정"이 아니라 판정 근거로 약하다.
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
            # PG chunks.id 그대로. 검색 결과를 이 값으로 PG에 되물으므로(_fetch_chunk_map)
            # 그리고 eval 채점이 chunk id 매칭이므로 필수다.
            "chunk_id": {"type": "long"},
            # 테넌트 격리 — PG는 RLS 없이 WHERE절이 유일한 방어선이고(rag/models.py:14-26),
            # OpenSearch에서는 그 보장을 filter로 처음부터 다시 세운다.
            "tenant_id": {"type": "keyword"},
            "document_id": {"type": "long"},
            "faq_id": {"type": "long"},
            "page": {"type": "integer"},
            "version": {"type": "integer"},
            "is_table": {"type": "boolean"},
            "filename": {"type": "keyword"},
            "heading_path": {"type": "keyword"},
            # 확인·디버깅용으로만 둔다. 색인하면 어휘 점수에 이중으로 관여한다.
            # 검색 결과의 본문은 PG에서 읽는다(모듈 docstring).
            "text": {"type": "text", "index": False},
            # 질의할 때는 반드시 lex_clause()를 쓴다 — 기본 match로 질의하면 2.18 hybrid가
            # 500으로 죽는다(실측 47/450). 사유·실측치는 lex_clause docstring.
            NORI_FIELD: {"type": "text", "analyzer": "nori_ko", "similarity": "bm25_kms"},
            LEX_BIGRAM_FIELD: {
                # rag.lexical.bigrams()의 출력을 공백으로 이어 붙인 문자열이 들어온다.
                # analyzer로 재현하지 않는 이유: bigrams()는 어절 내 문자 bigram이면서
                # **1글자 어절을 그대로 보존**하는데(rag/lexical.py:24-38), OpenSearch의 ngram
                # 토큰 필터는 min_gram 미달 토큰을 버려 그 동작이 재현되지 않는다. 흉내내면
                # bigram 대조군이 대조군이 아니게 되므로, 정의점 함수를 직접 호출해 토큰을
                # 만들고 여기서는 공백으로만 자른다 — 토큰이 바이트 단위로 같아져
                # 엔진 BM25와 앱 BM25의 차이가 구현 차이로 좁혀진다.
                "type": "text",
                "analyzer": "pretokenized",
                "similarity": "bm25_kms",
            },
            "dense": {
                "type": "knn_vector",
                "dimension": DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",       # cosinesimil 지원 + 필터를 kNN 탐색 단계에서 처리
                    "space_type": "cosinesimil",
                    "parameters": {"m": HNSW_M, "ef_construction": HNSW_EF_CONSTRUCTION},
                },
            },
        },
    },
}

_client = None


def client():
    """공용 AsyncOpenSearch 싱글톤.

    `opensearchpy`를 함수 안에서 import한다 — eval 전용 선택 의존(requirements의 주석 블록)
    이라, 미설치 환경에서도 이 모듈이 import 가능해야 한다. `settings.search_backend`가
    'pg'인 기본 경로는 이 함수에 도달하지 않는다.

    커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제다(rag/clients.py와 동일 제약).
    """
    global _client
    if _client is None:
        from opensearchpy import AsyncOpenSearch
        _client = AsyncOpenSearch([settings.opensearch_url],
                                  timeout=settings.opensearch_timeout,
                                  retry_on_timeout=True, max_retries=2)
    return _client


def lex_text(text: str, filename: str | None, heading_path) -> str:
    """어휘·Nori 필드의 입력 텍스트 — 인제스션·운영 어휘 채널과 **동일 조립**.

    문서는 '파일명>헤딩' 프리픽스(build_index_text), FAQ는 원문 그대로다. 이 비대칭은
    인제스션이 그렇게 넣기 때문이고(rag/index_text.py docstring), 운영 어휘 채널
    (rag/retriever.py의 _search_lexical)이 같은 규약을 쓴다. 폴더 설명은 넣지 않는다 —
    리랭커 전용이다(임베딩·어휘엔 미포함).

    두 벌로 갈라지면 어블레이션 눈금이 운영을 예언하지 못한다 — 그래서 매핑과 같은 파일에 둔다.
    """
    from rag.index_text import build_index_text
    if filename is None:
        return text
    return build_index_text(text, filename, list(heading_path or []))


def score_to_cosine_distance(score: float) -> float:
    """OpenSearch cosinesimil 점수 → pgvector cosine_distance.

    `cosinesimil` 점수가 (1 + cos)/2이므로 distance = 1 - cos = 2 - 2*score다.
    실측으로 확인했다: 같은 질의·같은 청크에서 pgvector `cosine_distance`와 소수점 6자리까지
    일치(2026-09-08, 5건). 게이트 신호가 이 값이므로(apply_gate) 환산이 틀리면 no_evidence
    판정이 조용히 어긋난다.
    """
    return 2.0 - 2.0 * score


def tenant_filter(tenant_id: str) -> dict:
    return {"term": {"tenant_id": tenant_id}}


def knn_clause(tenant_id: str, vector: list[float], n: int) -> dict:
    """kNN 절. filter를 kNN **안에** 넣는다 — lucene 엔진은 이걸 탐색 단계에서 처리하므로
    상위 k를 뽑은 뒤 걸러내는 post-filter와 달리 리콜이 조용히 깎이지 않는다.

    (PG는 HNSW LIMIT 이후 `_searchable_condition()`을 WHERE절로 거는 post-filter다 —
    rag/retriever.py의 그 함수 주석. 실문서에서 구판·비활성 문서 비중이 커지면 이 차이가
    OpenSearch의 실질 이득이 될 수 있다. #138에서 별도로 잰다.)
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

    운영 경로는 hybrid 쿼리를 쓰지 않지만(융합은 우리가 한다) 같은 절을 공유한다 —
    eval이 재는 것과 운영이 도는 것이 갈라지면 눈금이 운영을 예언하지 못한다.
    """
    return {"match": {field: {"query": query_text,
                              "auto_generate_synonyms_phrase_query": False}}}


def _ids(resp) -> list[int]:
    return [int(h["_source"]["chunk_id"]) for h in resp["hits"]["hits"]]


# ===== 운영 경로 (retriever가 백엔드 분기로 호출) =========================

async def search_dense_per_query(
    tenant_id: str,
    q_embs: list,
    candidates_per_branch: int,
) -> tuple[list[list[int]], list[tuple[int, float]]]:
    """`rag.retriever._search_dense_per_query`의 OpenSearch 대응물 — **반환 계약 동일**.

    (쿼리별 id 리스트, 원본 쿼리의 (id, cosine_distance)) 를 돌려준다. 두 번째 값은 게이트
    신호 전용이고 원본 쿼리(index 0) 결과만 담는다 — 변형 쿼리가 의미를 이탈해도 게이트가
    흔들리지 않게 하려는 PG 쪽 규약을 그대로 지킨다.

    거리는 `score_to_cosine_distance`로 환산한다(실측 검증됨).

    검색 가능 여부(`_searchable_condition`)를 여기서 걸지 않는다 — 색인이 이미 그 조건을
    통과한 청크만 담고 있다는 전제다. 그 전제가 **스냅샷**이라는 것이 운영 전환의 선결
    조건이다(config.py의 search_backend 주석).

    쿼리 수만큼 순차 호출한다 — PG 경로와 같은 순서·같은 후보 수를 유지해 비교를 성립시킨다.
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


async def search_lexical(tenant_id: str, query: str, limit: int) -> list[int]:
    """`rag.retriever._search_lexical`의 OpenSearch 대응물 — **반환 계약 동일**(상위 id 리스트).

    엔진의 네이티브 형태소 분석기(Nori)로 채점한다. bigram 필드로도 질의할 수 있지만 그러면
    Lucene을 BM25 계산기로만 쓰는 셈이고, 실측에서 앱 BM25와 결과가 동일했다
    (`hyb_os_bigram` ≡ `hyb_bigram`, 채널 단독 리콜 436 동일). 엔진을 들이는 값을 하려면
    엔진의 분석기를 써야 하므로 Nori를 쓴다.

    **단서**: Nori 기본 사전은 코퍼스의 고유명을 제대로 못 자른다 — '아더에러'가 단독으로는
    ['에러'], 문장 안에서는 ['아더','에러']로 갈려 색인문↔질의문이 어긋난다(실측).
    `rag/lexical.py:16-19`가 kiwi를 기각한 그 비대칭이다. 사용자 사전이 해법이고 그 유지
    비용이 판정 대상이다(#138).
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


# ===== eval 전용 헬퍼 (어블레이션 variant·게이트가 쓴다) ==================

async def ensure_pipeline(os_client, weights: tuple[float, float] = (0.7, 0.3)) -> None:
    """정규화 융합 검색 파이프라인 등록 — (dense, 어휘) 가중치.

    min_max 정규화 + arithmetic_mean 결합. weights 합은 1.0이어야 하고 hybrid 쿼리의
    서브쿼리 개수와 길이가 같아야 한다(2.18 normalization-processor 계약).
    기본 0.7/0.3은 dense 우세인 현행 체계(주입 상방 0/450)를 반영한 출발점일 뿐이다.
    **어블레이션 전용** — 운영 경로는 이 파이프라인을 타지 않는다.
    """
    assert abs(sum(weights) - 1.0) < 1e-9, f"weights 합이 1.0이 아니다: {weights}"
    await os_client.transport.perform_request(
        "PUT", f"/_search/pipeline/{PIPELINE}", body={
            "description": "#139 dense+lexical 정규화 융합 (RRF는 2.19+, 2.18엔 없음)",
            "phase_results_processors": [{
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {"technique": "arithmetic_mean",
                                    "parameters": {"weights": list(weights)}},
                },
            }],
        })


async def knn_ids(os_client, tenant_id: str, vector: list[float], n: int) -> list[int]:
    resp = await os_client.search(index=settings.opensearch_index, body={
        "size": n, "_source": ["chunk_id"],
        "query": knn_clause(tenant_id, vector, n),
    })
    return _ids(resp)


async def bm25_ids(os_client, tenant_id: str, field: str, query_text: str,
                   n: int) -> list[int]:
    """어휘 채널 단독. query_text는 필드에 맞춰 호출부가 준비한다 —
    NORI_FIELD면 원문 질의, LEX_BIGRAM_FIELD면 ' '.join(rag.lexical.bigrams(질의))."""
    if not query_text.strip():
        return []
    resp = await os_client.search(index=settings.opensearch_index, body={
        "size": n, "_source": ["chunk_id"],
        "query": {"bool": {"filter": [tenant_filter(tenant_id)],
                           "must": [lex_clause(field, query_text)]}},
    })
    return _ids(resp)


async def hybrid_ids(os_client, tenant_id: str, vector: list[float], query_text: str,
                     field: str, n: int) -> list[int]:
    """엔진 융합 — hybrid 쿼리 + normalization-processor. ensure_pipeline이 먼저."""
    resp = await os_client.search(
        index=settings.opensearch_index, params={"search_pipeline": PIPELINE}, body={
            "size": n, "_source": ["chunk_id"],
            "query": {"hybrid": {"queries": [
                knn_clause(tenant_id, vector, n),
                {"bool": {"filter": [tenant_filter(tenant_id)],
                          "must": [lex_clause(field, query_text)]}},
            ]}},
        })
    return _ids(resp)


async def analyze(os_client, analyzer: str, text: str) -> list[str]:
    """_analyze API — 토큰 값 대조 게이트가 쓴다."""
    resp = await os_client.indices.analyze(
        index=settings.opensearch_index, body={"analyzer": analyzer, "text": text})
    return [t["token"] for t in resp["tokens"]]


# ===== 색인 쓰기 경로 — 정합 관리 3층 =====================================
#
# PG는 청크와 어휘 색인을 **한 트랜잭션**에 쓴다(#135 B안: "별도 정합 관리가 없다"가
# rag/documents.py·faq_indexing.py 주석에 그대로 적혀 있다). 외부 엔진을 들이면 그 성질을
# 잃는다 — 그 비용을 어떻게 관리하는지가 아래 3층이다.
#
# ## 설계의 출발점 — 불일치 두 방향의 위험도가 다르다
#
#   OS에 있고 PG엔 없음/비검색  → 삭제·비공개 문서가 답변에 인용되는 **오답**
#   PG에 있고 OS엔 없음         → 못 찾는 **누락**(no_evidence로 귀결). 오답은 아니다
#
# 그래서 오답 방향은 **구조적으로 불가능하게** 만들고, 누락 방향만 최선노력 + 재동기화로
# 다룬다. 두 방향을 같은 비용으로 막으려 하면 과설계가 된다.
#
# ## 1층 — 읽기 권위 (오답 차단, 쓰기 작업 0)
#
# 검색은 id만 OpenSearch에서 받고 본문·메타는 PG로 되묻는다. 그 되묻는 쿼리에
# `_searchable_condition()`을 걸면(retriever._fetch_chunk_map의 searchable_only) PG를
# 통과하지 못하는 것은 결과에 못 들어온다. 덕분에 **문서 삭제·검색토글·폴더 off·FAQ off는
# 색인을 건드리지 않아도 즉시 반영된다** — 아래 2층 함수가 이 사건들을 다룰 필요가 없다.
# PG 경로의 post-filter와 같은 모양이라 새 개념도 아니다.
#
# ## 2층 — 이중 쓰기 (누락 최소화, 아래 함수들)
#
# 청크가 **생기고 사라지는** 지점에서 PG 커밋 *후*에 OS에 반영한다. 커밋 전에 하면 PG가
# 롤백됐는데 OS만 반영되는 더 나쁜 불일치가 된다.
#
# **실패해도 예외를 올리지 않는다.** PG는 이미 커밋됐고, 여기서 터뜨리면 OpenSearch 장애가
# 곧 업로드 장애가 된다 — 파생 인덱스가 정본의 가용성을 깎는 건 뒤바뀐 설계다. 대신 실패가
# 조용해지는 대가를 지표로 갚는다(metrics.SEARCH_INDEX_SYNC_TOTAL, result='error').
#
# `refresh='wait_for'`를 쓴다 — OpenSearch는 기본 1초 refresh라 쓰자마자 검색되지 않는다.
# 업로드 직후 검색이 되는 PG 의미에 최대한 붙이려는 것이고, 비용은 인제스션 끝에 최대 1초다
# (파싱·임베딩이 이미 초 단위라 무의미한 증가다).
#
# ## 3층 — 재동기화 (최종 안전판)
#
# `python -m eval._os_index --repair`가 PG↔OS의 id 집합을 대조해 누락분을 색인하고 잉여분을
# 지운다(reconcile). 2층이 놓친 것을 여기서 줍는다. 크론으로 돌릴 수 있다.
#
# ## 지금 하지 않는 것 — outbox (승격 경로)
#
# 정석은 outbox 테이블이다: PG 트랜잭션 안에 "이 청크들을 색인하라"는 행을 남기고 arq 워커가
# 드레인한다. 2층의 최선노력과 달리 프로세스가 죽어도 반영이 보장되고, 3층 배치를 기다리지
# 않는다. 지금 안 하는 이유는 값이다 — 테이블·마이그레이션·잡을 늘리는데, 이 백엔드는 품질
# 이득이 0으로 측정된 실험 경로다(eval/report_os_ablation_v1.md).
# **승격 트리거: 색인 지연이 SLA가 될 때** — 즉 "업로드하고 몇 초 뒤엔 반드시 검색돼야 한다"가
# 요구사항이 되는 순간. 그때는 이 주석을 지우고 outbox로 바꿔라.


def enabled() -> bool:
    """이 백엔드가 켜져 있는지. 쓰기 훅들이 첫 줄에서 이걸 보고 조용히 빠진다 —
    호출부(인제스션·라우터)가 백엔드를 알 필요가 없게 하려는 것이다."""
    return settings.search_backend == 'opensearch'


def build_doc(chunk, filename: str | None, version: int | None) -> dict:
    """OpenSearch 문서 본문 — **정의점**. 색인 배치(eval/_os_index)와 이중 쓰기가 같이 쓴다.

    두 벌로 갈라지면 배치로 만든 색인과 운영이 만든 색인의 모양이 달라지고, 그러면
    어블레이션 눈금이 운영을 예언하지 못한다.

    filename=None은 FAQ 청크다 — 'FAQ'로 적는다(_fetch_chunk_map과 동일 규약).
    """
    from rag.lexical import bigrams
    lex = lex_text(chunk.text, filename, chunk.heading_path)
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
        "text": chunk.text,
        NORI_FIELD: lex,
        LEX_BIGRAM_FIELD: " ".join(bigrams(lex)),
        # 임베딩을 재계산하지 않는다 — TEI는 호출마다 비결정적이다(실측 1.4e-4).
        # PG에 저장된 값을 그대로 옮긴다. float32 → float 변환은 json 직렬화 때문.
        "dense": [float(x) for x in chunk.dense],
    }


async def bulk_index(docs: list[dict]) -> int:
    """_bulk 색인. _id=chunk_id라 재실행이 중복을 만들지 않고 upsert가 된다.

    **_bulk은 원자적이지 않다** — 건별로 성공/실패한다. 부분 실패를 삼키면 색인이 조용히
    비므로 첫 오류를 예외로 올린다(호출부가 잡아 지표로 기록한다).
    """
    import json
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


async def _index_rows(session, *, document_ids=None, faq_ids=None) -> list[dict]:
    """색인할 청크를 PG에서 읽어 문서 본문으로 조립한다.

    `_searchable_condition()`을 걸어 검색 가능한 것만 담는다 — 비검색 청크를 넣어도 1층이
    걸러내므로 오답은 안 되지만, 인덱스 전체 df를 흔들어 어휘 점수를 움직인다
    (모듈 docstring의 "df 스코프" 잔차).

    지연 import: retriever가 이 모듈을 (함수 안에서) 참조하므로 톱레벨로 올리면 순환이다.
    """
    from sqlalchemy import select

    from rag.models import Chunk, Document, Faq, Folder
    from rag.retriever import _searchable_condition

    stmt = (
        select(Chunk, Document.filename, Document.version)
        .outerjoin(Document, Chunk.document_id == Document.id)
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .outerjoin(Faq, Chunk.faq_id == Faq.id)
        .where(_searchable_condition())
    )
    if document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))
    if faq_ids is not None:
        stmt = stmt.where(Chunk.faq_id.in_(list(faq_ids)))
    return [build_doc(c, fn, ver) for c, fn, ver in (await session.execute(stmt)).all()]


async def _run(op: str, coro):
    """쓰기 훅 공통 래퍼 — 실패를 삼키고 지표·경고로 드러낸다(사유는 모듈 상단 2층 설명)."""
    import logging

    from rag.metrics import SEARCH_INDEX_SYNC_TOTAL
    logger = logging.getLogger(__name__)
    try:
        n = await coro
        SEARCH_INDEX_SYNC_TOTAL.labels(op=op, result='ok').inc()
        return n
    except Exception as e:                     # noqa: BLE001 — 삼키는 것이 의도다
        SEARCH_INDEX_SYNC_TOTAL.labels(op=op, result='error').inc()
        logger.warning('검색 색인 동기화 실패 (%s) — 색인이 낡았다. '
                       '복구: python -m eval._os_index --repair · 원인=%s', op, e)
        return 0


async def sync_documents(session, document_ids) -> int:
    """문서들의 청크를 색인에 반영(upsert). 인제스션 커밋 **후**에 부른다."""
    if not enabled() or not document_ids:
        return 0
    return await _run('sync_documents', _sync_documents(session, document_ids))


async def _sync_documents(session, document_ids) -> int:
    return await bulk_index(await _index_rows(session, document_ids=document_ids))


async def sync_after_ingest(document_id: int, superseded_ids) -> int:
    """인제스션 커밋 후 훅 — 새 문서 색인 + supersede된 구버전 청크 제거.

    **자기 세션을 직접 연다**(호출부의 세션은 이미 닫혔다). 그리고 세션 열기까지 포함해
    전부 `_run`이 감싸므로 **이 함수는 어떤 경우에도 예외를 올리지 않는다** — 호출부
    (rag/documents.py)가 이걸 try 안에서 부르는데, 여기서 예외가 나면 이미 커밋된 인제스션이
    `_mark_failed`로 뒤집힌다. 파생 인덱스가 정본의 상태를 망치는 건 뒤바뀐 설계다.
    """
    if not enabled():
        return 0
    return await _run('sync_after_ingest', _sync_after_ingest(document_id, superseded_ids))


async def _sync_after_ingest(document_id: int, superseded_ids) -> int:
    from database import AsyncSessionLocal
    async with AsyncSessionLocal() as session:
        n = await bulk_index(await _index_rows(session, document_ids=[document_id]))
    if superseded_ids:
        await _delete_by_terms('document_id', list(superseded_ids))
    return n


async def drop_documents(document_ids) -> int:
    """문서들의 청크를 색인에서 제거. PG 청크 삭제 커밋 **후**에 부른다.

    1층이 이미 오답을 막으므로 이건 위생(인덱스 크기·df)과 재동기화 부담 절감용이다.
    """
    if not enabled() or not document_ids:
        return 0
    return await _run('drop_documents', _delete_by_terms('document_id', list(document_ids)))


async def sync_faqs(session, faq_ids) -> int:
    if not enabled() or not faq_ids:
        return 0
    return await _run('sync_faqs', _sync_faqs(session, faq_ids))


async def _sync_faqs(session, faq_ids) -> int:
    # FAQ 재인덱싱은 기존 청크를 지우고 새로 넣는다(faq_indexing.reindex_faq와 같은 모양) —
    # chunk_id가 바뀌므로 옛 문서를 먼저 지워야 유령이 남지 않는다.
    await _delete_by_terms('faq_id', list(faq_ids))
    return await bulk_index(await _index_rows(session, faq_ids=faq_ids))


async def drop_faqs(faq_ids) -> int:
    if not enabled() or not faq_ids:
        return 0
    return await _run('drop_faqs', _delete_by_terms('faq_id', list(faq_ids)))


async def reconcile(session) -> dict:
    """3층 — PG↔OS id 집합 대조 후 복구. `eval/_os_index --repair`가 부른다.

    반환: {'pg': n, 'os': n, 'indexed': 누락분 색인 수, 'deleted': 잉여분 삭제 수}
    2층이 놓친 것(OpenSearch 장애 중 업로드·삭제)을 여기서 줍는다.
    """
    from sqlalchemy import select

    from rag.models import Chunk, Document, Faq, Folder
    from rag.retriever import _searchable_condition

    # **순서가 계약이다: OpenSearch를 먼저 읽고 PG를 나중에 읽는다.**
    # 두 스냅샷 사이에 인제스션이 끼어들 수 있고, 그때 어느 쪽이 먼저인지가 결과를 가른다:
    #   PG 먼저 → 새 청크가 pg_ids에 없고 os_ids에는 있다 → 'extra'로 판정돼 **삭제된다**.
    #             방금 정상 색인된 청크를 재동기화가 지우는, 이 함수가 막아야 할 사고다.
    #   OS 먼저 → 새 청크가 os_ids에 없고 pg_ids에는 있다 → 'missing'으로 판정돼 재색인된다.
    #             이미 있는 것을 다시 넣는 것이므로 _id=chunk_id upsert에서 무해하다.
    # 잔여 경합을 무해한 방향으로 떨어뜨리는 것이 이 모듈의 설계 원칙과 같다(상단 참조).
    os_client = client()
    os_ids: set[int] = set()
    resp = await os_client.search(index=settings.opensearch_index, scroll='2m',
                                  body={"size": 1000, "_source": ["chunk_id"],
                                        "query": {"match_all": {}}})
    while resp["hits"]["hits"]:
        os_ids.update(int(h["_source"]["chunk_id"]) for h in resp["hits"]["hits"])
        resp = await os_client.scroll(scroll_id=resp["_scroll_id"], scroll='2m')
    await os_client.clear_scroll(scroll_id=resp["_scroll_id"])   # 스크롤 컨텍스트 반납

    pg_ids = set((await session.execute(
        select(Chunk.id)
        .outerjoin(Document, Chunk.document_id == Document.id)
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .outerjoin(Faq, Chunk.faq_id == Faq.id)
        .where(_searchable_condition())
    )).scalars().all())

    missing, extra = pg_ids - os_ids, os_ids - pg_ids
    indexed = 0
    if missing:
        from rag.models import Chunk as C
        stmt = (select(C, Document.filename, Document.version)
                .outerjoin(Document, C.document_id == Document.id)
                .where(C.id.in_(list(missing))))
        docs = [build_doc(c, fn, ver) for c, fn, ver in (await session.execute(stmt)).all()]
        indexed = await bulk_index(docs)
    deleted = await _delete_by_terms('chunk_id', list(extra)) if extra else 0
    return {'pg': len(pg_ids), 'os': len(os_ids), 'indexed': indexed, 'deleted': deleted}
