"""OpenSearch 검색 백엔드 (#139) — 매핑·질의 조립의 **정의점**.

`settings.search_backend == 'opensearch'`일 때 `rag/retriever.py`가 dense·어휘 두 채널을
이 모듈로 갈아끼운다. **기본값은 'pg'다.** 운영 전환 시 받아들이는 것(색인 반영 지연 1~5분)과
남은 것(BM25 통계 스코프, PG 벡터 컬럼 처분)은 `config.py`의 `search_backend` 주석이 정본이다.

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
- **청크 본문·메타는 엔진에서 읽는다**(`fetch_chunk_map`) — 실무 표준 구성이다. 검색 엔진이
  텍스트와 문서 메타를 함께 들고 있으므로 왕복 한 번으로 끝나고, 필터도 엔진에서 걸린다.
  대가는 **색인이 서빙의 정본이 된다**는 것 — 색인이 낡으면 낡은 텍스트·메타가 인용된다.
  그 대가를 쓰기 경로가 갚는다(아래 "정합 관리").

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
            # PG chunks.id 그대로. 색인·삭제의 멱등 키(_id)이자 outbox 연산의 대상이고,
            # eval 채점이 chunk id 매칭이라 필수다.
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
            # 검색 엔진에는 조인이 없다 — 필터·표시에 쓰는 문서 메타를 청크마다 복사한다
            # (실무 표준). 대가는 fan-out update다: 문서·폴더 메타가 바뀌면 그 문서의 모든
            # 청크 문서를 갱신해야 한다(sync_meta_documents·sync_meta_faqs).
            #
            # `searchable` = PG _searchable_condition()의 비정규화. **필터를 엔진에서 걸기
            # 위해** 반드시 여기 있어야 한다. 검색 후 PG로 걸러내면 상위 k를 뽑은 뒤 빼는
            # post-filter가 되어 후보가 조용히 깎인다(rag/retriever.py:57-67의 그 함정).
            # 그래서 비검색 청크도 색인하고 이 플래그로 가른다 — 토글 on/off가 문서 추가·삭제가
            # 아니라 플래그 갱신이 되어 fan-out이 가벼워진다.
            "searchable": {"type": "boolean"},
            "folder_id": {"type": "long"},            # 폴더 단위 fan-out update의 필터 키
            "folder_name": {"type": "keyword"},
            # 폴더 설명은 리랭커 입력에만 들어간다(rag/index_text.py) — 검색 대상이 아니다.
            "folder_description": {"type": "text", "index": False},
            # index:false — 검색 대상은 아래 두 어휘 필드다(여기까지 색인하면 어휘 점수에
            # 이중으로 관여한다). 하지만 _source로는 돌려받는다: 검색 결과의 본문이 이 값이다
            # (fetch_chunk_map — PG를 되묻지 않는다).
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
    """테넌트 격리 + 검색가능 필터. **둘 다 엔진에서 걸어야 한다** — 검색 후 걸러내면
    상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다(매핑의 searchable 주석)."""
    return {"bool": {"filter": [{"term": {"tenant_id": tenant_id}},
                                {"term": {"searchable": True}}]}}


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


async def fetch_chunk_map(ids: list[int]) -> dict:
    """id → RetrievedChunk. **엔진의 _source에서 직접 만든다 — PG를 되묻지 않는다.**

    실무 표준 구성이다: 검색 엔진이 텍스트·메타를 함께 들고 있으므로 왕복 한 번으로 끝난다.
    대가는 색인이 정본으로 쓰인다는 것 — 색인이 낡으면 낡은 텍스트·메타가 인용된다. 그래서
    쓰기 경로가 정합을 책임진다(모듈 상단 "정합 관리 3층"): 삭제·메타 변경은 즉시 부분
    갱신하고, 놓친 것은 `_os_index --repair`가 줍는다.

    검색가능 필터는 이미 엔진에서 걸렸다(tenant_filter) — 여기서 다시 걸지 않는다.
    mget으로 한 번에 읽는다. 색인에 없는 id는 조용히 빠진다(호출부가 순서대로 재조립한다).
    """
    from rag.retriever import RetrievedChunk    # 지연 import — 순환 회피
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


# ===== 색인 쓰기 경로 — 정합 관리 =========================================
#
# PG는 청크와 어휘 색인을 **한 트랜잭션**에 썼다(#135 B안: "별도 정합 관리가 없다"). 외부 엔진을
# 들이면 그 성질을 잃는다. 그 비용을 어떻게 관리하는지가 이 절이고, 정의점은 rag/outbox.py다.
#
# ## 전제 — 색인이 서빙의 정본이다
#
# 실무 표준 구성: 검색 결과의 텍스트·메타를 엔진이 돌려준다(`fetch_chunk_map`). PG로 되묻어
# 검증하지 않는다. 그래서 "색인이 낡으면 낡은 답이 나간다"가 구조적으로 가능하고, 그 자리를
# 아래가 메운다.
#
# ## 1) 필터를 엔진에 비정규화
#
# `searchable`(PG `_searchable_condition()`의 비정규화)·`tenant_id`를 청크 문서마다 넣고
# 엔진에서 필터한다. 검색 후 PG로 걸러내면 상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가
# 조용히 깎인다. 비검색 청크도 색인한다 — 토글 on/off가 문서 추가·삭제가 아니라 플래그
# 부분 갱신(`_update_meta`)이 되고, 벡터 없는 구성에서 재임베딩을 피한다.
#
# ## 2) 모든 변경은 트랜잭셔널 outbox로 — 처리는 단일 워커 cron 1분
#
# 업로드·삭제·토글·FAQ 변경은 라우터가 PG 변경과 **같은 트랜잭션**에 대기열 행을 남긴다
# (rag/outbox.enqueue). 워커 cron이 1분마다 id 순으로 처리한다. 커밋 후 직접 호출·인라인
# 처리는 **없다** — 아래 원시 연산들은 drain만 부른다.
#
# 인제스션(INDEX_DOCUMENT)은 rag/documents.index_pending_document가 파싱·임베딩·색인을 한
# 뒤 **마지막 커밋 하나**에 ready·supersede·done을 넣는다 — `ready ≡ 색인됨`이 커밋 단위로
# 성립한다. 색인 시 `searchable`은 그 커밋이 만들 상태(ready·active)로 미리 계산해 켜진 채로
# 넣고, 구버전은 그 뒤에 지운다 — 빈 창이 없다.
#
# **제품 결정 — 반영 지연을 받아들인다.** 삭제·비검색·FAQ 수정도 cron까지 최대 1분(재시도
# 포함 1~5분) 검색에 반영되지 않는다: "문서 변경은 검색에 최대 1~5분 뒤 반영될 수 있다"가
# 가이드다. 답변 캐시는 라우터가 즉시 무효화하므로 창은 새 검색에만 열린다. 좁히려면 워커
# 폴링 루프(10초)나 "지금 반영" 수동 트리거를 얹으면 되고 둘 다 이 구조 위에 그대로 붙는다.
#
# ## 3) 재동기화 — 안전판
#
# `python -m eval._os_index --repair`가 PG↔OS를 대조해 복구한다(`reconcile`). PG가 청크를 안
# 드는 구성에서는 **문서 단위**로만 본다 — 청크 일부 누락은 못 잡고 outbox의 원자성에 의존한다.
#
# 아래 원시 연산들은 **실패를 삼키지 않는다** — 예외를 올려 outbox가 횟수를 세고 MAX_ATTEMPTS에
# failed로 확정하게 한다. 전부 멱등이다: 색인은 _id=chunk_id upsert(결정적 id), 삭제는 없는 것을
# 지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이라 재시도가 겹쳐도 결과가 같다.


def enabled() -> bool:
    """이 백엔드가 켜져 있는지. 쓰기 훅들이 첫 줄에서 이걸 보고 조용히 빠진다 —
    호출부(인제스션·라우터)가 백엔드를 알 필요가 없게 하려는 것이다."""
    return settings.search_backend == 'opensearch'


def pg_stores_chunks() -> bool:
    """PG가 청크 **행 자체**를 드는지 (#139) = 백엔드가 pg인지.

    엔진이 검색·본문·메타를 모두 맡는 구성에서는 PG에 청크를 넣을 이유가 없다 — 서빙 경로가
    한 번도 읽지 않는 행을 쓰는 것이라 순수 낭비다. 그래서 `search_backend='opensearch'`면
    인제스션이 PG INSERT를 건너뛰고 파싱 결과를 곧장 색인한다(index_parsed_document).

    **되돌리기는 자유롭지 않다.** 엔진 구성으로 도는 동안 올라간 문서는 PG에 청크가 0건이라,
    `search_backend`를 pg로 되돌려도 그 문서들은 PG 검색에 잡히지 않는다 — 재인제스트가 필요하다.
    기존에 pg로 올린 문서는 그대로 검색된다(청크가 남아 있다).
    """
    return settings.search_backend != 'opensearch'


def chunk_os_id(*, document_id: int | None = None, faq_id: int | None = None,
                chunk_index: int = 0) -> int:
    """PG 시퀀스 없이 만드는 **결정적** chunk_id.

    PG가 청크 행을 안 들면 BIGSERIAL이 없다. (부모 id, chunk_index)로 계산해 쓴다 —
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
    """PG `_searchable_condition()`을 파이썬으로 옮긴 것 — 색인 시점에 계산해 넣는다.

    두 벌이 갈라지면 엔진 필터와 PG 필터가 다른 답을 낸다. 조건을 바꿀 때는
    rag/retriever.py의 `_searchable_condition()`과 **함께** 고쳐라.
    문서: 활성 + ready + 검색가능 + (폴더 없음 또는 폴더 검색가능) / FAQ: 활성.
    """
    if is_faq:
        return bool(faq_is_active)
    return bool(doc_is_active and doc_status == 'ready' and doc_is_searchable
                and (folder_is_searchable is None or folder_is_searchable))


def build_doc(chunk, filename: str | None, version: int | None,
              dense: list[float] | None = None, *,
              folder_id: int | None = None, folder_name: str | None = None,
              folder_description: str | None = None, searchable: bool = True) -> dict:
    """OpenSearch 문서 본문 — **정의점**. 색인 배치(eval/_os_index)와 이중 쓰기가 같이 쓴다.

    두 벌로 갈라지면 배치로 만든 색인과 운영이 만든 색인의 모양이 달라지고, 그러면
    어블레이션 눈금이 운영을 예언하지 못한다.

    filename=None은 FAQ 청크다 — 'FAQ'로 적는다(_fetch_chunk_map과 동일 규약).

    **dense를 명시로 받는 이유**: 엔진 구성에서는 PG에 청크 행이 없어 PG 벡터가 없다.
    그때는 인제스션이 방금 계산한 임베딩을 그대로 넘긴다 — PG를 경유하지 않으므로 왕복도
    없고, 컬럼이 없어도 동작한다. 넘기지 않으면 PG 값을 쓴다(전환 1단계·전량 색인 경로).
    """
    from rag.lexical import bigrams
    lex = lex_text(chunk.text, filename, chunk.heading_path)
    vec = dense if dense is not None else getattr(chunk, 'dense', None)
    if vec is None:
        # 조용히 벡터 없는 문서를 색인하면 kNN에서 안 잡히는 유령이 된다 — 크게 실패한다.
        raise ValueError(
            f'chunk {getattr(chunk, "id", "?")}: 벡터가 없다. PG에 dense가 없으면 '
            f'호출부가 dense=를 넘겨야 한다(재색인 경로는 _fill_vectors가 재임베딩한다).')
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
        # float32 → float 변환은 json 직렬화 때문. PG 값을 쓸 때는 재계산하지 않는다 —
        # TEI는 호출마다 비결정적이다(실측 1.4e-4).
        "dense": [float(x) for x in vec],
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

    stmt = _index_stmt()
    if document_ids is not None:
        stmt = stmt.where(Chunk.document_id.in_(list(document_ids)))
    if faq_ids is not None:
        stmt = stmt.where(Chunk.faq_id.in_(list(faq_ids)))
    return await _rows_to_docs((await session.execute(stmt)).all())


def _index_stmt():
    """색인 대상 조회 — **검색 가능 여부로 걸러내지 않는다.**

    비검색 청크도 색인하고 `searchable` 플래그로 가른다(매핑 주석). 그래야 토글 on/off가
    문서 추가·삭제가 아니라 플래그 부분 갱신이 되고, 벡터가 없는 구성에서 재임베딩을 피한다.

    지연 import: retriever가 이 모듈을 함수 안에서 참조하므로 톱레벨로 올리면 순환이다.
    """
    from sqlalchemy import select

    from rag.models import Chunk, Document, Faq, Folder
    return (
        select(Chunk, Document.filename, Document.version,
               Document.is_active, Document.status, Document.is_searchable,
               Folder.id, Folder.name, Folder.description, Folder.is_searchable,
               Faq.is_active)
        .outerjoin(Document, Chunk.document_id == Document.id)
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .outerjoin(Faq, Chunk.faq_id == Faq.id)
    )


def _row_meta(row) -> dict:
    """_index_stmt 행 → build_doc의 메타 키워드."""
    (_c, filename, version, d_active, d_status, d_searchable,
     f_id, f_name, f_desc, f_searchable, faq_active) = row
    return dict(
        filename=filename, version=version,
        folder_id=f_id, folder_name=f_name, folder_description=f_desc,
        searchable=effective_searchable(
            is_faq=filename is None, doc_is_active=d_active, doc_status=d_status,
            doc_is_searchable=d_searchable, folder_is_searchable=f_searchable,
            faq_is_active=faq_active),
    )


async def _rows_to_docs(rows) -> list[dict]:
    """(Chunk, filename, version) 행들 → OpenSearch 문서. 벡터가 없으면 **재임베딩**한다.

    PG에 청크는 있으나 벡터 컬럼이 비어 있는 경우(향후 컬럼 DROP 전 단계)에 전량 색인·재동기화가
    타는 경로다. 재임베딩 입력은 `lex_text()` — 인제스션이 임베딩에 넣은 것과 같은 조립이라
    (rag/documents.py의 index_texts, rag/index_text.build_index_text) 원래 벡터를 재현한다.
    TEI가 호출마다 비결정적이라 완전히 같은 값은 아니지만(실측 1.4e-4) 검색 품질에는 잡음
    수준이다.

    **이 재임베딩이 "PG에서 벡터를 버리는" 선택의 값이다** — 색인 재구축이 '몇 초'에서
    'TEI 배치'로 올라간다. 배치 분할은 embed_texts가 담당한다(TEI 상한 32).
    """
    need = [r for r in rows if getattr(r[0], 'dense', None) is None]
    vectors: dict[int, list[float]] = {}
    if need:
        from rag.embeddings import embed_texts
        texts = [lex_text(r[0].text, r[1], r[0].heading_path) for r in need]
        embs = await embed_texts(texts)
        vectors = {r[0].id: e.dense for r, e in zip(need, embs)}
    out = []
    for row in rows:
        chunk, meta = row[0], _row_meta(row)
        out.append(build_doc(chunk, meta.pop('filename'), meta.pop('version'),
                             dense=vectors.get(chunk.id), **meta))
    return out


# 아래 연산들은 **실패를 삼키지 않는다** — 예외를 올려 outbox가 재시도·백오프를 걸게 한다
# (rag/outbox.py). 삼킴과 지표는 그쪽 한 곳에 모여 있다. 전부 멱등이다: 색인은
# _id=chunk_id upsert, 삭제는 없는 것을 지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이라
# 재시도가 겹쳐 두 번 처리해도 결과가 같다(단일 워커 cron만 부른다 — 인라인 없음).


async def _update_meta(field: str, values: list, meta: dict) -> int:
    """_update_by_query로 메타 필드만 갱신한다 — **재색인하지 않는다.**

    재색인하면 PG가 벡터를 안 드는 구성에서 재임베딩이 따라온다(_rows_to_docs). 메타 하나
    바꾸는데 GPU를 태우는 건 뒤바뀐 설계다. 그래서 부분 갱신이 이 경로의 유일한 선택이다.

    conflicts=proceed: 같은 문서를 동시에 재색인 중이면 버전 충돌이 날 수 있다. 그때 전체를
    실패시키기보다 넘긴다 — 놓친 것은 재동기화(--repair)와 outbox 재시도가 잡고, 여기서
    멈추면 나머지 청크가 옛 메타로 남는 쪽이 더 나쁘다.

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
    """PG 행 없이 build_doc에 넘길 최소 형태 — 파싱 결과(chunk_file 산출물)의 어댑터.

    PG가 청크를 안 들 때 쓴다. id는 chunk_os_id로 계산한 결정적 값이다.
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
    """파싱 결과를 **PG를 거치지 않고** 곧장 색인한다 (#139).

    엔진이 서빙의 정본인 구성에서 PG 청크 INSERT는 서빙 경로가 한 번도 읽지 않는 행을 쓰는
    낭비다. 인제스션이 손에 든 청크·임베딩을 그대로 색인한다.

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
    """FAQ 청크를 PG 없이 색인. 항목당 청크 1개라 chunk_index는 0 고정이다."""
    await _delete_by_terms('faq_id', [faq_id])
    doc = build_doc(
        _ParsedChunk(cid=chunk_os_id(faq_id=faq_id), tenant_id=tenant_id,
                     document_id=None, faq_id=faq_id, text=text,
                     heading_path=[question], page=None, meta={},
                     dense=list(embedding.dense)),
        None, None, searchable=searchable)
    return await bulk_index([doc])


async def index_document_chunks(session, document_id: int) -> int:
    """문서의 청크를 색인에 반영(upsert).

    **eval·reconcile 전용** — PG에 청크 행이 있는 코퍼스(기존 A/B 코퍼스)를 다시 색인한다.
    인제스션은 이 함수를 쓰지 않는다: INDEX_DOCUMENT 핸들러(rag/documents.index_pending_document)가
    파싱 결과를 index_parsed_document로 직접 색인한다. PG에 청크는 있고 벡터만 없으면 본문에서
    재임베딩한다(_rows_to_docs). PG에 청크가 없으면 할 수 있는 게 없다 — 0을 돌려준다
    (reconcile은 그 구성에서 문서 단위로 재파싱 경로를 따로 탄다).
    """
    if not pg_stores_chunks():
        return 0
    return await bulk_index(await _index_rows(session, document_ids=[document_id]))


async def drop_documents_now(document_ids) -> int:
    return await _delete_by_terms('document_id', list(document_ids))


async def drop_faqs_now(faq_ids) -> int:
    return await _delete_by_terms('faq_id', list(faq_ids))


async def index_faq_chunks(session, faq_id: int) -> int:
    """FAQ 청크 재색인. 옛 문서를 먼저 지워야 색인에 유령이 남지 않는다.

    PG에 청크 행이 없는 구성이면 FAQ 행(질문·유사질문·답변)에서 텍스트를 다시 조립하고
    재임베딩한다 — FAQ는 파싱이 없어 원천이 그 행 자체다(문서 재파싱보다 훨씬 싸다).
    """
    from sqlalchemy import select

    from rag.models import Faq
    if not pg_stores_chunks():
        from rag.embeddings import embed_texts
        from rag.faq_indexing import build_faq_chunk_text
        faq = (await session.execute(select(Faq).where(Faq.id == faq_id))).scalars().first()
        if faq is None:
            return 0
        text = build_faq_chunk_text(faq.question, faq.variants or [], faq.answer)
        embs = await embed_texts([text])          # FAQ는 프리픽스 없이 원문 (인제스션과 동일)
        return await index_parsed_faq(
            faq_id=faq_id, tenant_id=faq.tenant_id, text=text, question=faq.question,
            searchable=effective_searchable(is_faq=True, faq_is_active=faq.is_active),
            embedding=embs[0])
    await _delete_by_terms('faq_id', [faq_id])
    return await bulk_index(await _index_rows(session, faq_ids=[faq_id]))


async def sync_meta_documents_now(session, document_ids) -> int:
    """문서 메타(검색가능·폴더) 부분 갱신. 문서 토글·폴더 이동, 그리고 폴더 자체의 변경
    (이름·설명·검색토글 → 호출부가 그 폴더의 document_ids를 넘긴다)에서 쓴다.

    같은 메타 값을 갖는 문서끼리 묶어 호출 수를 줄인다 — 폴더 변경이면 대개 1~2번이다.
    """
    from collections import defaultdict

    from sqlalchemy import select

    from rag.models import Document, Folder

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
    from collections import defaultdict

    from sqlalchemy import select

    from rag.models import Faq
    rows = (await session.execute(
        select(Faq.id, Faq.is_active).where(Faq.id.in_(list(faq_ids))))).all()
    groups = defaultdict(list)
    for fid, active in rows:
        groups[bool(active)].append(fid)
    n = 0
    for active, fids in groups.items():
        n += await _update_meta('faq_id', fids, {'searchable': active})
    return n


async def _reconcile_by_document(session) -> dict:
    """문서 단위 재동기화 — PG에 청크 행이 없는 구성용.

    청크 id를 대조할 수 없으므로 "PG에 있는 문서가 색인에도 있는가"만 본다. 청크 수준의
    드리프트(일부 청크 누락)는 못 잡는다 — 그건 outbox가 원자적으로 반영하는 것에 의존한다.
    """
    from sqlalchemy import select

    from rag.models import Document

    os_client = client()
    agg = await os_client.search(index=settings.opensearch_index, body={
        "size": 0,
        "aggs": {"d": {"terms": {"field": "document_id", "size": 65536}}}})
    os_docs = {int(b["key"]) for b in agg["aggregations"]["d"]["buckets"]}

    pg_docs = set((await session.execute(
        select(Document.id).where(Document.status == 'ready'))).scalars().all())

    missing, extra = pg_docs - os_docs, os_docs - pg_docs
    indexed = 0
    # 색인에 없는 ready 문서는 pending으로 되돌려 대기열에 넣는다 — 재파싱·재임베딩·ready 승격은
    # 인제스션 핸들러 한 곳(rag/documents.index_pending_document)이 맡는다. 두 벌을 두지 않는다.
    if missing:
        from sqlalchemy import update

        from rag import outbox
        from rag.models import Document as D
        await session.execute(update(D).where(D.id.in_(list(missing))).values(status='pending'))
        for did in sorted(missing):
            tenant = (await session.execute(select(D.tenant_id).where(D.id == did))).scalar()
            outbox.enqueue(session, tenant, outbox.INDEX_DOCUMENT, document_id=did)
        await session.commit()
        indexed = len(missing)          # 실제 색인은 다음 drain이 한다 — 여기선 재등재 건수
    deleted = await _delete_by_terms('document_id', sorted(extra)) if extra else 0
    return {'pg': len(pg_docs), 'os': len(os_docs), 'indexed': indexed, 'deleted': deleted,
            'unit': 'document'}


async def reconcile(session) -> dict:
    """3층 — PG↔OS id 집합 대조 후 복구. `eval/_os_index --repair`가 부른다.

    반환: {'pg': n, 'os': n, 'indexed': 누락분 색인 수, 'deleted': 잉여분 삭제 수}
    2층이 놓친 것(OpenSearch 장애 중 업로드·삭제)을 여기서 줍는다.
    """
    from sqlalchemy import select

    from rag.models import Chunk, Document

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

    # 비검색 청크도 색인 대상이다(플래그로 가른다) — 그래서 조건 없이 전 청크를 센다.
    #
    # **PG가 청크를 안 드는 구성에서는 이 대조가 성립하지 않는다** (#139) — 비교할 기준이
    # 없기 때문이다. 그때는 문서 단위로 본다: PG의 ready 문서 중 색인에 하나도 없는 것을
    # 찾아 재색인하고(원본 재파싱), PG에 없는 document_id의 청크를 지운다.
    if not pg_stores_chunks():
        return await _reconcile_by_document(session)

    pg_ids = set((await session.execute(select(Chunk.id))).scalars().all())

    missing, extra = pg_ids - os_ids, os_ids - pg_ids
    indexed = 0
    if missing:
        from rag.models import Chunk as C
        stmt = (select(C, Document.filename, Document.version)
                .outerjoin(Document, C.document_id == Document.id)
                .where(C.id.in_(list(missing))))
        # 벡터가 없으면 재임베딩한다(_rows_to_docs) — PG가 벡터를 안 드는 구성의 복구 경로.
        docs = await _rows_to_docs((await session.execute(stmt)).all())
        indexed = await bulk_index(docs)
    deleted = await _delete_by_terms('chunk_id', list(extra)) if extra else 0
    return {'pg': len(pg_ids), 'os': len(os_ids), 'indexed': indexed, 'deleted': deleted}
