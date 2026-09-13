"""OpenSearch — 검색 저장소·서빙 정본 (#139 도입 확정 2026-09-12). 매핑·질의 조립·색인 쓰기의 **정의점**.

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
  낡은 텍스트·메타가 인용된다. 그 대가를 쓰기 경로가 갚는다(아래 "색인 쓰기 경로 — 정합 관리").

## 파라미터 — OpenSearch 기본값

HNSW(m·ef_construction·ef_search)와 BM25(k1=1.2·b=0.75)는 **엔진 기본값을 그대로 쓴다** — 매핑에 명시하지
않는다(사용자 결정 2026-09-13: 표준/기본값으로). 도입 판정 A/B 때는 변인 격리를 위해 pgvector·앱 BM25와
맞춘 값(m=16/ef_construction=64/ef_search=40, k1=1.5)을 썼는데(이슈 #139·PR #140 본문), 도입이 끝난 뒤엔
그 값을 유지할 이유가 없다. 현 규모(1천 청크대)는 정확 탐색 구간이라 HNSW 파라미터는 리콜에 관여하지 않고,
k1 차이는 리랭커 뒤에서 흡수된다 — 재측정은 배포 후 전 축 eval에서.

거리 환산 `cosine_distance = 2 - 2*score`(`cosinesimil` 점수가 (1+cos)/2 — pgvector와 소수점 6자리까지 일치
실측)는 그대로다. bigram 토큰은 analyzer로 재현하지 않는다(`LEX_BIGRAM_FIELD` 주석).

## 알려진 잔차

1. **df 스코프**: 단일 인덱스라 Lucene의 df가 인덱스 전체다 — 다른 테넌트의 데이터가 우리
   idf를 움직인다(실측: 표시 지표는 미동). 테넌트별 인덱스가 대안.
2. **길이 노름**: Lucene은 필드 길이를 양자화해 저장한다.
3. **Nori 기본 사전의 고유명 오분할**('아더에러'→['에러']) — 사용자 사전 운영 절차가 후속 이슈.
4. **정확 탐색 구간**: `approximate_threshold`(15000) 미만이면 HNSW를 건너뛰고 정확 탐색한다 —
   지금 규모(1천 청크대)는 ANN 근사가 아직 없다.
"""
import logging

from config import settings

DIM = 1024              # 임베딩 모델 차원 (rag/embeddings.py)
# HNSW·BM25 파라미터는 엔진 기본값 — 모듈 docstring "파라미터 — OpenSearch 기본값".

NORI_FIELD = "index_text_nori"
LEX_BIGRAM_FIELD = "lex_bigram"

MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,      # 샤드를 늘리면 df가 샤드별로 갈려 BM25 점수가 흔들린다.
            "number_of_replicas": 0,    # 배포는 자체 설치 단일 노드(#139 결정) — 레플리카 둘 곳이 없다.
            "knn": True,
            # ef_search·BM25 similarity는 명시하지 않는다 — 엔진 기본값(ef_search 100, BM25 k1=1.2 b=0.75).
            # **현 규모에서는 ef_search가 쓰이지도 않는다**: index.knn.advanced.approximate_threshold
            # (기본 15000) 미만이면 HNSW를 건너뛰고 정확 탐색한다. 청크가 1만5천을 넘으면
            # ef_search가 리콜을 좌우하기 시작한다 — 그때 재측정.
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
            # chunk_os_id() — (부모 id, chunk_index)로 계산한 결정적 id. 색인·삭제의 멱등 키(_id)이자
            # eval 채점의 매칭 키다.
            "chunk_id": {"type": "long"},
            # 테넌트 격리 — PG 쪽이 RLS 없이 WHERE절이 유일한 방어선인 것처럼(rag/models.py:14-26),
            # 여기서는 filter가 그 방어선이다. 모든 질의가 tenant_filter()를 거친다.
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
            # `searchable` = effective_searchable()의 결과(문서 활성·ready·검색토글·폴더 토글 /
            # FAQ 활성). **필터를 엔진에서 걸기 위해** 반드시 여기 있어야 한다 — 검색 후 PG로
            # 걸러내면 상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다.
            # 비검색 청크도 색인하고 이 플래그로 가른다 — 토글 on/off가 문서 추가·삭제가 아니라
            # 플래그 부분 갱신이 되어 fan-out이 가볍고 재임베딩이 없다.
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
            NORI_FIELD: {"type": "text", "analyzer": "nori_ko"},
            LEX_BIGRAM_FIELD: {
                # rag.lexical.bigrams()의 출력을 공백으로 이어 붙인 문자열이 들어온다.
                # analyzer로 재현하지 않는 이유: bigrams()는 어절 내 문자 bigram이면서
                # **1글자 어절을 그대로 보존**하는데(rag/lexical.py:24-38), OpenSearch의 ngram
                # 토큰 필터는 min_gram 미달 토큰을 버려 그 동작이 재현되지 않는다. 흉내내면
                # 토큰이 어긋나므로, 정의점 함수를 직접 호출해 토큰을 만들고 여기서는 공백으로만
                # 자른다. 운영 어휘 채널은 Nori 필드를 쓴다 — 이 필드는 실험·대조군용이다.
                "type": "text",
                "analyzer": "pretokenized",
            },
            "dense": {
                "type": "knn_vector",
                "dimension": DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",       # cosinesimil 지원 + 필터를 kNN 탐색 단계에서 처리
                    "space_type": "cosinesimil",
                    # m·ef_construction 미명시 = 엔진 기본값 (모듈 docstring)
                },
            },
        },
    },
}

_client = None


def client():
    """공용 AsyncOpenSearch 싱글톤.

    `opensearchpy`는 함수 안에서 import한다 — 이 모듈은 0층 leaf라 톱레벨엔 config만 두고,
    순수 함수(chunk_os_id·build_doc 등)를 쓰는 테스트가 클라이언트 없이도 import할 수 있게.

    커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제다(rag/clients.py와 동일 제약).
    """
    global _client
    if _client is None:
        from opensearchpy import AsyncOpenSearch
        _client = AsyncOpenSearch([settings.opensearch_url],
                                  timeout=settings.opensearch_timeout,
                                  retry_on_timeout=True, max_retries=2)
    return _client


async def close_client() -> None:
    """공용 클라이언트를 닫고 싱글톤을 비운다 — 종료 훅(main lifespan·worker on_shutdown)과,
    테스트마다 루프를 갈아끼우는 conftest._loop_hygiene가 부른다(rag/clients.py의 http_async
    재생성과 같은 이유: aiohttp 커넥션이 닫힌 루프에 묶여 'Event loop is closed'로 죽는다)."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


def lex_text(text: str, filename: str | None, heading_path) -> str:
    """어휘·Nori 필드의 입력 텍스트 — 인제스션·운영 어휘 채널과 **동일 조립**.

    문서는 '파일명>헤딩' 프리픽스(build_index_text), FAQ는 원문 그대로다. 이 비대칭은
    임베딩 입력과 같다(rag/index_text.py docstring) — 임베딩·Nori·bigram 세 필드가 같은
    텍스트를 본다. 폴더 설명은 넣지 않는다 — 리랭커 전용이다(임베딩·어휘엔 미포함).
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


# ===== 색인 쓰기 경로 — 정합 관리 =========================================
#
# 청크가 PG 트랜잭션 밖(엔진)에 있으므로 "PG 변경 ↔ 색인 반영"의 정합은 따로 관리해야 한다.
# 그 방법이 이 절이고, 정의점은 rag/outbox.py다.
#
# ## 전제 — 색인이 서빙의 정본이다
#
# 실무 표준 구성: 검색 결과의 텍스트·메타를 엔진이 돌려준다(`fetch_chunk_map`). PG로 되묻어
# 검증하지 않는다. 그래서 "색인이 낡으면 낡은 답이 나간다"가 구조적으로 가능하고, 그 자리를
# 아래가 메운다.
#
# ## 1) 필터를 엔진에 비정규화
#
# `searchable`(effective_searchable)·`tenant_id`를 청크 문서마다 넣고 엔진에서 필터한다.
# 검색 후 PG로 걸러내면 상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다.
# 비검색 청크도 색인한다 — 토글 on/off가 문서 추가·삭제가 아니라 플래그 부분 갱신
# (`_update_meta`)이 되고 재임베딩이 없다.
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
# `python -m eval.os_reconcile`이 PG↔OS를 **문서 단위**로 대조해 복구한다(`reconcile`) — PG에
# 청크가 없으므로 청크 일부 누락은 못 잡고 outbox의 원자성(문서 단위 색인 후 한 커밋)에 의존한다.
#
# 아래 원시 연산들은 **실패를 삼키지 않는다** — 예외를 올려 outbox가 횟수를 세고 MAX_ATTEMPTS에
# failed로 확정하게 한다. 전부 멱등이다: 색인은 _id=chunk_id upsert(결정적 id), 삭제는 없는 것을
# 지워도 무해, 메타 갱신은 같은 값을 덮어쓸 뿐이라 재시도가 겹쳐도 결과가 같다.


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
    from rag.lexical import bigrams
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
    from sqlalchemy import select

    from rag.embeddings import embed_texts
    from rag.faq_indexing import build_faq_chunk_text
    from rag.models import Faq
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


async def ensure_index() -> None:
    """인덱스가 없으면 MAPPING으로 만든다 — 테스트(conftest)와 ensure_index_soft가 부른다.

    있으면 건드리지 않는다: 매핑 변경은 재색인이 따르는 별도 절차다(새 인덱스 + 전량 재색인).
    엔진에 못 붙으면 예외를 올린다(엄격 판). 기동 경로는 ensure_index_soft를 쓴다.
    두 프로세스(웹·워커)가 동시에 만들면 한쪽이 already-exists를 받는다 — 그건 성공으로 본다.
    """
    os_client = client()
    if await os_client.indices.exists(settings.opensearch_index):
        return
    try:
        await os_client.indices.create(settings.opensearch_index, body=MAPPING)
    except Exception as e:                       # noqa: BLE001 — 경합의 already-exists만 삼킨다
        if 'resource_already_exists_exception' not in str(e):
            raise


async def ensure_index_soft() -> bool:
    """기동용 — 엔진에 못 붙어도 **프로세스는 뜬다**(사용자 결정 2026-09-13: 로컬 개발 시 OpenSearch가
    없거나 개발계에 못 붙어도 앱은 기동돼야 한다). 실패는 ERROR 로그로 남기고 False를 돌려준다.

    그 상태에서 검색(`/kms/query`)·색인(outbox drain)은 호출 시점에 ConnectionError로 실패한다 —
    검색은 요청 단위 오류, 색인은 outbox attempts에 쌓여 엔진이 돌아오면 다음 회차가 반영한다(멱등).
    인덱스가 없는 채로 색인 요청이 먼저 오는 경우는 없다: bulk 전에 이 함수가 성공한 적이 없다면
    엔진 자체에 못 붙는 상태이고, 붙는 순간 다음 기동 또는 drain 회차의 ensure_index가 만든다.
    """
    try:
        await ensure_index()
        return True
    except Exception as e:                       # noqa: BLE001 — 기동을 막지 않는 것이 목적
        logging.getLogger(__name__).error(
            '검색 저장소(OpenSearch) 연결 실패 — 검색·색인이 동작하지 않는 상태로 기동한다. url=%s: %s',
            settings.opensearch_url, e)
        return False
