"""OpenSearch 연결·인덱스 스키마 — 검색 저장소의 진입로이자 매핑의 **정의점** (#139 도입 확정
2026-09-12, #146에서 구 rag/opensearch.py를 관심사별 4모듈로 분리).

이 파일이 드는 것은 둘뿐이다: 엔진에 붙는 유일한 통로(`client` 싱글톤)와 청크 문서가 어떤
형태로 사는지(`MAPPING`). 질의 조립·후보 회수는 `search` 쪽, 색인 쓰기는 `index` 쪽,
PG↔OS 재동기화는 `reconcile` 쪽이 맡는다.

청크(본문·메타·벡터·어휘 필드)는 엔진에만 있다. PG는 문서·FAQ·폴더·대화·캐시·outbox의
정본이고 청크 행을 쓰지 않는다.

톱레벨 import는 `config`뿐이다 — 0층 leaf.

## 파라미터 — OpenSearch 기본값

HNSW(m·ef_construction·ef_search)와 BM25(k1=1.2·b=0.75)는 **엔진 기본값을 그대로 쓴다** — 매핑에 명시하지
않는다(사용자 결정 2026-09-13: 표준/기본값으로). 도입 판정 A/B 때는 변인 격리를 위해 pgvector·앱 BM25와
맞춘 값(m=16/ef_construction=64/ef_search=40, k1=1.5)을 썼는데(이슈 #139·PR #140 본문), 도입이 끝난 뒤엔
그 값을 유지할 이유가 없다. 현 규모(1천 청크대)는 정확 탐색 구간이라 HNSW 파라미터는 리콜에 관여하지 않고,
k1 차이는 리랭커 뒤에서 흡수된다 — 재측정은 배포 후 전 축 eval에서.

거리 환산(`cosinesimil` 점수 → pgvector cosine_distance)은 그대로다 — 공식과 실측 근거는
`score_to_cosine_distance`의 docstring. bigram 토큰은 analyzer로 재현하지 않는다
(`LEX_BIGRAM_FIELD` 주석).

## 알려진 잔차 — 단일 인덱스·1샤드 구성의 귀결

1. **df 스코프**: 단일 인덱스라 Lucene의 df가 인덱스 전체다 — 다른 테넌트의 데이터가 우리
   idf를 움직인다(실측: 표시 지표는 미동). 테넌트별 인덱스가 대안.
2. **길이 노름**: Lucene은 필드 길이를 양자화해 저장한다.

(Nori 기본 사전의 고유명 오분할은 `search_lexical` docstring, 정확 탐색 구간은 아래 MAPPING의
`approximate_threshold` 주석이 정의점이다 — 여기서 반복하지 않는다.)
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
    매핑 상수만 보는 테스트가 `opensearchpy` 없이도 import할 수 있게.

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
