"""OpenSearch 연결과 인덱스 매핑 — 엔진에 붙는 유일한 통로(`client`)와 청크 문서 형태(`MAPPING`)의 **정의점**.

청크(본문·메타·벡터·어휘 필드)는 엔진에만 있고 PG는 청크를 쓰지 않는다.
HNSW·BM25 파라미터는 매핑에 명시하지 않고 엔진 기본값을 쓴다.
단일 인덱스라 BM25의 df가 인덱스 전체 기준이다 — 다른 테넌트 데이터가 idf를 움직인다.
"""

import logging

from config import settings

DIM = 1024  # 임베딩 모델 차원 (rag/embeddings.py)

NORI_FIELD = "index_text_nori"
LEX_BIGRAM_FIELD = "lex_bigram"

MAPPING = {
    "settings": {
        "index": {
            "number_of_shards": 1,  # 샤드를 늘리면 df가 샤드별로 갈려 BM25 점수가 흔들린다.
            "number_of_replicas": 0,  # 배포는 단일 노드 — 레플리카 둘 곳이 없다.
            "knn": True,
            # ef_search·BM25 similarity는 명시하지 않는다 — 엔진 기본값(ef_search 100, BM25 k1=1.2 b=0.75).
            # lucene 엔진은 규모와 무관하게 HNSW 근사 탐색을 탄다(approximate_threshold는 faiss·nmslib 전용).
            # 테넌트 조각이 5천을 넘으면 근사 탐색 리콜이 떨어진다 — 그때 knn_clause에 method_parameters.ef_search 400.
        },
        "analysis": {
            "tokenizer": {
                # mixed — 복합어를 통째로도, 조각으로도 남긴다(고유명·상품코드 리콜).
                "nori_kms": {"type": "nori_tokenizer", "decompound_mode": "mixed"},
            },
            "analyzer": {
                # 내장 nori analyzer의 표준 구성을 그대로 쓴다(조사·어미·구두점 제거).
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
            # 테넌트 격리의 유일한 방어선 — 모든 질의가 tenant_filter()를 거친다.
            "tenant_id": {"type": "keyword"},
            "document_id": {"type": "long"},
            "faq_id": {"type": "long"},
            "page": {"type": "integer"},
            "version": {"type": "integer"},
            "is_table": {"type": "boolean"},
            "filename": {"type": "keyword"},
            "heading_path": {"type": "keyword"},
            # 식별자 채널 — 토큰은 rag/identifiers가 만들고 엔진은 정확 일치(terms)만 한다(분석기로 재현하지 않는다).
            #   identifiers: 제목 경로 + 묶음 헤딩을 정규화한 토큰('11조'·'43조의2'·'3장').
            #   headings:    묶음이 본문 줄로 내려보낸 헤딩 원문 — 근거 준비의 구간 경계(cut_identifier_span). 검색 대상 아님.
            "identifiers": {"type": "keyword"},
            "headings": {"type": "keyword", "index": False},
            # 엔진엔 조인이 없어 문서 메타를 청크마다 복사한다 — 메타가 바뀌면 sync_meta_*가 모든 청크를 갱신한다.
            # `searchable` = effective_searchable(). 엔진에서 필터하려고 여기 둔다 — PG에서 걸러내면 post-filter로
            # 후보가 조용히 깎인다. 비검색 청크도 색인해 토글이 재임베딩 없는 플래그 갱신이 되게 한다.
            "searchable": {"type": "boolean"},
            "folder_id": {"type": "long"},  # 폴더 단위 fan-out update의 필터 키
            "folder_name": {"type": "keyword"},
            # 폴더 설명은 리랭커 입력에만 들어간다(rag/index_text.py) — 검색 대상이 아니다.
            "folder_description": {"type": "text", "index": False},
            # 검색 대상은 아래 어휘 필드라 색인하지 않는다(어휘 점수 이중 관여 방지). 본문은 _source로 돌려받는다.
            "text": {"type": "text", "index": False},
            # 표 행 묶음 청크의 표 식별자와 표 전체 — 검색 대상 아님. prepare_evidence가 같은 표 묶음을 표 전체로 바꾼다.
            "table_key": {"type": "keyword", "index": False},
            "table_text": {"type": "text", "index": False},
            # 질의는 반드시 lex_clause()로 — 기본 match는 2.18 hybrid 쿼리를 500으로 죽인다.
            NORI_FIELD: {"type": "text", "analyzer": "nori_ko"},
            LEX_BIGRAM_FIELD: {
                # rag.lexical.bigrams() 출력을 공백으로 이은 문자열 — ngram 필터는 1글자 어절을 버려 재현이 안 된다.
                # 운영 어휘 채널은 Nori 필드다 — 이 필드는 실험·대조군용.
                "type": "text",
                "analyzer": "pretokenized",
            },
            "dense": {
                "type": "knn_vector",
                "dimension": DIM,
                "method": {
                    "name": "hnsw",
                    "engine": "lucene",  # cosinesimil 지원 + 필터를 kNN 탐색 단계에서 처리
                    "space_type": "cosinesimil",
                    # m·ef_construction 미명시 = 엔진 기본값
                },
            },
        },
    },
}

_client = None


def client():
    """공용 AsyncOpenSearch 싱글톤. 커넥션이 이벤트 루프에 묶이므로 프로세스당 루프 1개 전제다.

    `opensearchpy`는 함수 안에서 import한다 — 매핑 상수만 보는 테스트가 그 패키지 없이 import할 수 있게.
    """
    global _client
    if _client is None:
        from opensearchpy import AsyncOpenSearch

        _client = AsyncOpenSearch(
            [settings.opensearch_url],
            timeout=settings.opensearch_timeout,
            # 반드시 명시 — 생략하면 기본 10이 프로세스 전역 동시 요청 상한이 된다.
            maxsize=settings.opensearch_maxsize,
            retry_on_timeout=True,
            max_retries=2,
        )
    return _client


async def close_client() -> None:
    """클라이언트를 닫고 싱글톤을 비운다 — 루프가 바뀌면 옛 커넥션이 'Event loop is closed'로 죽기 때문이다."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None


# 기존 인덱스에 재색인 없이 덧붙여도 안전한 필드 — 검색하지 않는 보관 필드와 새 keyword 정확 일치 필드뿐이다.
# 기존 필드의 변경(분석기 등)은 여기로 못 한다(새 인덱스 + 전량 재색인). 값을 채우려면 문서를 재인제스트해야 한다.
ADDITIVE_FIELDS = ("identifiers", "headings", "table_key", "table_text")


async def _add_missing_fields(os_client) -> None:
    """ADDITIVE_FIELDS 중 없는 것만 덧붙인다. 실패해도 경고만 남긴다.

    drain이 매 회차 ensure_index를 부르므로 여기서 예외를 올리면 색인 전체가 멈춘다.
    """
    props = MAPPING["mappings"]["properties"]
    try:
        current = await os_client.indices.get_mapping(index=settings.opensearch_index)
        have = next(iter(current.values()), {}).get("mappings", {}).get("properties", {})
        missing = {f: props[f] for f in ADDITIVE_FIELDS if f not in have}
        if missing:
            await os_client.indices.put_mapping(index=settings.opensearch_index, body={"properties": missing})
    except Exception as e:  # noqa: BLE001 — 부가 작업 실패로 색인을 막지 않는다
        logging.getLogger(__name__).warning(
            '검색 저장소 매핑 필드 추가 실패 — 색인은 계속한다(%s): %s', ', '.join(ADDITIVE_FIELDS), e
        )


async def ensure_index() -> None:
    """인덱스가 없으면 MAPPING으로 만들고, 있으면 ADDITIVE_FIELDS만 덧붙인다(매핑 변경은 재색인 절차).

    엔진에 못 붙으면 예외를 올린다. 동시 생성 경합의 already-exists는 성공으로 본다.
    """
    os_client = client()
    if await os_client.indices.exists(settings.opensearch_index):
        await _add_missing_fields(os_client)
        return
    try:
        await os_client.indices.create(settings.opensearch_index, body=MAPPING)
    except Exception as e:  # noqa: BLE001 — 경합의 already-exists만 삼킨다
        if 'resource_already_exists_exception' not in str(e):
            raise


async def ensure_index_soft() -> bool:
    """기동용 ensure_index — 엔진에 못 붙어도 프로세스는 뜨게 ERROR 로그만 남기고 False를 돌려준다.

    그 상태의 검색·색인은 호출 시점에 실패하고, 색인은 엔진이 돌아오면 outbox 다음 회차가 반영한다.
    """
    try:
        await ensure_index()
        return True
    except Exception as e:  # noqa: BLE001 — 기동을 막지 않는 것이 목적
        logging.getLogger(__name__).error(
            '검색 저장소(OpenSearch) 연결 실패 — 검색·색인이 동작하지 않는 상태로 기동한다. url=%s: %s',
            settings.opensearch_url,
            e,
        )
        return False
