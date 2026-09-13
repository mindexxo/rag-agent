"""OpenSearch 매핑·인덱스 생성 계약 (#139, #146에서 모듈 분리에 맞춰 3파일로 나눔)
— DB·OpenSearch 없이 도는 순수 테스트.

매핑은 색인 생성 시점에 굳고(ensure_index) 바꾸면 재색인이다. 여기 묶는 값들이 조용히 어긋나면
검색은 계속 돌면서 틀린 답을 낸다 — 그래서 코드로 고정한다.

  1. BM25·HNSW 파라미터를 매핑에 명시하지 않는다      (엔진 기본값 — 사용자 결정 2026-09-13)
  2. dense 차원·엔진·space_type이 고정돼 있다          (임베딩 1024·lucene·cosinesimil)
  3. bigram 필드는 공백만 자른다                       (토큰은 rag/lexical.py가 만든다)
  4. 엔진이 돌려준 청크에 인용 메타가 다 실린다        (PG를 되묻지 않으므로)
  5. ensure_index는 못 붙으면 예외, 경합의 already_exists는 성공, soft는 예외 없이 False
"""
import pytest

from rag import os_client as C
from rag import lexical


def test_bm25_hnsw_파라미터는_엔진_기본값을_쓴다():
    """k1·b·m·ef_construction·ef_search를 매핑에 박지 않는다 — A/B 때 pgvector와 맞춘 값(k1 1.5, ef 64/40)은
    도입 판정용이었고, 도입 후엔 표준 기본값으로 간다(사용자 결정). 누가 다시 박으면 여기서 걸린다."""
    settings_ = C.MAPPING["settings"]["index"]
    assert "similarity" not in settings_
    assert not any(k.startswith("knn.algo_param") for k in settings_)
    props = C.MAPPING["mappings"]["properties"]
    for field in (C.NORI_FIELD, C.LEX_BIGRAM_FIELD):
        assert props[field]["type"] == "text" and "similarity" not in props[field], field
    assert "parameters" not in props["dense"]["method"]


def test_dense_필드는_임베딩_차원과_lucene_cosinesimil로_고정된다():
    """차원이 어긋나면 색인이 통째로 실패하고, engine·space_type이 바뀌면 거리 환산(score_to_cosine_distance)이
    틀어져 게이트 신호가 조용히 어긋난다. HNSW 파라미터는 기본값(별도 테스트)."""
    dense = C.MAPPING["mappings"]["properties"]["dense"]
    assert dense["type"] == "knn_vector" and dense["dimension"] == C.DIM == 1024
    assert dense["method"]["name"] == "hnsw"
    assert dense["method"]["engine"] == "lucene"           # cosinesimil 지원 + 필터를 kNN 탐색 단계에서 처리
    assert dense["method"]["space_type"] == "cosinesimil"


def test_bigram_필드는_공백만_자른다():
    """analyzer로 bigram을 재현하지 않는다는 설계가 매핑에 남아 있는지 — 이게 바뀌면
    1글자 어절 보존이 깨져 bigram 대조군이 대조군이 아니게 된다(rag/lexical.py:24-38)."""
    props = C.MAPPING["mappings"]["properties"]
    analyzer = props[C.LEX_BIGRAM_FIELD]["analyzer"]
    assert C.MAPPING["settings"]["analysis"]["analyzer"][analyzer]["tokenizer"] == "whitespace"
    # 1글자 어절이 살아남는 것이 이 설계의 요점 — 정의점 함수로 확인
    assert lexical.bigrams("가 나다") == ["가", "나다"]


def test_엔진이_돌려준_청크에_인용_메타가_다_실린다():
    """PG를 되묻지 않으므로 RetrievedChunk의 모든 필드가 _source에 있어야 한다.
    하나라도 빠지면 인용 표시나 리랭커 입력이 조용히 비어버린다."""
    props = C.MAPPING['mappings']['properties']
    for f in ('chunk_id', 'document_id', 'faq_id', 'text', 'heading_path', 'page',
              'filename', 'version', 'is_table', 'folder_name', 'folder_description'):
        assert f in props, f


@pytest.mark.asyncio
async def test_엔진에_못_붙으면_ensure_index가_예외를_올린다(monkeypatch):
    """OpenSearch 없이 기동하면 죽어야 한다(E2E #10) — 검색 없는 서버가 조용히 뜨는 것을 막는 계약.
    main.lifespan·worker.startup이 이 함수를 첫 줄에서 부른다."""
    class _Indices:
        async def exists(self, *a, **k):
            raise ConnectionError('Cannot connect to host localhost:9200')

    class _Client:
        indices = _Indices()

    monkeypatch.setattr(C, 'client', lambda: _Client())
    with pytest.raises(ConnectionError):
        await C.ensure_index()


@pytest.mark.asyncio
async def test_이미_있는_인덱스는_건드리지_않고_경합의_already_exists는_성공으로_본다(monkeypatch):
    calls = []

    class _Indices:
        def __init__(self, exists):
            self._exists = exists

        async def exists(self, *a, **k):
            return self._exists

        async def create(self, *a, **k):
            calls.append('create')
            raise RuntimeError('resource_already_exists_exception: index already exists')

    class _Client:
        def __init__(self, exists):
            self.indices = _Indices(exists)

    monkeypatch.setattr(C, 'client', lambda: _Client(True))
    await C.ensure_index()
    assert calls == []                                   # 있으면 create를 부르지 않는다
    monkeypatch.setattr(C, 'client', lambda: _Client(False))
    await C.ensure_index()                               # 웹·워커 동시 생성 경합 — 예외 아님
    assert calls == ['create']


@pytest.mark.asyncio
async def test_ensure_index_soft는_못_붙어도_예외_없이_False(monkeypatch, caplog):
    """기동 경로(main lifespan·worker startup·drain)는 이걸 쓴다 — 로컬 개발에서 OpenSearch 없이도 앱이 뜨게
    (사용자 결정 2026-09-13). 실패는 삼키지 않고 ERROR 로그로 남긴다."""
    class _Indices:
        async def exists(self, *a, **k):
            raise ConnectionError('Cannot connect to host 10.1.32.20:23338')

    class _Client:
        indices = _Indices()

    monkeypatch.setattr(C, 'client', lambda: _Client())
    import logging
    with caplog.at_level(logging.ERROR, logger='rag.os_client'):
        assert await C.ensure_index_soft() is False
    assert any('연결 실패' in r.getMessage() for r in caplog.records)
