"""OpenSearch 매핑·질의 조립 계약 (#139) — DB·OpenSearch 없이 도는 순수 테스트.

매핑은 색인 생성 시점에 굳고(ensure_index) 바꾸면 재색인이다. 여기 묶는 값들이 조용히 어긋나면
검색은 계속 돌면서 틀린 답을 낸다 — 그래서 코드로 고정한다.

  1. BM25·HNSW 파라미터를 매핑에 명시하지 않는다      (엔진 기본값 — 사용자 결정 2026-09-13)
  2. dense 차원·엔진·space_type이 고정돼 있다          (임베딩 1024·lucene·cosinesimil)
  3. 어휘 입력 텍스트 조립이 임베딩 입력과 같다       (문서=프리픽스 / FAQ=원문)
  4. Nori 질의에 2.18 결함 회피 플래그가 붙어 있다    (조용히 사라지면 500이 돌아온다)
  5. 게이트 신호 환산이 실측값과 같다                 (틀리면 no_evidence 판정이 조용히 어긋난다)
  6. 필터가 엔진에서 걸린다                           (검색 후 걸러내면 post-filter가 되어 후보가 깎인다)
  7. searchable 판정이 조건표대로다                   (문서 활성·ready·토글·폴더 / FAQ 활성)
  8. build_doc는 청크에 실린 벡터로 색인하고, 없으면 크게 실패한다
  9. chunk_os_id는 결정적이고 문서·청크 간 충돌이 없으며 20비트 상한을 강제한다
"""
import re

import pytest

from rag import opensearch as B
from rag import lexical
from rag.index_text import build_index_text



def test_bm25_hnsw_파라미터는_엔진_기본값을_쓴다():
    """k1·b·m·ef_construction·ef_search를 매핑에 박지 않는다 — A/B 때 pgvector와 맞춘 값(k1 1.5, ef 64/40)은
    도입 판정용이었고, 도입 후엔 표준 기본값으로 간다(사용자 결정). 누가 다시 박으면 여기서 걸린다."""
    settings_ = B.MAPPING["settings"]["index"]
    assert "similarity" not in settings_
    assert not any(k.startswith("knn.algo_param") for k in settings_)
    props = B.MAPPING["mappings"]["properties"]
    for field in (B.NORI_FIELD, B.LEX_BIGRAM_FIELD):
        assert props[field]["type"] == "text" and "similarity" not in props[field], field
    assert "parameters" not in props["dense"]["method"]


def test_dense_필드는_임베딩_차원과_lucene_cosinesimil로_고정된다():
    """차원이 어긋나면 색인이 통째로 실패하고, engine·space_type이 바뀌면 거리 환산(score_to_cosine_distance)이
    틀어져 게이트 신호가 조용히 어긋난다. HNSW 파라미터는 기본값(별도 테스트)."""
    dense = B.MAPPING["mappings"]["properties"]["dense"]
    assert dense["type"] == "knn_vector" and dense["dimension"] == B.DIM == 1024
    assert dense["method"]["name"] == "hnsw"
    assert dense["method"]["engine"] == "lucene"           # cosinesimil 지원 + 필터를 kNN 탐색 단계에서 처리
    assert dense["method"]["space_type"] == "cosinesimil"


def test_어휘_입력_조립이_운영과_같다():
    """문서는 '파일명>헤딩' 프리픽스, FAQ는 원문 — 인제스션·운영 어휘 채널과 동일 비대칭."""
    text, fn, hp = "반품은 7일 이내", "환불반품정책.pdf", ["2. 반품", "2.1 기한"]
    assert B.lex_text(text, fn, hp) == build_index_text(text, fn, hp)
    assert B.lex_text(text, None, None) == text          # FAQ — 프리픽스 없음
    # 폴더 설명은 들어가지 않는다 (리랭커 전용, rag/index_text.py docstring)
    assert "폴더" not in B.lex_text(text, fn, hp)


def test_bigram_필드는_공백만_자른다():
    """analyzer로 bigram을 재현하지 않는다는 설계가 매핑에 남아 있는지 — 이게 바뀌면
    1글자 어절 보존이 깨져 bigram 대조군이 대조군이 아니게 된다(rag/lexical.py:24-38)."""
    props = B.MAPPING["mappings"]["properties"]
    analyzer = props[B.LEX_BIGRAM_FIELD]["analyzer"]
    assert B.MAPPING["settings"]["analysis"]["analyzer"][analyzer]["tokenizer"] == "whitespace"
    # 1글자 어절이 살아남는 것이 이 설계의 요점 — 정의점 함수로 확인
    assert lexical.bigrams("가 나다") == ["가", "나다"]


def test_nori_질의에_2_18_결함_회피가_붙어_있다():
    """auto_generate_synonyms_phrase_query=False가 빠지면 hybrid+nori가 500으로 죽는다
    (실측 47/450 — rag/opensearch.py의 lex_clause docstring). 조용히 사라지지 않게 묶는다."""
    clause = B.lex_clause(B.NORI_FIELD, "적립금 얼마")
    assert clause["match"][B.NORI_FIELD]["auto_generate_synonyms_phrase_query"] is False


def test_게이트_신호_환산이_실측과_같다():
    """OpenSearch cosinesimil 점수 → pgvector cosine_distance = 2 - 2*score.

    실측 대조표(2026-09-08, adererror 5건). 이 환산이 틀리면 apply_gate의 no_evidence
    판정이 조용히 어긋난다 — 운영은 게이트가 꺼져 있어(GATE_DISABLED) eval sweep에
    국한되지만, 그 sweep이 임계값을 정하는 근거다.

    허용오차 2e-6: 표의 두 값이 각각 소수점 6자리로 반올림된 기록이라 각 5e-7까지 어긋날 수
    있고, distance는 score에 2를 곱하므로 그만큼 증폭된다. 실제 환산은 정확하다."""
    for score, pg_distance in ((0.862583, 0.274834), (0.849968, 0.300063),
                               (0.842895, 0.314211), (0.842461, 0.315078),
                               (0.830557, 0.338887)):
        assert abs(B.score_to_cosine_distance(score) - pg_distance) < 2e-6, score


def test_필터는_엔진에서_걸린다():
    """테넌트·검색가능 둘 다 엔진 필터여야 한다.

    검색 후 PG로 걸러내면 상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다
    (구 PG 경로가 겪던 함정). 실무 표준 구성에서는 필터에 쓰는 메타를 엔진에 비정규화해
    두는 것이 그 대가다.
    """
    terms = B.tenant_filter('t1')['bool']['filter']
    assert {'term': {'tenant_id': 't1'}} in terms
    assert {'term': {'searchable': True}} in terms
    # 매핑에도 필드가 있어야 실제로 걸린다
    props = B.MAPPING['mappings']['properties']
    assert props['searchable']['type'] == 'boolean'
    assert props['folder_id']['type'] == 'long'    # 폴더 단위 fan-out update의 필터 키


def test_searchable_판정이_조건표대로다():
    """검색 가능 판정의 정의점(effective_searchable) — 문서: 활성+ready+검색토글+(미분류 or 폴더 on),
    FAQ: 활성. 색인 시 계산해 넣는 플래그라 조건이 바뀌면 전량 META 갱신이 따라야 한다."""
    ok = dict(is_faq=False, doc_is_active=True, doc_status='ready', doc_is_searchable=True)
    assert B.effective_searchable(**ok, folder_is_searchable=None) is True   # 미분류 문서
    assert B.effective_searchable(**ok, folder_is_searchable=True) is True
    assert B.effective_searchable(**ok, folder_is_searchable=False) is False  # 폴더 off
    assert B.effective_searchable(**{**ok, 'doc_is_active': False}) is False
    assert B.effective_searchable(**{**ok, 'doc_status': 'deleted'}) is False
    assert B.effective_searchable(**{**ok, 'doc_is_searchable': False}) is False
    assert B.effective_searchable(is_faq=True, faq_is_active=True) is True
    assert B.effective_searchable(is_faq=True, faq_is_active=False) is False


def test_엔진이_돌려준_청크에_인용_메타가_다_실린다():
    """PG를 되묻지 않으므로 RetrievedChunk의 모든 필드가 _source에 있어야 한다.
    하나라도 빠지면 인용 표시나 리랭커 입력이 조용히 비어버린다."""
    props = B.MAPPING['mappings']['properties']
    for f in ('chunk_id', 'document_id', 'faq_id', 'text', 'heading_path', 'page',
              'filename', 'version', 'is_table', 'folder_name', 'folder_description'):
        assert f in props, f


class _Chunk:
    """색인 대상 청크의 최소 형태(_ParsedChunk 모양). dense 속성이 없는 상태도 만들 수 있다."""

    def __init__(self, cid=1, text='반품은 7일 이내', dense=None, has_dense=True):
        self.id, self.text = cid, text
        self.tenant_id, self.document_id, self.faq_id = 't1', 10, None
        self.page, self.heading_path, self.meta = 3, ['2. 반품'], {}
        if has_dense:
            self.dense = dense


def test_벡터는_청크에_실려_온_값으로_색인된다():
    doc = B.build_doc(_Chunk(dense=[0.5] * 1024), '정책.pdf', 2)
    assert len(doc['dense']) == 1024 and doc['dense'][0] == pytest.approx(0.5)
    assert doc['chunk_id'] == 1 and doc['filename'] == '정책.pdf' and doc['version'] == 2


def test_벡터가_없으면_크게_실패한다():
    """조용히 벡터 없는 문서를 색인하면 kNN에 안 잡히는 유령이 된다."""
    with pytest.raises(ValueError, match='벡터가 없다'):
        B.build_doc(_Chunk(has_dense=False), '정책.pdf', 2)
    with pytest.raises(ValueError, match='벡터가 없다'):
        B.build_doc(_Chunk(dense=None), '정책.pdf', 2)


def test_어휘_필드는_본문에서_만든다():
    """Nori·bigram 필드는 lex_text(본문+프리픽스)에서 색인 시점에 조립된다."""
    ch = _Chunk(dense=[0.5] * 1024)
    doc = B.build_doc(ch, '정책.pdf', 2)
    expected = B.lex_text(ch.text, '정책.pdf', ch.heading_path)
    assert doc[B.NORI_FIELD] == expected
    assert doc[B.LEX_BIGRAM_FIELD] == ' '.join(lexical.bigrams(expected))


def test_chunk_os_id는_결정적이고_충돌이_없다():
    """PG 시퀀스 없이 (부모 id, chunk_index)로 만드는 id — 재색인이 같은 _id로 덮여 멱등하려면
    결정적이어야 하고, 문서·청크 조합마다 달라야 한다."""
    f = B.chunk_os_id
    assert f(document_id=1008, chunk_index=3) == f(document_id=1008, chunk_index=3)
    ids = {f(document_id=d, chunk_index=i) for d in (1, 2, 1008, 99999) for i in (0, 1, 7, 500)}
    assert len(ids) == 16
    # 문서 id 1의 첫 청크조차 2^20 이상 — 소규모 PG BIGSERIAL 청크 id와 겹치지 않는다
    assert f(document_id=1, chunk_index=0) == 1 << 20
    # FAQ는 음수 네임스페이스 — 문서 id 공간과 절대 겹치지 않는다(캐시의 -faq_id 관례와 같은 방식)
    assert f(faq_id=42) < 0 and f(faq_id=42) == -(42 << 20)
    assert {f(faq_id=q) for q in (1, 2, 42)}.isdisjoint(ids)


def test_chunk_os_id는_20비트_상한을_강제한다():
    """상한을 넘으면 상위 비트(문서 id 몫)를 침범해 다른 문서의 청크와 같은 id가 된다 —
    조용히 덮어쓰지 않고 크게 실패해야 한다."""
    assert B.chunk_os_id(document_id=5, chunk_index=(1 << 20) - 1) > 0
    with pytest.raises(ValueError, match='20비트'):
        B.chunk_os_id(document_id=5, chunk_index=1 << 20)


@pytest.mark.asyncio
async def test_엔진에_못_붙으면_ensure_index가_예외를_올린다(monkeypatch):
    """OpenSearch 없이 기동하면 죽어야 한다(E2E #10) — 검색 없는 서버가 조용히 뜨는 것을 막는 계약.
    main.lifespan·worker.startup이 이 함수를 첫 줄에서 부른다."""
    class _Indices:
        async def exists(self, *a, **k):
            raise ConnectionError('Cannot connect to host localhost:9200')

    class _Client:
        indices = _Indices()

    monkeypatch.setattr(B, 'client', lambda: _Client())
    with pytest.raises(ConnectionError):
        await B.ensure_index()


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

    monkeypatch.setattr(B, 'client', lambda: _Client(True))
    await B.ensure_index()
    assert calls == []                                   # 있으면 create를 부르지 않는다
    monkeypatch.setattr(B, 'client', lambda: _Client(False))
    await B.ensure_index()                               # 웹·워커 동시 생성 경합 — 예외 아님
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

    monkeypatch.setattr(B, 'client', lambda: _Client())
    import logging
    with caplog.at_level(logging.ERROR, logger='rag.opensearch'):
        assert await B.ensure_index_soft() is False
    assert any('연결 실패' in r.getMessage() for r in caplog.records)

