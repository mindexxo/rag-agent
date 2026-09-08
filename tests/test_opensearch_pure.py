"""OpenSearch 백엔드 계약 (#139) — DB·OpenSearch 없이 도는 순수 테스트.

여기서 지키는 것은 **변인 격리의 전제**다. A/B의 결론은 "검색 계층만 다르다"는 전제 위에
서는데, 그 전제는 값 몇 개가 PG와 같아야 성립한다. 그 값들이 조용히 어긋나면 측정은
계속 돌면서 틀린 답을 낸다 — 그래서 코드로 묶는다.

  1. BM25 k1·b가 rag.lexical의 기본값과 같다        (Lucene 기본 k1=1.2 함정)
  2. dense 차원·HNSW 파라미터가 schema.sql과 같다   (pgvector 인덱스와 동일 조건)
  3. 어휘 입력 텍스트 조립이 운영과 같다             (문서=프리픽스 / FAQ=원문)
  4. Nori 질의에 2.18 결함 회피 플래그가 붙어 있다   (조용히 사라지면 500이 돌아온다)
  5. 게이트 신호 환산이 실측값과 같다                (틀리면 no_evidence 판정이 조용히 어긋난다)
  6. 검색 백엔드 기본값이 pg다                       (실험 경로가 실수로 운영이 되지 않게)
"""
import inspect
import re
from pathlib import Path

from rag import opensearch as B
from rag import lexical
from rag.index_text import build_index_text

SCHEMA = Path(__file__).resolve().parent.parent / "schema.sql"


def test_bm25_파라미터가_운영_정의점과_같다():
    """Lucene 기본 k1은 1.2다 — 맞춰두지 않으면 파라미터 차이를 구현 차이로 오독한다."""
    sig = inspect.signature(lexical.bm25_rank).parameters
    assert B.BM25_K1 == sig["k1"].default
    assert B.BM25_B == sig["b"].default
    sim = B.MAPPING["settings"]["index"]["similarity"]["bm25_kms"]
    assert (sim["type"], sim["k1"], sim["b"]) == ("BM25", B.BM25_K1, B.BM25_B)


def test_모든_어휘_필드가_그_similarity를_쓴다():
    """필드 하나가 similarity를 빼먹으면 그 채널만 Lucene 기본 k1로 채점된다."""
    props = B.MAPPING["mappings"]["properties"]
    for field in (B.NORI_FIELD, B.LEX_BIGRAM_FIELD):
        assert props[field]["similarity"] == "bm25_kms", field


def test_dense_필드가_pgvector와_같은_조건이다():
    """차원·m·ef_construction을 schema.sql에서 읽어 대조 — 한쪽만 바뀌는 드리프트를 막는다."""
    ddl = SCHEMA.read_text()
    dim = int(re.search(r"dense\s+VECTOR\((\d+)\)", ddl).group(1))
    m, efc = re.search(r"hnsw \(dense vector_cosine_ops\)\s*WITH \(m = (\d+), ef_construction = (\d+)\)",
                       ddl).groups()

    dense = B.MAPPING["mappings"]["properties"]["dense"]
    assert dense["type"] == "knn_vector"
    assert dense["dimension"] == dim == B.DIM
    method = dense["method"]
    assert method["name"] == "hnsw"
    # cosinesimil — pgvector가 vector_cosine_ops(코사인)이므로. lucene 엔진을 쓰는 이유는
    # cosinesimil 지원 + 필터를 kNN 탐색 단계에서 처리(rag/opensearch.py의 knn_clause 주석).
    assert method["space_type"] == "cosinesimil"
    assert method["engine"] == "lucene"
    assert method["parameters"] == {"m": int(m), "ef_construction": int(efc)}
    # ef_search는 schema.sql에 없다 — pgvector의 세션 기본값(40)이다. OpenSearch 기본은
    # 100이라 명시하지 않으면 탐색 폭이 넓은 쪽이 유리해진다(리뷰 지적).
    assert B.MAPPING["settings"]["index"]["knn.algo_param.ef_search"] == B.HNSW_EF_SEARCH == 40


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


def test_융합_가중치는_합이_1이어야_한다():
    """2.18 normalization-processor 계약. 어긋난 채 측정되면 융합 축이 조용히 무의미해진다.

    코루틴이라 asyncio.run으로 깨운다 — assert가 첫 await 앞이라 클라이언트는 안 쓰인다."""
    import asyncio

    import pytest
    with pytest.raises(AssertionError):
        asyncio.run(B.ensure_pipeline(None, weights=(0.7, 0.7)))


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


def test_검색_백엔드_기본값은_pg다():
    """실험 경로(#139)가 실수로 운영이 되지 않게. 운영 전환 선결 조건은 아직 없다 —
    색인 동기화·BM25 통계 스코프(config.py의 search_backend 주석)."""
    from config import Settings
    assert Settings(database_url="postgresql+asyncpg://x/y").search_backend == "pg"
