"""OpenSearch 질의 조립 계약 (#139, #146에서 모듈 분리에 맞춰 3파일로 나눔)
— DB·OpenSearch 없이 도는 순수 테스트.

  1. Nori 질의에 2.18 결함 회피 플래그가 붙어 있다    (조용히 사라지면 500이 돌아온다)
  2. 게이트 신호 환산이 실측값과 같다                 (틀리면 no_evidence 판정이 조용히 어긋난다)
  3. 필터가 엔진에서 걸린다                           (검색 후 걸러내면 post-filter가 되어 후보가 깎인다)
"""
from rag import os_client as C
from rag import os_search as S


def test_nori_질의에_2_18_결함_회피가_붙어_있다():
    """auto_generate_synonyms_phrase_query=False가 빠지면 hybrid+nori가 500으로 죽는다
    (실측 47/450 — rag/os_search.py의 lex_clause docstring). 조용히 사라지지 않게 묶는다."""
    clause = S.lex_clause(C.NORI_FIELD, "적립금 얼마")
    assert clause["match"][C.NORI_FIELD]["auto_generate_synonyms_phrase_query"] is False


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
        assert abs(S.score_to_cosine_distance(score) - pg_distance) < 2e-6, score


def test_필터는_엔진에서_걸린다():
    """테넌트·검색가능 둘 다 엔진 필터여야 한다.

    검색 후 PG로 걸러내면 상위 k를 뽑은 뒤 빼는 post-filter가 되어 후보가 조용히 깎인다
    (구 PG 경로가 겪던 함정). 실무 표준 구성에서는 필터에 쓰는 메타를 엔진에 비정규화해
    두는 것이 그 대가다.
    """
    terms = S.tenant_filter('t1')['bool']['filter']
    assert {'term': {'tenant_id': 't1'}} in terms
    assert {'term': {'searchable': True}} in terms
    # 매핑에도 필드가 있어야 실제로 걸린다
    props = C.MAPPING['mappings']['properties']
    assert props['searchable']['type'] == 'boolean'
    assert props['folder_id']['type'] == 'long'    # 폴더 단위 fan-out update의 필터 키
