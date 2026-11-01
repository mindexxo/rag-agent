"""근거 게이트 단위 테스트 — retriever.apply_gate.

strict-grounded의 핵심 분기: 후보 없음 / 거리 초과 / 정상.
"""

from rag.search.retrieval_types import RetrievalCandidates, RetrievedChunk
from rag.search.retriever import apply_gate


def _chunk(**kw) -> RetrievedChunk:
    base = dict(chunk_id=1, document_id=1, text='본문', heading_path=[], page=None, filename='환불정책.pdf', version=1)
    return RetrievedChunk(**{**base, **kw})


def test_후보_없음이면_no_results():
    assert apply_gate(RetrievalCandidates(chunks=[], top_dense_distance=999.0)) == (True, 'no_results')


def test_거리_임계값_초과면_low_similarity():
    cands = RetrievalCandidates(chunks=[_chunk()], top_dense_distance=0.61)
    assert apply_gate(cands) == (True, 'low_similarity')


def test_거리_정상이면_통과():
    cands = RetrievalCandidates(chunks=[_chunk()], top_dense_distance=0.35)
    assert apply_gate(cands) == (False, None)


def test_경계값_임계값과_같으면_통과():
    # 판정은 '초과(>)' — 정확히 0.6이면 근거 인정 (현재 동작 고정)
    cands = RetrievalCandidates(chunks=[_chunk()], top_dense_distance=0.6)
    assert apply_gate(cands) == (False, None)


def test_임계값_인자로_조정_가능():
    cands = RetrievalCandidates(chunks=[_chunk()], top_dense_distance=0.5)
    assert apply_gate(cands, max_dense_distance=0.4) == (True, 'low_similarity')


def test_거리게이트를_꺼도_빈_후보는_no_results():
    # threshold sweep(inf로 거리 게이트 해제) 시에도 빈 결과 판정은 유지돼야 함 — 두 조건 병합 리팩토링 방지
    cands = RetrievalCandidates(chunks=[], top_dense_distance=999.0)
    assert apply_gate(cands, max_dense_distance=float('inf')) == (True, 'no_results')
