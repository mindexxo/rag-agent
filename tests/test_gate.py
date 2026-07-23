"""근거 게이트 단위 테스트 — retriever.apply_gate.

strict-grounded의 핵심 분기: 후보 없음 / 거리 초과 / 정상.
"""
from rag.retriever import RetrievalCandidates, RetrievedChunk, _keep_single_table, apply_gate


def _chunk(**kw) -> RetrievedChunk:
    base = dict(chunk_id=1, document_id=1, text='본문', heading_path=[], page=None,
                rrf_score=0.03, branches=['dense'], filename='환불정책.pdf', version=1)
    return RetrievedChunk(**{**base, **kw})


def test_후보_없음이면_no_results():
    assert apply_gate(RetrievalCandidates(chunks=[], top_dense_distance=999.0)) \
        == (True, 'no_results')


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


class TestKeepSingleTable:
    """표 청크는 RRF 최상위 1개만 유지 (F1a '한 시트만') — 순서 보존 필수."""

    def test_두번째_이후_표는_제외(self):
        chunks = [_chunk(chunk_id=1, is_table=True), _chunk(chunk_id=2),
                  _chunk(chunk_id=3, is_table=True), _chunk(chunk_id=4)]
        kept = _keep_single_table(chunks)
        assert [c.chunk_id for c in kept] == [1, 2, 4]        # 표는 1번만, 원래 순서 유지

    def test_표가_없으면_그대로(self):
        chunks = [_chunk(chunk_id=1), _chunk(chunk_id=2)]
        assert _keep_single_table(chunks) == chunks

    def test_표가_하위_순위에_있어도_첫_표만(self):
        chunks = [_chunk(chunk_id=1), _chunk(chunk_id=2, is_table=True),
                  _chunk(chunk_id=3, is_table=True)]
        assert [c.chunk_id for c in _keep_single_table(chunks)] == [1, 2]

    def test_빈_입력(self):
        assert _keep_single_table([]) == []
