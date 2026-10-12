"""근거 게이트 단위 테스트 — retriever.apply_gate.

strict-grounded의 핵심 분기: 후보 없음 / 거리 초과 / 정상.
"""
from rag.retriever import _keep_single_table, apply_gate
from rag.retrieval_types import RetrievalCandidates, RetrievedChunk


def _chunk(**kw) -> RetrievedChunk:
    base = dict(chunk_id=1, document_id=1, text='본문', heading_path=[], page=None,
                filename='환불정책.pdf', version=1)
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


def _t(chunk_id, doc=1, sheet='시트1', **kw):
    return _chunk(chunk_id=chunk_id, document_id=doc, heading_path=[sheet], is_table=True, filename='표.xlsx', **kw)


class TestKeepSingleTable:
    """표 청크는 최상위 **시트 하나**의 것만 유지 (F1a '한 시트만', #222) — 순서 보존 필수."""

    def test_다른_시트의_표는_제외(self):
        chunks = [_t(1), _chunk(chunk_id=2), _t(3, sheet='시트2'), _chunk(chunk_id=4)]
        kept = _keep_single_table(chunks)
        assert [c.chunk_id for c in kept] == [1, 2, 4]        # 다른 시트 표는 빠지고 원래 순서 유지

    def test_다른_파일의_표도_제외(self):
        assert [c.chunk_id for c in _keep_single_table([_t(1), _t(2, doc=2)])] == [1]

    def test_같은_시트의_묶음은_모두_남긴다(self):
        # 여기서 접으면 검색 지표가 정답 묶음을 형제 묶음에 밀려 놓친 것으로 채점한다(#222) — 접기는 prepare_evidence
        chunks = [_t(1), _chunk(chunk_id=2), _t(3), _t(4, sheet='시트2')]
        assert [c.chunk_id for c in _keep_single_table(chunks)] == [1, 2, 3]

    def test_표가_없으면_그대로(self):
        chunks = [_chunk(chunk_id=1), _chunk(chunk_id=2)]
        assert _keep_single_table(chunks) == chunks

    def test_표가_하위_순위에_있어도_첫_시트만(self):
        chunks = [_chunk(chunk_id=1), _t(2), _t(3, sheet='시트2')]
        assert [c.chunk_id for c in _keep_single_table(chunks)] == [1, 2]

    def test_빈_입력(self):
        assert _keep_single_table([]) == []


class TestPrepareEvidence:
    """근거 준비(#222) — 같은 시트 접기 → top_k 자르기 → 시트 완성본으로 확장 → 근거 예산. 운영·생성 평가 공용."""

    def test_같은_시트_묶음은_한_칸으로_접히고_빈_칸은_아래_순위가_채운다(self):
        from rag.retriever import prepare_evidence
        chunks = [_t(1, sheet_text='전체'), _t(2, sheet_text='전체'), _chunk(chunk_id=3), _t(4, sheet_text='전체'),
                  _chunk(chunk_id=5), _chunk(chunk_id=6)]
        out = prepare_evidence(chunks, top_k=3)
        assert [c.chunk_id for c in out] == [1, 3, 5]

    def test_표_묶음은_시트_완성본으로_바뀌고_식별자는_그대로(self):
        from rag.retriever import prepare_evidence
        out = prepare_evidence([_t(7, text='묶음 일부', sheet_text='[설명]\n| h |\n| --- |\n| 1 |\n| 2 |'),
                                _chunk(chunk_id=8, text='산문')], top_k=5)
        assert out[0].text.endswith('| 2 |') and out[0].chunk_id == 7 and out[0].document_id == 1
        assert out[1].text == '산문'

    def test_완성본이_없는_표_청크는_그대로(self):
        # #222 이전에 색인된 엑셀(재인제스트 전)은 sheet_text가 없다 — 본문 그대로 넣는다
        from rag.retriever import prepare_evidence
        assert prepare_evidence([_t(1, text='옛 시트 본문')], top_k=5)[0].text == '옛 시트 본문'

    def test_예산을_넘으면_순위가_낮은_산문부터_뺀다(self, monkeypatch):
        import rag.retriever as R
        monkeypatch.setattr(R, 'EVIDENCE_BUDGET_TOKENS', 100)
        chunks = [_t(1, sheet_text='가' * 100), _chunk(chunk_id=2, text='나' * 50), _chunk(chunk_id=3, text='다' * 50)]
        out = R.prepare_evidence(chunks, top_k=5)
        assert [c.chunk_id for c in out] == [1]                # 표는 남기고 낮은 순위 산문부터 제거

    def test_인용_후보는_문서_단위라_바뀌지_않는다(self):
        # 묶음이 여럿 걸려도 한 시트 = 한 문서 = 인용 번호 하나(#222 교차 지점)
        from rag.citation_labels import sources_from_chunks
        from rag.retriever import prepare_evidence
        chunks = [_t(1, sheet_text='전체'), _t(2, sheet_text='전체'), _chunk(chunk_id=3, document_id=9)]
        assert [s.document_id for s in sources_from_chunks(prepare_evidence(chunks, 5))] == [1, 9]
