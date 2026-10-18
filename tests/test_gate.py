"""근거 게이트 단위 테스트 — retriever.apply_gate.

strict-grounded의 핵심 분기: 후보 없음 / 거리 초과 / 정상.
"""
from rag.retriever import apply_gate
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
    return _chunk(chunk_id=chunk_id, document_id=doc, heading_path=[sheet], is_table=True, table_key=sheet,
                  filename='표.xlsx', **kw)


def _pt(chunk_id, key, doc=1, **kw):
    """PDF·DOCX 큰 표의 묶음(#257) — table_key는 문서 안 표 순번."""
    return _chunk(chunk_id=chunk_id, document_id=doc, heading_path=['규정'], is_table=True, table_key=key,
                  filename='규정.pdf', **kw)


def test_한_시트만_필터는_없다():
    # #257 결정 1 — 표는 몇 개든 후보에 남는다(PDF는 한 문서에 표가 여럿인 게 정상). 접기·확장·예산은 prepare_evidence
    import rag.retriever as R
    assert not hasattr(R, '_keep_single_table')


class TestPrepareEvidence:
    """근거 준비(#222·#257) — 같은 표 접기 → top_k 자르기 → 표 전체로 확장 → 근거 예산. 운영·생성 평가 공용."""

    def test_같은_시트_묶음은_한_칸으로_접히고_빈_칸은_아래_순위가_채운다(self):
        from rag.retriever import prepare_evidence
        chunks = [_t(1, table_text='전체'), _t(2, table_text='전체'), _chunk(chunk_id=3), _t(4, table_text='전체'),
                  _chunk(chunk_id=5), _chunk(chunk_id=6)]
        out = prepare_evidence(chunks, top_k=3)
        assert [c.chunk_id for c in out] == [1, 3, 5]

    def test_표_묶음은_시트_완성본으로_바뀌고_식별자는_그대로(self):
        from rag.retriever import prepare_evidence
        out = prepare_evidence([_t(7, text='묶음 일부', table_text='[설명]\n| h |\n| --- |\n| 1 |\n| 2 |'),
                                _chunk(chunk_id=8, text='산문')], top_k=5)
        assert out[0].text.endswith('| 2 |') and out[0].chunk_id == 7 and out[0].document_id == 1
        assert out[1].text == '산문'

    def test_완성본이_없는_표_청크는_그대로(self):
        # #257 이전에 색인된 표(재인제스트 전)는 table_text가 없다 — 본문 그대로 넣는다
        from rag.retriever import prepare_evidence
        assert prepare_evidence([_t(1, text='옛 시트 본문')], top_k=5)[0].text == '옛 시트 본문'

    def test_예산을_넘으면_순위가_낮은_산문부터_뺀다(self, monkeypatch):
        import rag.retriever as R
        monkeypatch.setattr(R, 'EVIDENCE_BUDGET_TOKENS', 100)
        chunks = [_t(1, table_text='가' * 100), _chunk(chunk_id=2, text='나' * 50), _chunk(chunk_id=3, text='다' * 50)]
        out = R.prepare_evidence(chunks, top_k=5)
        assert [c.chunk_id for c in out] == [1]                # 표는 남기고 낮은 순위 산문부터 제거

    def test_인용_후보는_문서_단위라_바뀌지_않는다(self):
        # 묶음이 여럿 걸려도 한 시트 = 한 문서 = 인용 번호 하나(#222 교차 지점)
        from rag.citation_labels import sources_from_chunks
        from rag.retriever import prepare_evidence
        chunks = [_t(1, table_text='전체'), _t(2, table_text='전체'), _chunk(chunk_id=3, document_id=9)]
        assert [s.document_id for s in sources_from_chunks(prepare_evidence(chunks, 5))] == [1, 9]

    def test_서로_다른_표는_모두_남고_각자_표_전체로_바뀐다(self):
        # #257 결정 1 — 옛 '한 시트만' 필터라면 시트2·다른 파일 표는 빠졌다
        from rag.retriever import prepare_evidence
        chunks = [_t(1, table_text='시트1 전체'), _t(2, sheet='시트2', table_text='시트2 전체'),
                  _pt(3, 't1', doc=5, table_text='표1 전체'), _pt(4, 't2', doc=5, table_text='표2 전체')]
        assert [c.text for c in prepare_evidence(chunks, top_k=5)] == ['시트1 전체', '시트2 전체', '표1 전체', '표2 전체']

    def test_PDF_큰_표_묶음은_문서와_표_순번으로_접힌다(self):
        from rag.retriever import prepare_evidence
        chunks = [_pt(1, 't1', table_text='A표'), _pt(2, 't1', table_text='A표'),     # 같은 문서 같은 표 → 하나로
                  _pt(3, 't1', doc=2, table_text='B표'),                              # 다른 문서의 t1은 다른 표
                  _chunk(chunk_id=4, text='산문')]
        out = prepare_evidence(chunks, top_k=5)
        assert [(c.chunk_id, c.text) for c in out] == [(1, 'A표'), (3, 'B표'), (4, '산문')]

    def test_산문이_없으면_순위가_낮은_표를_통째로_뺀다(self, monkeypatch):
        # #257 결정 4 — 예전엔 표만 남으면 멈춰 예산을 넘긴 채 넘겼다(표가 하나뿐이라 닿지 않았다)
        import rag.retriever as R
        monkeypatch.setattr(R, 'EVIDENCE_BUDGET_TOKENS', 100)
        chunks = [_t(1, table_text='가' * 60), _t(2, sheet='시트2', table_text='나' * 60), _pt(3, 't1', doc=7, table_text='다' * 60)]
        out = R.prepare_evidence(chunks, top_k=5)
        assert [(c.chunk_id, len(c.text)) for c in out] == [(1, 60)]          # 일부만 남기지 않는다 — 통째로 빠진다

    def test_산문을_먼저_빼고_그래도_넘치면_표를_뺀다(self, monkeypatch):
        import rag.retriever as R
        monkeypatch.setattr(R, 'EVIDENCE_BUDGET_TOKENS', 100)
        chunks = [_t(1, table_text='가' * 60), _chunk(chunk_id=2, text='나' * 20), _t(3, sheet='시트2', table_text='다' * 60)]
        assert [c.chunk_id for c in R.prepare_evidence(chunks, top_k=5)] == [1]

    def test_1등은_혼자_예산을_넘어도_남는다(self, monkeypatch):
        import rag.retriever as R
        monkeypatch.setattr(R, 'EVIDENCE_BUDGET_TOKENS', 10)
        out = R.prepare_evidence([_chunk(chunk_id=1, text='가' * 50), _t(2, table_text='나' * 50)], top_k=5)
        assert [c.chunk_id for c in out] == [1]

    def test_식별자로_걸린_표_묶음도_표_전체가_그대로_들어간다(self):
        # #238 오리기는 확장 뒤에 돈다 — 묶음 머리로 옮겨 온 조 제목(headings)이 있어도 표 전체엔 그 줄이 없어 바뀌지 않는다
        from rag.retriever import prepare_evidence
        whole = '[별표 1] 징계 양정 기준\n| 사유 | 양정 |\n| --- | --- |\n| 횡령 | 파면 |\n| 지각 | 견책 |'
        ch = _pt(1, 't1', text='제5조(징계)\n[별표 1] 징계 양정 기준\n| 사유 | 양정 |\n| --- | --- |\n| 횡령 | 파면 |',
                 table_text=whole, headings=['제5조(징계)'], matched_ident='5조')
        assert prepare_evidence([ch], top_k=5)[0].text == whole
