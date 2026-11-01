"""표 파싱 검증 하네스의 순수 로직 — docling·서버 없이. 기대 렌더 규칙(expected), markdown 표 파싱·채점(score).

채점기가 틀리면 측정 전체가 틀린다 — 조립이 틀린 채로 측정한 사고(AGENTS.md)를 막는 테스트다.
하네스 부속(문서 빌드·질문·E2E 집계·진단)은 tests/test_table_parsing_harness.py가 본다.
"""

from types import SimpleNamespace

from eval.table_parsing.expected import Cell, expected_cells, expected_rows, expected_tables, flat
from eval.table_parsing.score import ActualTable, chunk_tables, docling_tables, match_tables, score_cells, score_rows
from eval.table_parsing.specs import BY_ID, SPECS, paths_of

flat_dims = flat

S02 = BY_ID['S02_colspan']
S16 = BY_ID['S16_rowheader']


class TestSpecs:
    def test_정의는_23축_하나씩이고_격자가_반듯하다(self):
        axes = {}
        for s in SPECS:
            axes.setdefault(s['axis'], []).append(s['id'])
            for t in expected_tables(s):
                n, m = flat(t)
                for r, row in enumerate(t):
                    assert len(row) == m, (s['id'], r)  # 들쭉날쭉한 행은 격자 비교를 깨뜨린다
                for c in expected_cells(t):  # 병합 범위가 격자 밖으로 안 나간다
                    assert c.row + c.row_span <= n and c.col + c.col_span <= m, (s['id'], c)
        assert all(len(v) == 1 for v in axes.values()), {k: v for k, v in axes.items() if len(v) != 1}
        assert len(axes) == 23
        assert all(set(paths_of(s)) <= {'img', 'pdf', 'docx', 'xlsx'} for s in SPECS)


class TestExpectedRows:
    def test_전_경로_공통_grid_규칙_병합은_가로_세로_모두_칸마다(self):
        # PDF도 docling_grid_rows — 2단 헤더를 점으로 잇지 않고 첫 행이 헤더, 둘째 헤더 행은 데이터 행
        for path in ('pdf', 'docx', 'xlsx', 'img'):
            rows = expected_rows(S02, S02['rows'], path)
            assert rows[0] == ['직급', '국내 출장', '국내 출장', '해외 출장', '해외 출장']  # 가로 병합은 칸마다 반복
            assert rows[1] == ['직급', '일비', '숙박비', '일비', '숙박비'] and len(rows) == 5

    def test_가로_머리글_표도_첫_행이_헤더다(self):
        # 어느 경로든 첫 행이 헤더다(열 이름 0,1,2…가 아니라)
        rows = expected_rows(S16, S16['rows'], 'pdf')
        assert rows[0] == ['문서명', '휴양시설 이용 안내'] and rows[1] == ['개정일', '2026-03-01']

    def test_중첩_표는_L2에선_어느_경로든_평탄화_L1_docx만_바깥_안쪽_따로(self):
        s = BY_ID['V15_nested']
        for path in ('pdf', 'docx', 'img'):
            (flat,) = expected_tables(s, path)  # L2: docling PDF 모양 5×3 — DOCX도 같은 모양
            assert flat_dims(flat) == (5, 3)
        assert len(expected_tables(s, 'docx', 'l1')) == 2 and len(expected_tables(s, 'pdf', 'l1')) == 1
        cells = {(c.row, c.col): (c.text, c.row_span, c.col_span) for c in expected_cells(flat)}
        assert cells[(0, 1)] == ('세부', 1, 2) and cells[(1, 0)] == ('평가 항목', 3, 1)
        assert (
            cells[(1, 1)] == ('항목', 1, 1) and cells[(3, 2)] == ('40', 1, 1) and cells[(4, 1)] == ('매년 12월', 1, 2)
        )
        assert expected_rows(s, flat, 'docx')[1] == ['평가 항목', '항목', '배점']  # grid 규칙: 세로 병합 반복


class TestChunkTables:
    def _chunks(self, *texts):
        return [SimpleNamespace(text=t, meta=None) for t in texts]

    def test_구분선으로_표를_가르고_헤더를_잡는다(self):
        t = chunk_tables(self._chunks('머리말\n| a | b |\n| --- | --- |\n| 1 | 2 |\n본문\n| c |\n| --- |\n| 3 |'))
        assert t == [[['a', 'b'], ['1', '2']], [['c'], ['3']]]

    def test_청크_경계에서_갈린_긴_표는_앞_표에_잇는다(self):
        t = chunk_tables(self._chunks('| a | b |\n| --- | --- |\n| 1 | 2 |', '| 3 | 4 |\n| 5 | 6 |'))
        assert t == [[['a', 'b'], ['1', '2'], ['3', '4'], ['5', '6']]]

    def test_같은_헤더가_반복되는_행_묶음은_한_표다(self):
        # 긴 표는 행 묶음마다 헤더를 반복한다 — 묶음 둘을 표 하나로 읽어야 셀 대조가 된다. 다른 헤더면 새 표
        t = chunk_tables(
            self._chunks(
                '| a | b |\n| --- | --- |\n| 1 | 2 |', '| a | b |\n| --- | --- |\n| 3 | 4 |', '| c |\n| --- |\n| 9 |'
            )
        )
        assert t == [[['a', 'b'], ['1', '2'], ['3', '4']], [['c'], ['9']]]

    def test_xlsx는_시트_완성본만_읽는다(self):
        c = [
            SimpleNamespace(
                text='[표]\n| a |\n| --- |\n| 1 |', meta={'table_text': '[표]\n| a |\n| --- |\n| 1 |\n| 2 |'}
            ),
            SimpleNamespace(
                text='[표]\n| a |\n| --- |\n| 2 |', meta={'table_text': '[표]\n| a |\n| --- |\n| 1 |\n| 2 |'}
            ),
        ]
        assert chunk_tables(c, xlsx=True) == [[['a'], ['1'], ['2']]]

    def test_PDF_큰_표_묶음은_완성본이_있어도_묶음에서_잇는다(self):
        # 표만 있는 PDF는 모든 청크가 표 묶음(table_text 있음)이라 청크 모양으로 xlsx와 구별되지 않는다 — 형식은 호출부가 준다
        whole = '| a |\n| --- |\n| 1 |\n| 2 |'
        c = [
            SimpleNamespace(text='| a |\n| --- |\n| 1 |', meta={'table_text': whole}),
            SimpleNamespace(text='| a |\n| --- |\n| 2 |', meta={'table_text': whole}),
        ]
        c[1].text = '| a |\n| --- |\n| 2 | 깨진 칸 |'  # 묶음이 깨지면 그대로 드러나야 한다(완성본으로 가리지 않는다)
        assert chunk_tables(c) == [[['a'], ['1'], ['2', '깨진 칸']]]


class TestScore:
    def _exp(self):
        return [Cell(0, 0, 1, 2, '제목'), Cell(1, 0, 1, 1, '값 1'), Cell(1, 1, 1, 1, '3,000원')]

    def test_전부_맞으면_오류_없음(self):
        act = ActualTable([Cell(0, 0, 1, 2, '제목'), Cell(1, 0, 1, 1, '값 1'), Cell(1, 1, 1, 1, '3,000원')], 2, 2)
        s = score_cells(self._exp(), act, (2, 2))
        assert s.cells_ok == 3 and (s.numeric_ok, s.numeric_total) == (2, 2) and s.span_ok == 1 and s.kinds == []

    def test_오독_누락_병합을_가른다(self):
        act = ActualTable([Cell(0, 0, 1, 1, '제목'), Cell(1, 0, 1, 1, ''), Cell(1, 1, 1, 1, '3,OOO원')], 2, 2)
        s = score_cells(self._exp(), act, (2, 2))
        assert s.kinds == ['누락', '병합', '오독'] and s.numeric_ok == 0
        assert s.presence_ok == 1  # '제목'만 어딘가에 있다

    def test_행열이_다르면_구조이고_존재율은_따로_센다(self):
        act = ActualTable([Cell(0, 0, 1, 1, '제목'), Cell(0, 1, 1, 1, '값 1'), Cell(0, 2, 1, 1, '3,000원')], 1, 3)
        s = score_cells(self._exp(), act, (2, 2))
        assert not s.dims_ok and '구조' in s.kinds and s.presence_ok == 3

    def test_표를_못_찾으면_미검출(self):
        s = score_cells(self._exp(), None, (2, 2))
        assert s.kinds == ['미검출'] and not s.detected and s.cells_total == 3

    def test_공백_정규화는_하되_붙여쓰기는_오독이다(self):
        # OCR이 '지원 기준'을 '지원기준'으로 내는 것은 검색 토큰이 달라지므로 맞은 것으로 치지 않는다
        act = ActualTable([Cell(0, 0, 1, 2, '제목'), Cell(1, 0, 1, 1, '값  1'), Cell(1, 1, 1, 1, '3,000원')], 2, 2)
        assert score_cells(self._exp(), act, (2, 2)).cells_ok == 3
        act = ActualTable([Cell(0, 0, 1, 2, '제목'), Cell(1, 0, 1, 1, '값1'), Cell(1, 1, 1, 1, '3,000원')], 2, 2)
        assert score_cells(self._exp(), act, (2, 2)).kinds == ['오독']

    def test_표_짝짓기는_글자가_가장_많이_겹치는_것(self):
        a = ActualTable([Cell(0, 0, 1, 1, '국내'), Cell(1, 0, 1, 1, '임원')], 2, 1)
        b = ActualTable([Cell(0, 0, 1, 1, '해외'), Cell(1, 0, 1, 1, '임원')], 2, 1)
        exp = [
            [Cell(0, 0, 1, 1, '해외'), Cell(1, 0, 1, 1, '임원')],
            [Cell(0, 0, 1, 1, '국내'), Cell(1, 0, 1, 1, '임원')],
        ]
        assert match_tables(exp, [a, b]) == [b, a]
        # 기대 표가 둘인데 둘 다 겹침이 없으면 짝을 안 짓는다(미검출). 하나뿐일 때의 예외는 아래 테스트
        assert match_tables([[Cell(0, 0, 1, 1, '없는글자')], [Cell(0, 0, 1, 1, '또없음')]], [a, b]) == [None, None]

    def test_L2_행_채점은_같은_규칙(self):
        s = score_rows([['a', 'b'], ['1', '2']], [['a', 'b'], ['1', '9']])
        assert s.cells_ok == 3 and s.kinds == ['오독'] and (s.numeric_ok, s.numeric_total) == (1, 2)

    def test_기대에_없는_병합이_생겨도_병합_오류다(self):
        act = ActualTable([Cell(0, 0, 1, 2, '제목'), Cell(1, 0, 1, 2, '값 1'), Cell(1, 1, 1, 1, '3,000원')], 2, 2)
        assert '병합' in score_cells(self._exp(), act, (2, 2)).kinds

    def test_존재율은_띄어쓰기를_무시한_정확_일치(self):
        # '5'가 '150,000' 안에 있다고 존재로 치면 안 된다(리뷰) — 부분 문자열이 아니라 칸 단위 일치
        exp = [Cell(0, 0, 1, 1, '5'), Cell(0, 1, 1, 1, '지원 기준')]
        act = ActualTable([Cell(0, 0, 1, 1, '150,000'), Cell(0, 1, 1, 1, '지원기준')], 1, 2)
        s = score_cells(exp, act, (1, 2))
        assert s.presence_ok == 1 and s.cells_ok == 0

    def test_존재율_풀을_주면_짝지은_표_밖도_센다(self):
        # 쪽 넘김으로 갈린 뒷 표의 글자를 "유실"이 아니라 "갈림"으로 가르기 위해서다
        exp = [Cell(0, 0, 1, 1, 'a'), Cell(1, 0, 1, 1, 'b')]
        act = ActualTable([Cell(0, 0, 1, 1, 'a')], 1, 1)
        assert score_cells(exp, act, (2, 1), {'a', 'b'}).presence_ok == 2

    def test_겹침이_없어도_기대_표가_하나면_남은_표와_짝짓는다(self):
        # OCR이 전부 틀린 표를 미검출로 세지 않는다 — 오독으로 집계된다
        a = ActualTable([Cell(0, 0, 1, 1, '틀린글자')], 1, 1)
        assert match_tables([[Cell(0, 0, 1, 1, '정답')]], [a]) == [a]


class TestDoclingDrift:
    """기대 렌더 규칙이 docling_core·rag/chunking의 실제 함수와 같은지 — 규칙을 손으로 베낀 것이라 드리프트를 여기서 잡는다."""

    def _item(self, spec, header_rows=1):
        from docling_core.types.doc import TableCell, TableData, TableItem

        cells = []
        for c in expected_cells(spec['rows']):
            cells.append(
                TableCell(
                    start_row_offset_idx=c.row,
                    end_row_offset_idx=c.row + c.row_span,
                    start_col_offset_idx=c.col,
                    end_col_offset_idx=c.col + c.col_span,
                    row_span=c.row_span,
                    col_span=c.col_span,
                    text=c.text,
                    column_header=c.row < header_rows,
                )
            )
        n, m = flat(spec['rows'])
        return TableItem(self_ref='#/tables/0', data=TableData(num_rows=n, num_cols=m, table_cells=cells))

    def test_기대_규칙은_rag_chunking의_grid_렌더와_같다(self):
        # PDF·DOCX·XLSX 모두 docling_grid_rows 하나다 — 하네스 기대 규칙이 그 함수와 어긋나면 여기서 잡힌다
        from rag.chunking import docling_grid_rows

        for spec in (S02, BY_ID['S03_rowspan'], BY_ID['S04_form'], S16):
            for path in ('pdf', 'docx'):
                assert docling_grid_rows(self._item(spec)) == expected_rows(spec, spec['rows'], path), (
                    spec['id'],
                    path,
                )

    def test_docling_tables는_병합_원점만_남기고_헤더행을_센다(self):
        from docling_core.types.doc import DoclingDocument

        doc = DoclingDocument(name='t')
        doc.add_table(data=self._item(S02, header_rows=2).data)  # add_table이 body 참조까지 건다
        (t,) = docling_tables(doc)
        assert (t.n_rows, t.n_cols, t.header_rows) == (5, 5, 2)
        assert len(t.cells) == len(expected_cells(S02['rows']))  # grid의 반복 칸이 아니라 원점만
