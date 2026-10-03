"""xlsx 청커 단위 테스트 — _cell / _to_markdown / chunk_xlsx.

is_table meta와 표 설명 병합은 retriever의 '표는 한 시트만' 필터·검색 보강이 의존한다.
"""
from pathlib import Path

import openpyxl
import pytest

from io import BytesIO

from rag.xlsx_chunking import (XLSX_MAX_ROWS, XlsxDescriptionRequired, XlsxTooManyRows, XlsxUploadRejected,
                               _cell, _to_markdown, chunk_xlsx, description_missing, validate_xlsx_upload)

CORPUS_XLSX = (Path(__file__).resolve().parent.parent
               / 'sample_docs' / 'corpus_v2' / 'homeplus' / 'homeplus_10_멤버십혜택표.xlsx')


class TestCell:
    def test_None은_빈칸_수치는_문자열화(self):
        assert _cell(None) == ''
        assert _cell(3000) == '3000'
        assert _cell(0.5) == '0.5'
        assert _cell('텍스트') == '텍스트'


class TestToMarkdown:
    def test_기본_표_형식(self):
        md = _to_markdown(['등급', '적립률'], [['VIP', '2.0']])
        assert md.splitlines() == [
            '| 등급 | 적립률 |',
            '| --- | --- |',
            '| VIP | 2.0 |',
        ]

    def test_짧은_행은_빈칸_패딩_긴_행은_잘림(self):
        md = _to_markdown(['a', 'b'], [['1'], ['1', '2', '3']])
        lines = md.splitlines()
        assert lines[2] == '| 1 |  |'
        assert lines[3] == '| 1 | 2 |'


class TestChunkXlsx:
    def test_시트당_청크1개_meta와_시트명(self):
        chunks = chunk_xlsx(CORPUS_XLSX)
        # 실제 시트명 고정 — meta==heading_path 자기참조 비교는 둘이 같이 틀려도 통과한다
        assert [c.heading_path for c in chunks] == [['등급기준'], ['등급별쿠폰'], ['포인트정책']]
        assert [c.meta for c in chunks] == [
            {'is_table': True, 'sheet': '등급기준'},
            {'is_table': True, 'sheet': '등급별쿠폰'},
            {'is_table': True, 'sheet': '포인트정책'},
        ]
        assert [c.chunk_index for c in chunks] == [0, 1, 2]

    def test_표_설명_병합(self):
        with_desc = chunk_xlsx(CORPUS_XLSX, description='멤버십 기준표')
        assert with_desc[0].text.startswith('[멤버십 기준표]\n')
        without = chunk_xlsx(CORPUS_XLSX)
        assert not without[0].text.startswith('[')              # 설명 없으면 표만

    def test_빈_시트_스킵(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = '데이터'
        ws.append(['품목', '가격'])
        ws.append(['닭가슴살', 5900])
        wb.create_sheet('빈시트')                                # 값 없는 시트
        p = tmp_path / 't.xlsx'
        wb.save(p)
        chunks = chunk_xlsx(p)
        assert len(chunks) == 1 and chunks[0].heading_path == ['데이터']
        # 헤더 행(rows[0])이 표 첫 줄로 포함 — 첫 데이터 행이 헤더로 둔갑하는 회귀 방지 (뮤테이션 생존자)
        assert chunks[0].text.splitlines()[0] == '| 품목 | 가격 |'

    def test_행_상한_초과시_거절(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['col'])
        for i in range(XLSX_MAX_ROWS + 1):
            ws.append([i])
        p = tmp_path / 'big.xlsx'
        wb.save(p)
        with pytest.raises(XlsxTooManyRows):
            chunk_xlsx(p)

    def test_중간_빈_행은_스킵(self, tmp_path):
        # 빈 행이 표에 끼어도 데이터로 세지 않아야 함 (행 상한 오거절·빈 행 노이즈 방지)
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['col'])
        ws.append([1])
        ws.append([None])
        ws.append([2])
        p = tmp_path / 'gap.xlsx'
        wb.save(p)
        chunks = chunk_xlsx(p)
        assert chunks[0].text.splitlines()[2:] == ['| 1 |', '| 2 |']

    def test_정확히_상한이면_허용(self, tmp_path):
        # off-by-one(> → >=)으로 정상 상한 파일이 거절되는 회귀 방지
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['col'])
        for i in range(XLSX_MAX_ROWS):
            ws.append([i])
        p = tmp_path / 'exact.xlsx'
        wb.save(p)
        assert len(chunk_xlsx(p)) == 1


def _xlsx_bytes(n_rows: int, *, blank_in_middle: int = 0) -> bytes:
    """헤더 1행 + 데이터 n_rows행을 메모리에서 만든다 — 라우터가 받는 것과 같은 '디스크 안 거친 bytes'."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['col'])
    for i in range(n_rows):
        ws.append([i])
        if blank_in_middle and i == n_rows // 2:
            for _ in range(blank_in_middle):
                ws.append([None])
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


class TestValidateXlsxUpload:
    """업로드 사전검증 (#194) — 라우터가 blob을 쓰기 전에 부른다. 거절은 전부 XlsxUploadRejected 하위."""

    def test_설명_없으면_거절(self):
        with pytest.raises(XlsxDescriptionRequired):
            validate_xlsx_upload(_xlsx_bytes(1), None)

    def test_공백만이면_미전송과_동일(self):
        with pytest.raises(XlsxDescriptionRequired):
            validate_xlsx_upload(_xlsx_bytes(1), '  \t ')

    def test_설명_있고_정확히_상한이면_통과(self):
        validate_xlsx_upload(_xlsx_bytes(XLSX_MAX_ROWS), '멤버십 혜택표')     # off-by-one 회귀 방지

    def test_상한_초과면_설명이_있어도_거절(self):
        with pytest.raises(XlsxTooManyRows) as exc:
            validate_xlsx_upload(_xlsx_bytes(XLSX_MAX_ROWS + 1), '설명')
        assert exc.value.rows == XLSX_MAX_ROWS + 1

    def test_설명_검사가_워크북보다_먼저다(self):
        """설명이 비면 워크북을 열지 않는다 — 깨진 바이트를 줘도 설명 예외가 난다(싼 검사 먼저)."""
        with pytest.raises(XlsxDescriptionRequired):
            validate_xlsx_upload(b'not an xlsx at all', None)

    def test_두_거절_모두_공통_베이스로_잡힌다(self):
        """라우터는 XlsxUploadRejected 하나만 잡는다 — 하위 타입이 늘어도 라우터를 안 고치게."""
        for content, desc in [(_xlsx_bytes(1), None), (_xlsx_bytes(XLSX_MAX_ROWS + 1), 'x')]:
            with pytest.raises(XlsxUploadRejected):
                validate_xlsx_upload(content, desc)

    def test_행_세는_기준이_chunk_xlsx와_같다(self, tmp_path):
        """중간 빈 행은 둘 다 세지 않는다 — 같은 _iter_sheets를 쓰므로 '검증 통과·색인 거절'이 없다."""
        content = _xlsx_bytes(XLSX_MAX_ROWS, blank_in_middle=3)
        validate_xlsx_upload(content, '설명')                      # 사전검증 통과
        p = tmp_path / 'edge.xlsx'
        p.write_bytes(content)
        assert len(chunk_xlsx(p, description='설명')) == 1        # 색인도 통과(시트 1 = 청크 1)


class TestDescriptionMissing:
    def test_None_빈문자열_공백만은_없음(self):
        assert description_missing(None)
        assert description_missing('')
        assert description_missing('  \t\n')

    def test_글자가_하나라도_있으면_있음(self):
        assert not description_missing(' a ')
