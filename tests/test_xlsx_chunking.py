"""xlsx 청커 단위 테스트 — _cell / _to_markdown / chunk_xlsx.

is_table·table_key·table_text meta와 표 설명 병합은 근거 준비(retriever.prepare_evidence)의 표 접기·확장과 검색 보강이 의존한다.
"""
from pathlib import Path

import openpyxl
import pytest

from io import BytesIO

from rag.chunking import TABLE_MAX_TOKENS

from rag.xlsx_chunking import (XLSX_BATCH_TOKENS, CHARS_PER_TOKEN, XlsxDescriptionRequired,
                               XlsxTableTooLarge, XlsxUploadRejected, _batch_rows, _cell, _to_markdown, chunk_xlsx,
                               chunk_xlsx_plain, description_missing, estimate_tokens, xlsx_plain_text,
                               validate_xlsx_upload)

CORPUS_XLSX = (Path(__file__).resolve().parent.parent
               / 'sample_docs' / 'corpus_v2' / 'homeplus' / 'homeplus_10_멤버십혜택표.xlsx')


def _gap_xlsx(tmp_path):
    wb = openpyxl.Workbook()
    ws = wb.active
    for row in (['col'], [1], [None], [2]):
        ws.append(row)
    p = tmp_path / 'gap.xlsx'
    wb.save(p)
    return p


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


def _max_rows_at_limit() -> int:
    """헤더 'col' + 숫자 1열 표에서 추정 토큰이 상한 이하인 최대 데이터 행 수 — 경계 테스트용(off-by-one 회귀 방지)."""
    n = 1
    while estimate_tokens(_to_markdown(['col'], [[str(i)] for i in range(n + 1)])) <= TABLE_MAX_TOKENS:
        n += 1
    return n


class TestChunkXlsx:
    def test_작은_시트는_시트당_청크1개이고_본문이_곧_완성본(self):
        chunks = chunk_xlsx(CORPUS_XLSX)
        # 실제 시트명 고정 — meta==heading_path 자기참조 비교는 둘이 같이 틀려도 통과한다
        assert [c.heading_path for c in chunks] == [['등급기준'], ['등급별쿠폰'], ['포인트정책']]
        assert [(c.meta['is_table'], c.meta['table_key']) for c in chunks] == [
            (True, '등급기준'), (True, '등급별쿠폰'), (True, '포인트정책')]
        assert all(c.meta['table_text'] == c.text for c in chunks)   # 한 묶음 = 시트 전체(#222 이전과 같은 모양)
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

    @pytest.mark.parametrize('chunker', [chunk_xlsx, chunk_xlsx_plain], ids=['docling', 'openpyxl'])
    def test_큰_시트는_행_묶음으로_나뉘고_묶음마다_설명과_헤더를_반복한다(self, tmp_path, chunker):
        # #222 — 검색은 작은 묶음으로, 모델에는 시트 완성본으로. 두 읽기 경로가 같은 모양을 내야 한다
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = '단가'
        ws.append(['상품코드', '상품명', '단가'])
        for i in range(200):
            ws.append([f'A-{i:04d}', f'상품 {i}', 1000 + i])
        p = tmp_path / 'big.xlsx'
        wb.save(p)
        chunks = chunker(p, description='단가표')
        assert len(chunks) > 1
        budget = int(XLSX_BATCH_TOKENS * CHARS_PER_TOKEN)
        for c in chunks:
            lines = c.text.splitlines()
            assert lines[:3] == ['[단가표]', '| 상품코드 | 상품명 | 단가 |', '| --- | --- | --- |']
            assert len(c.text) <= budget + len('[단가표]\n')
            assert c.meta['table_key'] == '단가' and c.heading_path == ['단가']
        whole = chunks[0].meta['table_text']
        assert all(c.meta['table_text'] == whole for c in chunks)
        assert whole.count('| 상품코드 |') == 1 and whole.count('[단가표]') == 1   # 완성본엔 헤더·설명 한 번
        body = [l for c in chunks for l in c.text.splitlines()[3:]]
        assert body == whole.splitlines()[3:]                     # 묶음을 이으면 행이 빠짐·중복 없이 완성본과 같다
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    @pytest.mark.parametrize('header, body', [
        (['구분\n(원)', '단가'], [[f'항목{i}', str(1000 + i)] for i in range(120)]),            # 헤더 칸 줄바꿈
        (['코드', '설명'], [[f'A{i}', f'첫줄\n| 둘째줄 {i}'] for i in range(120)]),             # 칸 줄바꿈 뒤 '|'로 시작
        (['코드', '비고'], [[f'A{i}', '---\n--- | ---' if i % 7 == 0 else '정상'] for i in range(120)]),  # '---' 든 칸
        (['a', 'b', 'c'], [['', str(i), ''] for i in range(120)]),                             # 빈 칸 많은 행
    ], ids=['헤더_줄바꿈', '칸_줄바꿈_뒤_파이프', '대시_든_칸', '빈_칸'])
    def test_행_묶음은_행_단계에서_나눠_헤더와_행이_깨지지_않는다(self, header, body):
        # #222 리뷰 — 렌더한 글을 줄 단위로 다시 해석하던 첫 구현은 앞의 두 경우에서 헤더가 깨지고 행이 중간에서 잘렸다
        batches = _batch_rows(header, body, budget_chars=400)
        head = _to_markdown(header, [])
        assert len(batches) > 1 and all(b.startswith(head + '\n') for b in batches)   # 묶음마다 헤더 전체 + 구분선
        rebuilt, i = [], 0                                                       # 묶음 = 원래 행을 순서대로 나눈 것
        for b in batches:
            k = 0
            while i + k < len(body) and _to_markdown(header, body[i:i + k + 1]) == b[:len(_to_markdown(header, body[i:i + k + 1]))]:
                k += 1
            assert b == _to_markdown(header, body[i:i + k])                       # 행 경계에서만 잘렸다
            rebuilt += body[i:i + k]
            i += k
        assert rebuilt == body
        assert all(len(b) <= 400 for b in batches)

    def test_행_하나가_예산보다_길면_혼자_한_묶음이고_자르지_않는다(self):
        batches = _batch_rows(['h'], [['짧음'], ['긴' * 500], ['짧음2']], budget_chars=50)
        assert batches == [_to_markdown(['h'], [['짧음']]), _to_markdown(['h'], [['긴' * 500]]), _to_markdown(['h'], [['짧음2']])]

    def test_데이터_행이_없으면_헤더만(self):
        assert _batch_rows(['h1', 'h2'], [], budget_chars=50) == [_to_markdown(['h1', 'h2'], [])]

    def test_크기_상한_초과시_거절_문구는_토큰을_말하지_않는다(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = '단가표'
        ws.append(['col'])
        for i in range(_max_rows_at_limit() + 1):
            ws.append([i])
        p = tmp_path / 'big.xlsx'
        wb.save(p)
        with pytest.raises(XlsxTableTooLarge) as exc:
            chunk_xlsx(p)
        msg = str(exc.value)
        # #222 결정 9 — 허용 크기 대비 %와 현재 행×열. 토큰이라는 말·나눠 올리라는 안내는 없다
        assert msg.startswith("시트 '단가표'의 표가 허용 크기의 1") and '%입니다(현재 ' in msg and '행 × 1열)' in msg
        assert '토큰' not in msg and '나눠' not in msg and '줄여 다시 올려 주세요' in msg

    def test_중간_빈_행은_스킵(self, tmp_path):
        # 빈 행이 표에 끼어도 데이터로 세지 않아야 함 (크기 오판·빈 행 노이즈 방지) — openpyxl 경로
        p = _gap_xlsx(tmp_path)
        chunks = chunk_xlsx_plain(p)
        assert chunks[0].text.splitlines()[2:] == ['| 1 |', '| 2 |']

    def test_docling은_완전히_빈_행에서_표를_가르지만_값은_한_청크에_남는다(self, tmp_path):
        # #220 결정 9 — docling 기본(gap_tolerance=0)은 완전히 빈 행을 표 경계로 본다. 아랫부분은 첫 데이터 행이
        # 헤더가 되지만 값은 빠지지 않고 같은 시트 청크에 이어진다. 허용치 1은 안내문·양식 시트를 openpyxl처럼
        # 잘못 읽어(실문서 2건 모두 해당) 기각했다. 이 테스트는 그 대가를 고정한다 — 바뀌면 결정을 다시 볼 것.
        p = _gap_xlsx(tmp_path)
        chunks = chunk_xlsx(p)
        assert len(chunks) == 2                 # 표 두 개 → 묶음 두 개(#222), 같은 시트 완성본
        whole = chunks[0].meta['table_text'].splitlines()
        assert whole[:3] == ['| col |', '| --- |', '| 1 |'] and '| 2 |' in whole   # 갈린 아랫부분 — 값 보존

    def test_정확히_상한이면_허용(self, tmp_path):
        # off-by-one(> → >=)으로 정상 상한 파일이 거절되는 회귀 방지
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['col'])
        for i in range(_max_rows_at_limit()):
            ws.append([i])
        p = tmp_path / 'exact.xlsx'
        wb.save(p)
        assert chunk_xlsx(p)

    def test_첨부_텍스트는_묶지_않고_시트_전체(self, tmp_path):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['col'])
        for i in range(300):
            ws.append([f'값{i}'])
        p = tmp_path / 'a.xlsx'
        wb.save(p)
        assert xlsx_plain_text(p).count('| col |') == 1

    def test_첨부는_색인용_크기_상한을_걸지_않는다(self, tmp_path):
        # #222 리뷰 — 첨부엔 자체 상한(6천 자, 라우터가 사유 안내)이 있다. 여기서 막으면 "추출할 수 없는 파일"로 가려진다
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.append(['col'])
        for i in range(_max_rows_at_limit() + 50):
            ws.append([i])
        p = tmp_path / 'big.xlsx'
        wb.save(p)
        with pytest.raises(XlsxTableTooLarge):
            chunk_xlsx(p)                                    # 색인은 막고
        assert xlsx_plain_text(p)                            # 첨부 추출은 통과(크기는 라우터의 6천 자 상한이 판단)


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
    """업로드 사전검증 (#194·#222) — 라우터가 blob을 쓰기 전에 부른다. 거절은 전부 XlsxUploadRejected 하위."""

    def test_설명_없으면_거절(self):
        with pytest.raises(XlsxDescriptionRequired):
            validate_xlsx_upload(_xlsx_bytes(1), None)

    def test_공백만이면_미전송과_동일(self):
        with pytest.raises(XlsxDescriptionRequired):
            validate_xlsx_upload(_xlsx_bytes(1), '  \t ')

    def test_설명_있고_정확히_상한이면_통과(self):
        validate_xlsx_upload(_xlsx_bytes(_max_rows_at_limit()), '멤버십 혜택표')     # off-by-one 회귀 방지

    def test_상한_초과면_설명이_있어도_거절(self):
        n = _max_rows_at_limit() + 1
        with pytest.raises(XlsxTableTooLarge) as exc:
            validate_xlsx_upload(_xlsx_bytes(n), '설명')
        assert exc.value.rows == n and exc.value.cols == 1 and exc.value.percent > 100

    def test_설명_검사가_워크북보다_먼저다(self):
        """설명이 비면 워크북을 열지 않는다 — 깨진 바이트를 줘도 설명 예외가 난다(싼 검사 먼저)."""
        with pytest.raises(XlsxDescriptionRequired):
            validate_xlsx_upload(b'not an xlsx at all', None)

    def test_두_거절_모두_공통_베이스로_잡힌다(self):
        """라우터는 XlsxUploadRejected 하나만 잡는다 — 하위 타입이 늘어도 라우터를 안 고치게."""
        for content, desc in [(_xlsx_bytes(1), None), (_xlsx_bytes(_max_rows_at_limit() + 1), 'x')]:
            with pytest.raises(XlsxUploadRejected):
                validate_xlsx_upload(content, desc)

    def test_크기_재는_기준이_chunk_xlsx와_같다(self, tmp_path):
        """중간 빈 행은 둘 다 세지 않는다 — 같은 _iter_sheets·렌더를 쓰므로 '검증 통과·색인 거절'이 없다."""
        content = _xlsx_bytes(_max_rows_at_limit(), blank_in_middle=3)
        validate_xlsx_upload(content, '설명')                      # 사전검증 통과
        p = tmp_path / 'edge.xlsx'
        p.write_bytes(content)
        assert chunk_xlsx(p, description='설명')                  # 색인도 통과


class TestDescriptionMissing:
    def test_None_빈문자열_공백만은_없음(self):
        assert description_missing(None)
        assert description_missing('')
        assert description_missing('  \t\n')

    def test_글자가_하나라도_있으면_있음(self):
        assert not description_missing(' a ')
