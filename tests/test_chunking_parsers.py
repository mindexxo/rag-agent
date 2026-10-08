"""청킹 파서 단위 테스트 — _pdf_pages / _docx_text / chunk_file (비ML 파서 리팩토링 검증).

입력은 corpus_v2 합성 문서 실물. heading_path는 pdf·docx·md 모두 채워진다
(txt만 헤딩 개념이 없어 빈다).
"""
from pathlib import Path

import pytest

from rag.chunking import _docx_text, _md_sections, _pdf_pages, chunk_file
from config import settings


@pytest.fixture(autouse=True)
def _pdfplumber_path(monkeypatch):
    """PDF 케이스는 pdfplumber **비상 경로**(docling_enabled=False)를 검사한다 — 이 파일은 형식 분기와
    파서 계약이 대상이고 모델 의존이 없어야 한다. docling 경로는 tests/test_chunking_docling.py."""
    monkeypatch.setattr(settings, 'docling_enabled', False)

CORPUS = Path(__file__).resolve().parent.parent / 'sample_docs' / 'corpus_v2'

PDF = CORPUS / 'summers' / 'summers_01_환불반품정책.pdf'
DOCX = CORPUS / 'goodpeople' / 'goodpeople_02_정기후원신청변경해지.docx'
MD = CORPUS / 'harim' / 'harim_06_대량주문B2B.md'


class TestPdfPages:
    def test_페이지_번호_1부터_텍스트_보존(self):
        pages = _pdf_pages(PDF)
        assert pages[0][0] == 1
        assert [n for n, _ in pages] == list(range(1, len(pages) + 1))
        assert '환불' in pages[0][1]


    def test_빈_페이지는_빈_문자열_그리고_청킹_무해(self, tmp_path):
        # 스캔/빈 페이지에서 extract_text()가 None — '' 폴백이 없으면 split_text(None) 크래시
        from reportlab.pdfgen import canvas
        p = tmp_path / 'two.pdf'
        c = canvas.Canvas(str(p))
        c.drawString(100, 700, 'page one text')
        c.showPage()
        c.showPage()          # 내용 없는 2페이지
        c.save()
        pages = _pdf_pages(p)
        assert pages[1] == (2, '')
        chunks = chunk_file(p)                       # 빈 페이지가 있어도 크래시 없이
        assert all(c.page == 1 for c in chunks)      # 빈 페이지는 청크를 만들지 않음


class TestDocxText:
    def test_문단과_표_셀_포함(self):
        text = _docx_text(DOCX)
        assert '정기후원' in text
        assert ' | ' in text          # 표 행은 셀을 ' | '로 조인 — 표 누락 방지


class TestChunkTxt:
    def test_기본_청킹(self, tmp_path):
        p = tmp_path / 'a.txt'
        p.write_text('첫 문장입니다. 둘째 문장입니다.', encoding='utf-8')
        chunks = chunk_file(p)
        assert len(chunks) == 1
        assert chunks[0].text == '첫 문장입니다. 둘째 문장입니다.'
        assert chunks[0].heading_path == [] and chunks[0].page is None

    def test_긴_텍스트는_분할되고_index_순차(self, tmp_path):
        p = tmp_path / 'b.txt'
        p.write_text('문장입니다. ' * 200, encoding='utf-8')   # 512자 초과
        chunks = chunk_file(p)
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_빈_파일은_청크_없음(self, tmp_path):
        # split_text('')==[''] 함정 — 빈 청크가 임베딩·DB로 새지 않아야 함 (PDF 가드와 동일)
        p = tmp_path / 'c.txt'
        p.write_text('   \n  ', encoding='utf-8')
        assert chunk_file(p) == []


class TestChunkFile:
    def test_pdf_page_보존_heading_채움(self):
        chunks = chunk_file(PDF)
        assert chunks, 'PDF에서 청크 0개'
        assert all(c.heading_path for c in chunks)         # 글자 크기로 헤딩 판정
        assert all(c.page is not None and c.page >= 1 for c in chunks)
        assert len({c.page for c in chunks}) >= 2          # 다중 페이지 문서여야 아래 연속성 검증이 유효
        # chunk_index는 섹션·페이지 경계를 넘어 문서 전체에서 연속 (리셋 회귀 방지)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_pdf_크기차이_없으면_heading_없이_폴백(self, tmp_path):
        # 한 가지 글자 크기만 쓰는 PDF(스캔·단순 조판) → 헤딩 0개, 변경 전과 동일 동작
        from reportlab.pdfgen import canvas
        p = tmp_path / 'flat.pdf'
        c = canvas.Canvas(str(p))
        for i in range(12):
            c.setFont('Helvetica', 11)
            c.drawString(60, 760 - i * 22, f'Uniform body line number {i} with policy content.')
        c.save()
        chunks = chunk_file(p)
        assert chunks
        assert all(ch.heading_path == [] for ch in chunks)

    def test_docx_heading_path_채움_page_없음(self):
        chunks = chunk_file(DOCX)
        assert chunks
        assert all(c.page is None for c in chunks)         # Word는 페이지를 파일에 저장하지 않음
        assert all(c.heading_path for c in chunks)         # Heading 스타일 → 섹션 경로
        assert any('해지' in h for c in chunks for h in c.heading_path)

    def test_docx_헤딩만_있는_청크_없음(self):
        # 본문 없는 상위 헤딩이 청크로 새면 인덱스 노이즈 — 자손의 조상 경로로만 남아야 함
        chunks = chunk_file(DOCX)
        heads = {h for c in chunks for h in c.heading_path}
        assert not [c for c in chunks if c.text.strip() in heads]

    def test_docx_표는_자기_섹션에_남는다(self):
        # 문단·표를 따로 훑으면 표가 문서 끝으로 밀린다 — body 순서 보존 회귀 방지
        chunks = chunk_file(DOCX)
        table_chunks = [c for c in chunks if '출금일 | 재출금' in c.text]
        assert table_chunks, '표 청크를 못 찾음'
        assert any('결제수단별 출금일' in h for h in table_chunks[0].heading_path)

    def test_md는_heading_path_채움(self):
        chunks = chunk_file(MD)
        assert any(c.heading_path for c in chunks)         # md만 헤딩 구조 보존
        heads = [h for c in chunks for h in c.heading_path]
        assert any('대량주문' in h for h in heads)

    def test_chunk_index_순차(self):
        chunks = chunk_file(MD)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_긴_문서는_여러_청크로(self):
        assert len(chunk_file(MD)) > 1                     # 3,600자 문서 — 512자 분할


class TestMdSections:
    """md 섹션 파서 — pdf·docx와 같은 계약을 지키는지 (#42).

    기존 md 테스트 3개(heading_path 채움 / chunk_index 순차 / 분할)는 `any(...)`,
    `len(...) > 1`로 헐거워 청킹이 나빠져도 통과한다. 이 클래스가 이번 변경의
    실제 계약(헤딩 제외·빈 섹션 스킵·펜스 가드)을 못박는다.
    """

    def _write(self, tmp_path, text):
        p = tmp_path / 'doc.md'
        p.write_text(text, encoding='utf-8')
        return p

    def test_본문_없는_헤딩은_섹션이_아니다(self, tmp_path):
        # 옛 MarkdownNodeParser 경로는 '## 2. 봉제 불량' 한 줄짜리 청크를 만들었다
        # (실측 md 청크의 19%). docx는 진작 막고 있던 것을 md에도 맞춘 것.
        p = self._write(tmp_path, '# 제목\n\n## 2. 상위\n\n### 2.1 하위\n\n실제 본문이다.\n')
        bodies = [s.body for s in _md_sections(p)]
        assert bodies == ['실제 본문이다.']

    def test_헤딩은_본문에_안_들어간다(self, tmp_path):
        # heading_path가 들고 있고 index_text가 앞에 붙이므로 본문에 넣으면 중복
        p = self._write(tmp_path, '# 제목\n\n## 1. 절\n\n내용 한 줄.\n')
        section = _md_sections(p)[0]
        assert section.heading_path == ['제목', '1. 절']
        assert '#' not in section.body

    def test_빈_줄은_보존된다(self, tmp_path):
        # md는 빈 줄이 문단 구분 — 걷어내면 문단이 붙어버린다
        p = self._write(tmp_path, '# 제목\n\n첫 문단.\n\n둘째 문단.\n')
        assert _md_sections(p)[0].body == '첫 문단.\n\n둘째 문단.'

    def test_백틱_펜스_안의_샾은_헤딩이_아니다(self, tmp_path):
        p = self._write(tmp_path, '# 제목\n\n```bash\n# 이건 셸 주석이다\necho hi\n```\n끝.\n')
        sections = _md_sections(p)
        assert [s.heading_path for s in sections] == [['제목']]
        assert '# 이건 셸 주석이다' in sections[0].body

    def test_틸드_펜스도_가드한다(self, tmp_path):
        # 옛 MarkdownNodeParser는 백틱만 봤다 — 여기는 개선분이라 회귀로 잠근다
        p = self._write(tmp_path, '# 제목\n\n~~~python\n# 파이썬 주석\n~~~\n끝.\n')
        assert [s.heading_path for s in _md_sections(p)] == [['제목']]

    def test_펜스는_같은_문자로만_닫힌다(self, tmp_path):
        # ``` 안에서 ~~~를 만나도 안 닫혀야 그 뒤 '#'이 헤딩으로 새지 않는다
        p = self._write(tmp_path, '# 제목\n\n```\n~~~\n# 코드 안\n```\n\n실제 본문.\n')
        assert [s.heading_path for s in _md_sections(p)] == [['제목']]

    def test_들여쓰기_코드블록의_샾도_헤딩이_아니다(self, tmp_path):
        # _HEADING_RE가 열 0을 요구해 공짜로 걸러진다 — 그 사실을 고정
        p = self._write(tmp_path, '# 제목\n\n    # 들여쓴 코드\n\n본문.\n')
        assert [s.heading_path for s in _md_sections(p)] == [['제목']]

    def test_레벨_건너뛰기(self, tmp_path):
        # H1 → H3 (H2 생략) — pdf·docx와 같은 `del stack[level-1:]` 동작
        p = self._write(tmp_path, '# 제목\n\n### 건너뜀\n\n본문.\n')
        assert _md_sections(p)[0].heading_path == ['제목', '건너뜀']

    def test_page는_항상_None(self, tmp_path):
        p = self._write(tmp_path, '# 제목\n\n본문.\n')
        assert _md_sections(p)[0].page is None

    def test_빈_파일은_섹션_0개(self, tmp_path):
        assert _md_sections(self._write(tmp_path, '')) == []

    def test_헤딩만_있는_파일도_섹션_0개(self, tmp_path):
        assert _md_sections(self._write(tmp_path, '# 제목\n\n## 절\n')) == []


class TestChunkFileDispatch:
    """형식 분기가 chunk_file 한 곳뿐임을 고정 (#42 — 두 곳이던 게 xlsx 버그의 원인)."""

    def test_xlsx는_chunk_xlsx로_위임된다(self):
        # 옛 CLI 경로는 xlsx를 md로 취급해 ZIP 바이너리를 색인했다
        xlsx = CORPUS / 'homeplus' / 'homeplus_10_멤버십혜택표.xlsx'
        chunks = chunk_file(xlsx)
        assert all(c.meta and c.meta.get('is_table') for c in chunks)
        assert not any(c.text.startswith('PK') for c in chunks)

    def test_지원하지_않는_형식은_ValueError(self, tmp_path):
        p = tmp_path / 'x.bin'
        p.write_bytes(b'\x00\x01')
        with pytest.raises(ValueError, match='지원하지 않는 형식'):
            chunk_file(p)


class TestDocxFallbackHeadings:
    """Heading 스타일 없는 docx의 조문 패턴 승격 (#205). 튜플 단위 + python-docx 생성 fixture."""

    def test_스타일_헤딩이_하나라도_있으면_무변경(self):
        from rag.chunking import _docx_fallback_headings
        items = [(True, 1, '제목', False), (False, 0, '제1조(목적) 본문', False)]
        assert _docx_fallback_headings(items) == items

    def test_제목_장_조_3단(self):
        from rag.chunking import _docx_fallback_headings
        items = [(False, 0, '장애관리 지침서\n< 컨택센터 사업부문 >', False), (False, 0, '제 1 장 총칙', False),
                 (False, 0, '제1조(목적)', False), (False, 0, '이 지침은 …', False), (False, 0, '제2조(적용 범위)', False)]
        assert _docx_fallback_headings(items) == [
            (True, 1, '장애관리 지침서', False), (False, 0, '< 컨택센터 사업부문 >', False), (True, 2, '제 1 장 총칙', False),
            (True, 3, '제1조(목적)', False), (False, 0, '이 지침은 …', False), (True, 3, '제2조(적용 범위)', False)]

    def test_장이_없으면_조는_L2_제목_없으면_한_단계_위(self):
        from rag.chunking import _docx_fallback_headings
        with_title = [(False, 0, '지침서', False), (False, 0, '제1조(목적)', False), (False, 0, '본문', False)]
        assert [lv for h, lv, _, _ in _docx_fallback_headings(with_title) if h] == [1, 2]
        no_title = [(False, 0, '제1조(목적)', False), (False, 0, '본문', False)]
        assert [lv for h, lv, _, _ in _docx_fallback_headings(no_title) if h] == [1]

    def test_번호_목록과_표_행과_장애_같은_낱말은_승격하지_않는다(self):
        from rag.chunking import _docx_fallback_headings
        items = [(False, 0, '제목', False), (False, 0, '1. 출근시간 선택', False), (False, 0, '제3조(장애) | 설명', True),
                 (False, 0, '제3조의 장애 등급', False), (False, 0, '<표 1. 장애의 분류>', False)]
        assert [h for h, _, _, _ in _docx_fallback_headings(items)] == [True, False, False, False, False]

    def test_같은_문단에_이어진_본문은_헤딩에서_분리된다(self):
        from rag.chunking import _docx_fallback_headings
        items = [(False, 0, '제목', False), (False, 0, '제8조(역할과 책임) ① 운영관리자는 장애를 접수한다.\n② 복구는 담당자가 한다.', False)]
        assert _docx_fallback_headings(items)[1:] == [
            (True, 2, '제8조(역할과 책임)', False), (False, 0, '① 운영관리자는 장애를 접수한다.\n② 복구는 담당자가 한다.', False)]
        # 괄호 제목이 없는 장은 첫 줄 전체가 머리
        assert _docx_fallback_headings([(False, 0, '제 1 장 총칙\n이 장은 …', False)])[0] == (True, 1, '제 1 장 총칙', False)

    def test_가지_조문과_전각_괄호(self):
        from rag.chunking import _docx_fallback_headings
        items = [(False, 0, '제목', False), (False, 0, '제3조의2（특례）', False), (False, 0, '제1장(총칙)', False)]
        assert [(h, lv, txt) for h, lv, txt, _ in _docx_fallback_headings(items)] == [
            (True, 1, '제목'), (True, 2, '제3조의2（특례）'), (True, 2, '제1장(총칙)')]

    def test_첫_문단이_길면_제목이_없고_한_단계_위로(self):
        from rag.chunking import _docx_fallback_headings
        items = [(False, 0, '가' * 61, False), (False, 0, '제1조(목적)', False), (False, 0, '본문', False)]
        out = _docx_fallback_headings(items)
        assert out[0] == (False, 0, '가' * 61, False) and out[1] == (True, 1, '제1조(목적)', False)

    def test_Heading_스타일_문서는_폴백을_타지_않고_첨부_평문도_불변(self):
        from rag.chunking import _docx_fallback_headings, _docx_items
        from docx import Document as _Doc
        items = list(_docx_items(DOCX))                         # corpus 합성 docx — Heading 스타일 있음
        assert _docx_fallback_headings(items) is items
        d = _Doc(str(DOCX))
        expected = [p.text.strip() for p in d.paragraphs if p.text.strip()]
        assert all(line in _docx_text(DOCX) for line in expected)

    def test_생성한_docx_제목_장_조_3단_경로(self, tmp_path):
        from docx import Document as _Doc
        d = _Doc()
        d.add_paragraph('지침서'); d.add_paragraph('제 1 장 총칙'); d.add_paragraph('제1조(목적) 이 지침은 절차를 정한다.')
        p = tmp_path / 'three_levels.docx'; d.save(str(p))
        chunks = chunk_file(p)
        assert chunks[0].heading_path == ['지침서', '제 1 장 총칙', '제1조(목적)'] and '이 지침은 절차를 정한다.' in chunks[0].text

    def test_생성한_docx에서_조문별_heading_path가_채워지고_표_행이_한_청크에_남는다(self, tmp_path):
        from docx import Document as _Doc
        d = _Doc()
        d.add_paragraph('장애관리 지침서')
        d.add_paragraph('제1조(목적)'); d.add_paragraph('이 지침은 장애 대응 절차를 정한다.')
        d.add_paragraph('제8조(역할과 책임)')
        tbl = d.add_table(rows=2, cols=2)
        tbl.cell(0, 0).text = '역할'; tbl.cell(0, 1).text = '책임'
        tbl.cell(1, 0).text = '운영관리'; tbl.cell(1, 1).text = '장애 발생 시 1차 대응하며,\n대응 불가 시 복구 담당자를 지정함'
        p = tmp_path / 'no_heading_style.docx'; d.save(str(p))
        chunks = chunk_file(p)
        assert all(c.heading_path for c in chunks)
        assert any(c.heading_path[-1] == '제8조(역할과 책임)' and '1차 대응하며' in c.text and '복구 담당자를 지정함' in c.text for c in chunks)
