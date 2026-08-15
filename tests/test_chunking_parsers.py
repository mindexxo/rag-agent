"""청킹 파서 단위 테스트 — _pdf_pages / _docx_text / chunk_file (비ML 파서 리팩토링 검증).

입력은 corpus_v2 합성 문서 실물. heading_path는 pdf·docx·md 모두 채워진다
(txt만 헤딩 개념이 없어 빈다).
"""
from pathlib import Path

import pytest

from rag.chunking import _docx_text, _md_sections, _pdf_pages, chunk_file, chunk_txt

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
        chunks = chunk_txt(p)
        assert len(chunks) == 1
        assert chunks[0].text == '첫 문장입니다. 둘째 문장입니다.'
        assert chunks[0].heading_path == [] and chunks[0].page is None

    def test_긴_텍스트는_분할되고_index_순차(self, tmp_path):
        p = tmp_path / 'b.txt'
        p.write_text('문장입니다. ' * 200, encoding='utf-8')   # 512자 초과
        chunks = chunk_txt(p)
        assert len(chunks) > 1
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_빈_파일은_청크_없음(self, tmp_path):
        # split_text('')==[''] 함정 — 빈 청크가 임베딩·DB로 새지 않아야 함 (PDF 가드와 동일)
        p = tmp_path / 'c.txt'
        p.write_text('   \n  ', encoding='utf-8')
        assert chunk_txt(p) == []


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
