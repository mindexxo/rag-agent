"""청킹 파서 단위 테스트 — _pdf_pages / _docx_text / chunk_file (비ML 파서 리팩토링 검증).

입력은 corpus_v2 합성 문서 실물. 새 스펙: heading_path는 md만 채워진다.
"""
from pathlib import Path

from rag.chunking import _docx_text, _extract_heading_path, _pdf_pages, chunk_file, chunk_txt

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


class TestExtractHeadingPath:
    """조상 경로(metadata) × 자기 헤딩(text 첫 줄) 4조합 — 간접 테스트로는 두 로직이 구분 안 됨."""

    def test_조상과_자기헤딩_결합(self):
        meta = {'header_path': '/1. 상위/1.2 중위/'}
        assert _extract_heading_path(meta, '### 1.2.1 하위\n본문') == ['1. 상위', '1.2 중위', '1.2.1 하위']

    def test_조상만(self):
        meta = {'header_path': '/1. 상위/1.2 중위/'}
        assert _extract_heading_path(meta, '헤딩 아닌 본문') == ['1. 상위', '1.2 중위']

    def test_자기헤딩만(self):
        assert _extract_heading_path({'header_path': '/'}, '# 제목\n본문') == ['제목']

    def test_선행_공백_후_헤딩(self):
        # lstrip 경로 — 노드 텍스트가 개행으로 시작해도 첫 줄 헤딩을 잡아야 함
        assert _extract_heading_path({'header_path': '/'}, '\n\n# 제목\n본문') == ['제목']

    def test_둘_다_없음(self):
        assert _extract_heading_path({}, '그냥 본문') == []


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
