"""PDF 청킹 결함 6건 회귀 테스트 (#137).

실문서 21건을 색인하고 원본과 쪽 단위로 대조해 확정한 결함들이다. 여섯 개 전부
**기존 픽스처가 조건을 만들지 못해** 통과하고 있었다 — 합성 PDF가 단일 폰트·단일
크기·헤더푸터 없음·표 없음이었고, 실물 PDF를 쓰는 단정문은 `all(c.heading_path)`,
`len({c.page}) >= 2`처럼 존재만 확인해 값의 정확성을 보지 않았다.

그래서 여기 픽스처는 **결함 조건을 의도적으로 재현한다**: 인쇄 부산물, 폰트 혼용,
목록 번호, 빈 셀 격자표, 여러 쪽에 걸친 섹션.
"""
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rag.chunking import (_pdf_tables, _restore_leading_marker,
                          _strip_orphan_marker, chunk_file, pdf_image_area_ratio)

W, H = A4


def _print_furniture(c):
    """브라우저 '인쇄 → PDF'가 상·하단 여백에 찍는 것 — 실측 위치·형태를 그대로."""
    c.setFont('Helvetica', 8)
    c.drawString(20, H - 20, '26. 9. 8. 오후 10:25 [총무 | 가이드] 문서 제목')
    c.drawString(20, 18, 'https://echo.example.com/egate50/kr/eip/home/home.nsf/win_print?readform 1/1')


class TestFurnitureVote:
    """결함 1 — 인쇄 부산물이 최빈 글자 크기 투표를 빼앗으면 본문이 통째로 소멸한다."""

    def _짧은_인쇄본(self, tmp_path):
        # 부산물(8pt) 글자 수가 본문(9.7pt)을 이기도록 — 실측 최악 사례는 190 대 177이었다
        p = tmp_path / 'printed.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        _print_furniture(c)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(50, H - 80, 'Software License Guide')
        c.setFont('Helvetica', 10)
        c.drawString(50, H - 120, 'Apply through the ECHO approval form.')
        c.drawString(50, H - 140, 'Unused licenses must be returned.')
        c.save()
        return p

    def test_본문이_헤딩으로_승격되지_않는다(self, tmp_path):
        chunks = chunk_file(self._짧은_인쇄본(tmp_path))
        body = '\n'.join(ch.text for ch in chunks)
        assert 'ECHO approval form' in body
        assert 'Unused licenses' in body

    def test_인쇄_부산물은_본문에_남지_않는다(self, tmp_path):
        chunks = chunk_file(self._짧은_인쇄본(tmp_path))
        joined = '\n'.join(ch.text for ch in chunks) + '\n'.join(
            ' '.join(ch.heading_path) for ch in chunks)
        assert 'https://' not in joined          # URL 푸터
        assert '오후 10:25' not in joined         # 인쇄 시각 헤더


class TestSymbolHeading:
    """결함 2 — 기호 조각이 같은 레벨 헤딩이 되어 진짜 제목을 덮어썼다."""

    def test_기호만_있는_줄은_제목을_덮지_않는다(self, tmp_path):
        # 실문서에선 한글과 대괄호가 다른 폰트라 베이스라인이 3pt 어긋나 줄이 갈렸다.
        # 여기선 그 결과(같은 크기의 기호 전용 줄)를 직접 만든다.
        p = tmp_path / 'symbol.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.setFont('Helvetica-Bold', 13)
        c.drawString(50, H - 80, 'Real Document Title')
        c.drawString(50, H - 100, '[ | ]')
        c.setFont('Helvetica', 10)
        for i, y in enumerate(range(140, 260, 20)):
            c.drawString(50, H - y, f'Body sentence number {i} with enough text to weigh.')
        c.save()
        chunks = chunk_file(p)
        heads = [h for ch in chunks for h in ch.heading_path]
        assert 'Real Document Title' in heads
        assert '[ | ]' not in heads
        assert '[ | ]' not in '\n'.join(ch.text for ch in chunks)   # 본문으로도 새지 않는다


class TestOrphanMarker:
    """결함 3 — SentenceSplitter가 '3. 항목'을 '3.' / '항목'으로 끊어 번호만 남았다."""

    def test_꼬리_번호는_떼어낸다(self):
        assert _strip_orphan_marker('앞 내용이 이어지다가\n3.') == '앞 내용이 이어지다가'
        assert _strip_orphan_marker('본문\n(2)') == '본문'

    def test_문장_끝_마침표는_건드리지_않는다(self):
        assert _strip_orphan_marker('정상적인 문장이다.') == '정상적인 문장이다.'

    def test_번호_바로_뒤에서_시작하면_되살린다(self):
        # 실문서는 번호와 항목이 같은 줄이다 — 스플리터가 그 사이를 끊는다
        body = '앞 내용\n3. 숙박료의 실비정산 기준'
        offset = body.index('숙박료의')
        assert _restore_leading_marker(body, offset, '숙박료의 실비정산 기준') \
            == '3. 숙박료의 실비정산 기준'

    def test_번호가_독립_줄이어도_되살린다(self):
        body = '앞 내용\n3.\n숙박료의 실비정산 기준'
        offset = body.index('숙박료의')
        assert _restore_leading_marker(body, offset, '숙박료의 실비정산 기준') \
            == '3. 숙박료의 실비정산 기준'

    def test_overlap으로_이미_갖고_있으면_주입하지_않는다(self):
        # 초기 구현이 '다음 청크가 그 번호로 시작하지 않으면 주입'으로 판정했다가
        # overlap 안쪽에 번호가 이미 있는 경우를 놓쳐 중복 주입했다 (txt 코퍼스 6건).
        body = '앞 내용\n[상황 2] 배송 지연 항의\n\n1. "배송이 늦어져 죄송합니다."'
        offset = body.index('[상황 2]')
        chunk = body[offset:]
        assert _restore_leading_marker(body, offset, chunk) == chunk   # 그대로

    def test_번호가_아닌_것_뒤에서는_주입하지_않는다(self):
        body = '문장이 끝났다.\n다음 문장'
        offset = body.index('다음')
        assert _restore_leading_marker(body, offset, '다음 문장') == '다음 문장'


class TestTableStructure:
    """결함 5 — 빈 셀이 사라져 표시의 열 위치가 소실됐다(유실 0인데 의미 반전)."""

    def _격자표(self, tmp_path, rows):
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors
        p = tmp_path / 'grid.pdf'
        table = Table(rows)
        table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
        SimpleDocTemplate(str(p), pagesize=A4).build([table])
        return p

    def test_빈_셀이_자리를_지킨다(self, tmp_path):
        p = self._격자표(tmp_path, [
            ['Part', 'Free', 'Company', 'Personal'],
            ['Board', '2y', 'O', ''],
            ['Water', 'none', '', 'O'],
        ])
        body = '\n'.join(ch.text for ch in chunk_file(p))
        assert '| Board | 2y | O |  |' in body
        assert '| Water | none |  | O |' in body     # 빈 셀을 지우면 두 줄이 같아진다

    def test_산문은_표로_바뀌지_않는다(self, tmp_path):
        # 조항형 규정의 본문 테두리가 격자로 오인돼 산문이 표로 둔갑했다
        # (실측: 노사협의회 규정 19줄이 `| 문장 |  |  |` 꼴이 되고 조항 번호가 잘렸다)
        p = self._격자표(tmp_path, [
            ['제 5조 (구성) 협의회는 노사를 대표하는 위원 각 4명으로 구성한다.', '', ''],
            ['① 사용자를 대표하는 위원은 대표이사와 대표이사가 위촉하는 자로 한다.', '', ''],
            ['② 근로자를 대표하는 위원은 근로자가 선출한다.', '', ''],
        ])
        import pdfplumber
        with pdfplumber.open(p) as pdf:
            assert _pdf_tables(pdf.pages[0]) == []      # 표로 인정하지 않는다


class TestChunkPage:
    """결함 6 — 한 섹션의 모든 청크가 섹션 시작 페이지를 물려받았다(실측 49% 오류)."""

    def test_여러_쪽에_걸친_섹션은_청크마다_제_쪽을_단다(self, tmp_path):
        p = tmp_path / 'multipage.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.setFont('Helvetica-Bold', 16)
        c.drawString(50, H - 60, 'Single Heading For Whole Document')
        c.setFont('Helvetica', 10)
        for page_no in range(3):
            if page_no:
                c.setFont('Helvetica', 10)
            for i in range(30):
                c.drawString(50, H - 100 - i * 20,
                             f'Page {page_no + 1} line {i} with enough words to fill the chunk budget.')
            c.showPage()
        c.save()
        chunks = chunk_file(p)
        assert len({ch.page for ch in chunks}) >= 2      # 한 쪽으로 뭉개지지 않는다
        for ch in chunks:
            first = ch.text.strip().split('\n')[0]
            if first.startswith('Page '):
                assert ch.page == int(first.split()[1])  # 청크가 실제로 있는 쪽을 가리킨다


class TestImageRatio:
    """결함 4 — 이미지로 렌더된 도표는 텍스트가 0인데 문서는 ready가 된다."""

    def test_이미지_없는_문서는_0이다(self, tmp_path):
        p = tmp_path / 'text.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.drawString(50, H - 60, 'text only')
        c.save()
        assert pdf_image_area_ratio(p) == 0.0

    def test_이미지가_있으면_면적_비율을_돌려준다(self, tmp_path):
        from reportlab.lib.utils import ImageReader
        try:
            from PIL import Image
        except ImportError:
            pytest.skip('Pillow 없음 — 이미지 픽스처를 만들 수 없다')
        img = tmp_path / 'box.png'
        Image.new('RGB', (200, 200), 'white').save(img)
        p = tmp_path / 'withimage.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.drawImage(ImageReader(str(img)), 50, H - 300, width=300, height=200)
        c.drawString(50, 60, 'caption text outside the image')
        c.save()
        assert pdf_image_area_ratio(p) > 5.0

    def test_pdf가_아니면_0이다(self, tmp_path):
        f = tmp_path / 'a.txt'
        f.write_text('hello')
        assert pdf_image_area_ratio(f) == 0.0
