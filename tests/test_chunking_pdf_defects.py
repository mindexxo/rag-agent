"""PDF 청킹 결함 회귀 테스트 (#137) — 결함 3·4, 파서 무관 후단(순수 함수·pdfplumber 직접 호출, docling 불필요).

결함 1·2·5(인쇄 부산물·기호 제목·표 빈 셀)는 #143에서 pdfplumber 휴리스틱을 걷어내고 docling이
구조적으로 처리하므로 tests/test_chunking_docling.py로 옮겼다. 결함 6(청크마다 실제 쪽)도 같은 파일의
실변환 테스트가 같은 합성 PDF로 검사한다 — 여기 있던 pdfplumber 비상 경로 판은 #272에서 지웠다.

실문서 21건을 색인하고 원본과 쪽 단위로 대조해 확정한 결함들이다. 당시 여섯 개 전부
**기존 픽스처가 조건을 만들지 못해** 통과하고 있었다 — 합성 PDF가 단일 폰트·단일
크기·헤더푸터 없음·표 없음이었고, 실물 PDF를 쓰는 단정문은 `all(c.heading_path)`,
`len({c.page}) >= 2`처럼 존재만 확인해 값의 정확성을 보지 않았다.

그래서 픽스처는 **결함 조건을 의도적으로 재현한다** — 여기 남은 결함 3·4는 목록 번호와 이미지 쪽.
"""
from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from rag.chunking import _restore_leading_marker, _strip_orphan_marker, pdf_image_area_ratio

W, H = A4


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
