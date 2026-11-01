"""XLSX docling 경로 — 생성한 Excel 파일을 실제로 변환해 시트·병합·그림 캡션·크기 상한을 고정한다.

Excel은 SimplePipeline이라 레이아웃 모델이 없어 변환이 빠르다. 그림 캡션은 로컬 VLM 대역(conftest.fake_vlm)이
고정 문장을 줘 외부 서비스 없이 결정적이다.
"""

from io import BytesIO

import openpyxl
import pytest

from config import settings
from rag.chunking import PictureLimitError, chunk_file
from tests.conftest import FAKE_CAPTION

pytest.importorskip('docling')


def _png(size=(120, 80)) -> bytes:
    # 1×1은 안 된다 — docling Word 백엔드가 아주 작은 그림을 여백(ContentLayer.INVISIBLE)으로 분류해 iterate_items에서 빠진다
    from PIL import Image

    buf = BytesIO()
    Image.new('RGB', size, (200, 30, 30)).save(buf, format='PNG')
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _vlm(fake_vlm, monkeypatch):
    """캡션은 로컬 대역이 준다. 설정이 바뀌면 변환 프로세스가 스냅숏 불일치로 재기동하므로 싱글턴을 비울 필요가 없다."""
    return fake_vlm


def _xlsx(tmp_path, build, name='t.xlsx'):
    wb = openpyxl.Workbook()
    build(wb)
    p = tmp_path / name
    wb.save(p)
    return p


class TestXlsxDocling:
    def test_표_위_병합_제목은_한_번_세로_병합_분류는_행마다(self, tmp_path):
        def build(wb):
            ws = wb.active
            ws.title = '양식'
            ws.append(['업무 교통비 청구서', None, None])
            ws.merge_cells('A1:C1')
            ws.append(['구분', '항목', '금액'])
            ws.append(['교통', '버스', 1500])
            ws.append([None, '택시', 29100])
            ws.merge_cells('A3:A4')

        chunks = chunk_file(_xlsx(tmp_path, build), description='교통비 양식')
        assert {c.heading_path[0] for c in chunks} == {'양식'}  # 묶음이 여럿이어도 한 시트
        text = chunks[0].meta['table_text']  # 시트 완성본 — 모델이 보는 본문
        # 표 폭 전체에 걸친 첫 행 제목은 docling이 표 밖 글로 뺀다 — 가로 병합 반복과 무관하게 한 번
        assert text.count('업무 교통비 청구서') == 1 and '\n업무 교통비 청구서\n' in text
        assert '| 교통 | 버스 | 1500 |' in text and '| 교통 | 택시 | 29100 |' in text
        assert text.startswith('[교통비 양식]\n') and text.count('[교통비 양식]') == 1
        assert all(c.text.startswith('[교통비 양식]\n') for c in chunks)  # 묶음마다 설명 반복(검색 보강)
        assert all(c.meta['is_table'] and c.meta['table_key'] == '양식' for c in chunks)

    def test_시트_위_그림은_그_시트_청크에_들어간다(self, tmp_path):
        from openpyxl.drawing.image import Image as XlImage

        def build(wb):
            ws = wb.active
            ws.title = '첫시트'
            ws.append(['등급', '적립률'])
            ws.append(['VIP', '2.0'])
            ws.add_image(XlImage(BytesIO(_png())), 'D2')
            ws2 = wb.create_sheet('둘째')
            ws2.append(['항목', '값'])
            ws2.append(['x', '1'])

        chunks = chunk_file(_xlsx(tmp_path, build))
        by_sheet = {c.heading_path[0]: c.meta['table_text'] for c in chunks}
        assert list(by_sheet) == ['첫시트', '둘째']  # 시트 순서 유지(시트당 묶음 여럿 가능)
        assert FAKE_CAPTION in by_sheet['첫시트'] and FAKE_CAPTION not in by_sheet['둘째']

    def test_긴_시트의_그림도_캡션이_붙는다(self, tmp_path):
        # docling은 시트를 칸 단위로 재고 OneCellAnchor 그림을 1×1칸으로 잡아, 칸이 100개만 넘어도 면적 비율이 하한(1%)
        # 아래로 떨어져 큰 그림도 캡션 없이 자리표시로 남는다. XLSX는 하한 0으로 전부 캡션한다
        from openpyxl.drawing.image import Image as XlImage

        from rag.chunking.docling_elements import PICTURE_PLACEHOLDER

        def build(wb):
            ws = wb.active
            ws.title = '긴시트'
            ws.append(['번호', '품목', '단가'])
            for i in range(1, 121):  # 3열 × 121행 = 363칸 → 1칸 그림은 0.28%
                ws.append([i, f'품목-{i:03d}', i * 100])
            ws.add_image(XlImage(BytesIO(_png())), 'E60')

        chunks = chunk_file(_xlsx(tmp_path, build), description='긴 시트')
        text = chunks[0].meta['table_text']
        assert FAKE_CAPTION in text and PICTURE_PLACEHOLDER not in text

    def test_XLSX_그림도_장수_상한에_걸린다(self, tmp_path, monkeypatch):
        # 하한 0이라 엑셀 그림은 전부 캡션 대상이고, 그림 장수 상한도 전부를 센다
        from openpyxl.drawing.image import Image as XlImage

        monkeypatch.setattr(settings, 'docling_max_pictures', 2)

        def build(wb):
            ws = wb.active
            ws.append(['구분', '값'])
            for i in range(60):
                ws.append([f'항목{i}', i])
            for cell in ('D2', 'D20', 'D40'):
                ws.add_image(XlImage(BytesIO(_png())), cell)

        with pytest.raises(PictureLimitError, match='그림 3장'):
            chunk_file(_xlsx(tmp_path, build), description='그림 많은 시트')

    def test_숨긴_시트는_색인하지_않고_행_상한에도_세지_않는다(self, tmp_path):
        # 숨긴 시트는 건너뛴다(openpyxl은 세고 docling은 비워 내서 문서 전체가 실패하지 않게)
        from rag.chunking.xlsx import validate_xlsx_upload

        def build(wb):
            wb.active.title = '보임1'
            wb.active.append(['col'])
            wb.active.append(['v1'])
            hidden = wb.create_sheet('참조')
            hidden.append(['k'])
            for i in range(5000):  # 크기 상한 초과 — 숨겼으니 거절 사유가 아니다
                hidden.append([i])
            hidden.sheet_state = 'hidden'
            very = wb.create_sheet('내부')
            very.append(['x'])
            very.append(['y'])
            very.sheet_state = 'veryHidden'
            wb.create_sheet('보임2').append(['col'])
            wb['보임2'].append(['v2'])

        p = _xlsx(tmp_path, build)
        assert list(dict.fromkeys(c.heading_path[0] for c in chunk_file(p))) == ['보임1', '보임2']
        validate_xlsx_upload(p.read_bytes(), '설명')  # 예외 없음

    def test_공백만_든_시트는_빈_시트로_건너뛴다(self, tmp_path):
        # docling은 공백 칸을 내용으로 보지 않는다 — openpyxl만 세면 '시트를 비워 냈다'로 문서 전체가 실패한다
        def build(wb):
            wb.active.append(['col'])
            wb.active.append(['v'])
            blank = wb.create_sheet('공백')
            blank['A1'] = ' '
            blank['B2'] = ''

        assert len(chunk_file(_xlsx(tmp_path, build))) == 1

    def test_셀_없이_그림만_있는_시트와_차트_시트는_보이는_시트_뒤에_붙는다(self, tmp_path):
        # openpyxl은 셀이 없는 시트·차트 시트를 세지 않는다(행 상한 대상 아님). docling이 내용(그림·차트)을 내면 뒤에 붙인다
        from openpyxl.chart import BarChart, Reference
        from openpyxl.drawing.image import Image as XlImage

        def build(wb):
            ws = wb.active
            ws.title = '데이터'
            for row in (['k', 'v'], ['a', 1], ['b', 2]):
                ws.append(row)
            wb.create_sheet('그림만').add_image(XlImage(BytesIO(_png())), 'B2')
            chart = BarChart()
            chart.add_data(Reference(ws, min_col=2, min_row=1, max_row=3), titles_from_data=True)
            wb.create_chartsheet('차트').add_chart(chart)

        chunks = chunk_file(_xlsx(tmp_path, build))
        assert chunks[0].heading_path == ['데이터']
        assert {c.heading_path[0] for c in chunks[1:]} == {'그림만', '차트'}
        assert all(c.meta['is_table'] and c.meta['table_key'] == c.heading_path[0] for c in chunks)

    def test_빈_엑셀은_청크_0개(self, tmp_path):
        assert chunk_file(_xlsx(tmp_path, lambda wb: None)) == []

    def test_docling이_셀_있는_시트를_비워_내면_실패한다(self, tmp_path, monkeypatch):
        # 조용히 시트를 빠뜨린 채 ready가 되지 않게 — PDF 경로의 불완전 결과 가드와 같은 규율
        import rag.chunking.xlsx as xc

        def build(wb):
            wb.active.append(['col'])
            wb.active.append(['1'])

        monkeypatch.setattr(xc, '_docling_sheet_blocks', lambda doc: {})
        with pytest.raises(RuntimeError, match='비워 냈다'):
            chunk_file(_xlsx(tmp_path, build))

    def test_크기_상한은_docling_변환_전에_openpyxl로_건다(self, tmp_path, monkeypatch):
        from rag.chunking import convert_process
        from rag.chunking.xlsx import XlsxTableTooLarge

        def build(wb):
            wb.active.append(['col'])
            for i in range(5000):  # 추정 ≈ 1만 토큰 이상 — 상한(6,000) 초과
                wb.active.append([i])

        def boom(*a, **k):
            raise AssertionError('크기 상한 위반인데 docling 변환까지 갔다')

        monkeypatch.setattr(convert_process, 'docling_convert', boom)
        with pytest.raises(XlsxTableTooLarge):
            chunk_file(_xlsx(tmp_path, build))
