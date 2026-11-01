"""DOCX docling 경로 — 생성한 Word 파일을 실제로 변환해 제목 층·표·그림 캡션·중첩·긴 표를 고정한다.

Word는 SimplePipeline이라 레이아웃 모델이 없어 변환이 빠르다. 그림 캡션은 로컬 VLM 대역(conftest.fake_vlm)이
고정 문장을 줘 외부 서비스 없이 결정적이다. 캡션 실패는 문서 재시도(CaptionFailedError)라 따로 단언한다.
"""

from io import BytesIO

import pytest

from config import settings
from rag.chunking import (
    CaptionFailedError,
    PictureLimitError,
    chunk_file,
    count_picture_placeholders,
)
from rag.chunking.docx import docx_sections
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


def _docx(tmp_path, build, name='t.docx'):
    from docx import Document

    doc = Document()
    build(doc)
    p = tmp_path / name
    doc.save(p)
    return p


class TestDocxDocling:
    def test_헤딩_스타일_레벨이_그대로_섹션_층이_된다(self, tmp_path):
        # Heading 스타일 문서는 docling 레벨 = 스타일 레벨이다. PDF 매핑의
        # 제목/장 휴리스틱을 쓰면 3단이 2단으로 눌린다 — 이 테스트가 그 회귀를 잡는다.
        def build(doc):
            doc.add_heading('배송 정책', level=1)
            doc.add_heading('1. 배송 유형', level=2)
            doc.add_heading('1.1 일반 배송', level=3)
            doc.add_paragraph('일반 배송은 결제 후 2영업일 이내 출고한다.')
            doc.add_heading('1.2 새벽 배송', level=3)
            doc.add_paragraph('새벽 배송은 전날 23시까지 주문분이다.')

        p = _docx(tmp_path, build)
        new = [
            s.heading_path for s in docx_sections(p)
        ]  # 섹션 단위로 본다 — 청크는 형제 섹션을 묶어 공통 조상만 남긴다
        assert new == [['배송 정책', '1. 배송 유형', '1.1 일반 배송'], ['배송 정책', '1. 배송 유형', '1.2 새벽 배송']]

    def test_표_셀은_한_번만_들어가고_굵게_표시가_없다(self, tmp_path):
        # docling은 서식 있는 셀의 내용을 표와 별도 텍스트로 한 번 더 내놓는다 — 건너뛰지 않으면 셀이 두 번 색인된다.
        # export_to_dataframe은 굵은 셀을 '**…**'로 내보낸다 — 셀 원문으로 그려야 없다.
        # 중복은 **서식 있는 칸에서만** 생긴다 — 그래서 세는 문구를 굵은 칸에 둔다(평범한 칸으로 세면 건너뛰기를 꺼도 통과한다)
        def build(doc):
            doc.add_heading('요금표', level=1)
            t = doc.add_table(rows=2, cols=2)
            t.cell(0, 0).paragraphs[0].add_run('지역').bold = True
            t.cell(0, 1).text = '요금'
            t.cell(1, 0).paragraphs[0].add_run('제주고유지역').bold = True
            t.cell(1, 1).text = '5000원'

        chunks = chunk_file(_docx(tmp_path, build))
        text = '\n'.join(c.text for c in chunks)
        assert text.count('제주고유지역') == 1
        assert '**' not in text
        assert '| 지역 | 요금 |' in text

    def test_표_셀_안의_중첩_표를_읽는다(self, tmp_path):
        # 셀 안의 표도 읽어야 한다(python-docx로는 못 읽던 부분)
        def build(doc):
            doc.add_heading('인사규정', level=1)
            outer = doc.add_table(rows=1, cols=1)
            inner = outer.cell(0, 0).add_table(rows=1, cols=2)
            inner.cell(0, 0).text = '중첩셀고유문구'
            inner.cell(0, 1).text = 'CL2'

        assert '중첩셀고유문구' in '\n'.join(c.text for c in chunk_file(_docx(tmp_path, build)))

    def test_스타일_없는_문서는_조문_폴백이_docling_경로에서도_돈다(self, tmp_path):
        # Heading 스타일 0개인 문서. docling도 헤딩을 0개 내놓으므로 폴백이 같은 층을 만들어야 한다.
        def build(doc):
            doc.add_paragraph('장애관리 지침서')
            doc.add_paragraph('제1장 총칙')
            doc.add_paragraph('제1조(목적) 이 지침은 장애 대응 절차를 정한다.')
            doc.add_paragraph('제2조(적용) 전 시스템에 적용한다.')

        assert [s.heading_path for s in docx_sections(_docx(tmp_path, build))] == [
            ['장애관리 지침서', '제1장 총칙', '제1조(목적)'],
            ['장애관리 지침서', '제1장 총칙', '제2조(적용)'],
        ]

    def test_표_칸_안의_그림은_그_칸에_설명이_붙는다(self, tmp_path):
        # 표 안 요소를 건너뛰는 규칙에 칸 안 그림까지 걸리면 캡션이 자리표시도 없이 버려진다 — 그 칸에 붙어야 한다.
        def build(doc):
            doc.add_heading('매뉴얼', level=1)
            t = doc.add_table(rows=1, cols=2)
            t.cell(0, 0).text = '1. 로그인 화면'
            t.cell(0, 1).paragraphs[0].add_run().add_picture(BytesIO(_png()))

        chunks = chunk_file(_docx(tmp_path, build))
        assert f'| 1. 로그인 화면 | {FAKE_CAPTION} |' in chunks[0].text
        assert count_picture_placeholders(chunks) == 0

    def test_빈_워드는_청크_0개(self, tmp_path):
        # 호출부(index_pending_document)가 '추출된 텍스트가 없습니다'로 failed 처리한다 — 여기선 예외 없이 빈 목록
        assert chunk_file(_docx(tmp_path, lambda doc: None)) == []

    def test_그림_캡션이_실패하면_문서째_일시_실패한다(self, tmp_path, fake_vlm):
        # docling은 VLM 오류를 삼키고 빈 캡션을 준다. 그대로 두면 그림 내용이 빠진 문서가 ready가 된다.
        # CaptionFailedError(ConnectionError 하위)라 outbox가 일시로 분류해 재시도한다(test_outbox_pure).
        fake_vlm.status = 500

        def build(doc):
            doc.add_paragraph('제1조(목적) 본문.')
            doc.add_picture(BytesIO(_png()))
            doc.add_paragraph('제2조(범위) 본문.')

        with pytest.raises(CaptionFailedError):
            chunk_file(_docx(tmp_path, build))

    def test_VLM이_응답하지_않으면_캡션_타임아웃으로_일시_실패한다(self, tmp_path, monkeypatch, fake_vlm):
        # 장애 주입 — 서버가 살아 있지만 멈춘 경우. docling은 읽기 타임아웃을 재시도하지 않고 빈 캡션을 준다
        fake_vlm.delay = 5
        monkeypatch.setattr(settings, 'vlm_caption_timeout_seconds', 1.0)

        def build(doc):
            doc.add_picture(BytesIO(_png()))

        with pytest.raises(CaptionFailedError):
            chunk_file(_docx(tmp_path, build))

    def test_캡션이_되면_그림_자리에_캡션이_들어간다(self, tmp_path):
        def build(doc):
            doc.add_paragraph('제1조(목적) 본문.')
            doc.add_picture(BytesIO(_png()))

        chunks = chunk_file(_docx(tmp_path, build))
        assert FAKE_CAPTION in chunks[0].text and count_picture_placeholders(chunks) == 0

    def test_그림이_상한을_넘으면_VLM을_부르기_전에_결정적_실패한다(self, tmp_path, monkeypatch, fake_vlm):
        # 앞 N장만 캡션하면 조용한 불완전 색인이라 문서를 거부한다.
        # 셈은 캡션 단계의 첫 배치에서 한 번 — 넘으면 캡션을 하나도 요청하지 않는다(넘는 문서에 VLM 시간을 안 쓴다).
        monkeypatch.setattr(settings, 'docling_max_pictures', 2)

        def build(doc):
            for _ in range(3):
                doc.add_picture(BytesIO(_png()))

        with pytest.raises(PictureLimitError, match='그림 3장'):
            chunk_file(_docx(tmp_path, build))
        assert fake_vlm.calls == 0

    def test_상한과_같은_장수는_통과한다(self, tmp_path, monkeypatch, fake_vlm):
        monkeypatch.setattr(settings, 'docling_max_pictures', 2)

        def build(doc):
            for _ in range(2):
                doc.add_picture(BytesIO(_png()))

        assert chunk_file(_docx(tmp_path, build))
        assert fake_vlm.calls == 2

    def test_조문처럼_시작하는_그림_캡션은_헤딩이_되지_않는다(self, tmp_path, monkeypatch):
        # 그림은 from_table=True로 헤딩 폴백 승격에서 빠진다. 자리표시는 원래 조문 패턴에 안 걸리므로 캡션을 바꿔 끼워 본다
        import rag.chunking as ch

        monkeypatch.setattr(
            ch.docling_elements, 'docling_picture_text', lambda element: '제9조(그림 속 조문) 캡처 내용'
        )

        def build(doc):
            doc.add_paragraph('제1조(목적) 본문.')
            doc.add_picture(BytesIO(_png()))

        sections = docx_sections(_docx(tmp_path, build))
        assert all('제9조(그림 속 조문)' not in ' '.join(sec.heading_path) for sec in sections)
        assert any('제9조(그림 속 조문) 캡처 내용' in sec.body for sec in sections)


class TestNestedAndLong:
    def test_표_안의_표는_평탄화돼_바깥_글자가_안쪽_행마다_붙는다(self, tmp_path):
        # 안쪽 표가 통째로 사라지지 않아야 한다(청크에 `평가 항목 | (빈칸)`). PDF(TableFormer)와 같은 5×3 모양
        def build(doc):
            t = doc.add_table(rows=3, cols=2)
            t.cell(0, 0).text, t.cell(0, 1).text = '구분', '세부'
            t.cell(1, 0).text = '평가 항목'
            inner = t.cell(1, 1).add_table(rows=3, cols=2)
            for r, (a, b) in enumerate((('항목', '배점'), ('업적', '60'), ('역량', '40'))):
                inner.cell(r, 0).text, inner.cell(r, 1).text = a, b
            t.cell(2, 0).text, t.cell(2, 1).text = '평가 시기', '매년 12월'

        chunks = chunk_file(_docx(tmp_path, build))
        text = '\n'.join(c.text for c in chunks)
        assert (
            '| 평가 항목 | 항목 | 배점 |' in text
            and '| 평가 항목 | 업적 | 60 |' in text
            and '| 평가 항목 | 역량 | 40 |' in text
        )
        assert '| 평가 시기 | 매년 12월 |' in text

    def test_긴_docx_표는_행_묶음으로_나뉘고_묶음마다_헤더가_있다(self, tmp_path):
        def build(doc):
            t = doc.add_table(rows=46, cols=3)
            for c, h in enumerate(('번호', '품목', '단가')):
                t.cell(0, c).text = h
            for i in range(1, 46):
                t.cell(i, 0).text, t.cell(i, 1).text, t.cell(i, 2).text = (
                    str(i),
                    f'품목-{i:03d}',
                    f'{i * 1234 % 97000 + 1000:,}원',
                )

        chunks = chunk_file(_docx(tmp_path, build))
        assert len(chunks) >= 2
        assert all(c.text.startswith('| 번호 | 품목 | 단가 |') for c in chunks)  # 묶음마다 헤더
        rows = [ln for c in chunks for ln in c.text.split('\n')[2:]]
        assert rows == [f'| {i} | 품목-{i:03d} | {i * 1234 % 97000 + 1000:,}원 |' for i in range(1, 46)]  # 행 절단 없음

    def test_셀_안_줄바꿈은_공백으로_접혀_행이_깨지지_않는다(self, tmp_path):
        def build(doc):
            t = doc.add_table(rows=2, cols=3)
            t.cell(0, 0).text, t.cell(0, 1).text, t.cell(0, 2).text = '단계', '처리 내용', '담당'
            t.cell(1, 0).text = '접수'
            t.cell(1, 1).text = '1. 요청서 수신'
            t.cell(1, 1).add_paragraph('2. 담당자 배정')
            t.cell(1, 2).text = '총무팀'

        chunks = chunk_file(_docx(tmp_path, build))
        assert '| 접수 | 1. 요청서 수신 2. 담당자 배정 | 총무팀 |' in chunks[0].text

    def test_중첩_칸에_적힌_글자는_첫_행으로_남고_3단_중첩은_한_번만(self, tmp_path):
        # 칸의 라벨 글자가 사라지지 않고, 깊은 중첩이 두 번 들어가지 않는다
        def build(doc):
            t = doc.add_table(rows=1, cols=2)
            t.cell(0, 0).text = '구성'
            t.cell(0, 1).text = '세부 라벨'
            mid = t.cell(0, 1).add_table(rows=2, cols=2)
            mid.cell(0, 0).text, mid.cell(0, 1).text = '항목', '값'
            mid.cell(1, 0).text = '깊은'
            deep = mid.cell(1, 1).add_table(rows=1, cols=2)
            deep.cell(0, 0).text, deep.cell(0, 1).text = '최하', '99'

        text = '\n'.join(c.text for c in chunk_file(_docx(tmp_path, build)))
        assert '| 구성 | 세부 라벨 |' in text  # 칸의 라벨
        assert text.count('99') == 1 and '| 구성 | 깊은 | 최하 | 99 |' in text  # 3단 값은 한 번, 바깥 글자 반복
