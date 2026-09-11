"""docling PDF 경로 (#143) — 매핑 규칙은 docling 없이, 실변환은 설치 시에만.

두 층으로 나눈다:
  - `_sections_from_docling_elements` 단위 테스트: 튜플만 먹인다. 헤딩 층 복원·표·그림·NBSP.
    모델 다운로드가 없어 항상 돈다.
  - 실변환 테스트: `pytest.importorskip("docling")` — 패키지 1.4GB + 첫 실행 모델 다운로드가
    필요해 CI에선 건너뛰고 로컬·워커에서 돈다.
"""
import pytest

from config import settings
from rag.chunking import (_PICTURE_PLACEHOLDER, _docling_norm, _sections_from_docling_elements,
                          chunk_file)


def _h(text, page=1):
    return ('section_header', text, page, None)


def _t(text, page=1):
    return ('text', text, page, None)


class TestHeadingLevels:
    """docling은 장·조를 같은 레벨로 낸다 — 한국 규정 번호 체계로 층을 복원해야 조항끼리 묶인다."""

    def test_제목_장_조_3단(self):
        secs = _sections_from_docling_elements([
            _h('주식회사 인티큐브 취업규칙'), _h('제 1 장 총 칙'), _h('제 1 조(목 적)'), _t('본문 A'),
            _h('제 2 조(적용범위)'), _t('본문 B'),
            _h('제 2 장 인 사'), _h('제 3 조(채용)'), _t('본문 C'),
        ])
        assert [s.heading_path for s in secs] == [
            ['주식회사 인티큐브 취업규칙', '제 1 장 총 칙', '제 1 조(목 적)'],
            ['주식회사 인티큐브 취업규칙', '제 1 장 총 칙', '제 2 조(적용범위)'],
            ['주식회사 인티큐브 취업규칙', '제 2 장 인 사', '제 3 조(채용)'],
        ]

    def test_장이_없는_가이드_문서는_제목_아래_2단(self):
        secs = _sections_from_docling_elements([
            _h('[ 총무 | 가이드 ] 노트북 대여'), _h('■ 신청 절차'), _t('본문'), _h('■ 참고 사항'), _t('본문2'),
        ])
        assert [s.heading_path for s in secs] == [
            ['[ 총무 | 가이드 ] 노트북 대여', '■ 신청 절차'],
            ['[ 총무 | 가이드 ] 노트북 대여', '■ 참고 사항'],
        ]

    def test_같은_장_아래_형제_조항은_pack_sections가_묶을_수_있다(self):
        # joinable = 공통 조상 깊이 ≥2. 층 복원이 없으면(전부 L1) 공통 조상이 []라 아무것도 안 묶인다.
        from rag.chunking import _common_prefix
        secs = _sections_from_docling_elements([
            _h('제목'), _h('제 1 장 총칙'), _h('제 1 조'), _t('a'), _h('제 2 조'), _t('b')])
        assert len(_common_prefix([s.heading_path for s in secs])) == 2

    def test_장_패턴_변형(self):
        from rag.chunking import _CHAPTER_RE
        assert _CHAPTER_RE.match('제 3 장 복 무') and _CHAPTER_RE.match('제3장 총칙') and _CHAPTER_RE.match('제 12 장')
        assert not _CHAPTER_RE.match('제 3 조(용어)') and not _CHAPTER_RE.match('장기근속휴가')


class TestElementMapping:
    def test_표는_markdown_그대로_본문에_들어간다(self):
        md = '| 구 분 | 무상수리 |\n| --- | --- |\n| 침수 | 없음 |'
        secs = _sections_from_docling_elements([_h('제목'), _h('■ 기준'), ('table', '', 1, md)])
        assert secs[0].body == md
        assert secs[0].line_pages == [1, 1, 1]          # 표 3줄 ↔ 쪽 3개 — 1:1 대응 (결함 6)

    def test_그림은_자리표시로_남긴다(self):
        secs = _sections_from_docling_elements([_h('제목'), _h('(1) 프로세스'), ('picture', '', 1, None), _t('설명')])
        assert secs[0].body.split('\n') == [_PICTURE_PLACEHOLDER, '설명']

    def test_헤더_푸터는_요소로_오지_않는다는_전제(self):
        # docling은 page_header/page_footer를 iterate_items()에 넣지 않는다(실측). 만약 들어오면
        # 본문으로 취급되므로 여기서 그 라벨이 본문에 섞이지 않게 막을지는 #143 후속 판단 — 현재 전제만 고정.
        secs = _sections_from_docling_elements([_h('제목'), _t('본문')])
        assert secs[0].body == '본문'

    def test_NBSP_정규화(self):
        assert _docling_norm('■ 신청\xa0절차') == '■ 신청 절차'
        assert _docling_norm('\xa0\xa0개인부담\xa0') == '개인부담'

    def test_빈_헤딩과_빈_텍스트는_건너뛴다(self):
        secs = _sections_from_docling_elements([_h('제목'), _h('\xa0'), _t(''), _t('본문')])
        assert [s.heading_path for s in secs] == [['제목']] and secs[0].body == '본문'


docling = pytest.importorskip('docling', reason='docling 미설치 — 실변환 테스트는 설치 환경에서만')


@pytest.fixture(autouse=True)
def _docling_path(monkeypatch):
    monkeypatch.setattr(settings, 'docling_enabled', True)


class TestRealConversion:
    """실제 변환 — 결함 1·2·5가 모델로 처리되는지. reportlab 합성 픽스처."""

    def test_격자표_빈_셀이_자리를_지킨다(self, tmp_path):
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        p = tmp_path / 'grid.pdf'
        table = Table([['Part', 'Free', 'Company', 'Personal'],
                       ['Board', '2y', 'O', ''],
                       ['Water', 'none', '', 'O']])
        table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
        SimpleDocTemplate(str(p), pagesize=A4).build([table])
        body = '\n'.join(c.text for c in chunk_file(p))
        assert '| Board | 2y | O |  |' in body
        assert '| Water | none |  | O |' in body        # 빈 셀을 지우면 두 줄이 같아진다 (결함 5)

    def test_인쇄_헤더_푸터가_본문에_없다(self, tmp_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        W, H = A4
        p = tmp_path / 'printed.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.setFont('Helvetica', 8)
        c.drawString(20, H - 20, '26. 9. 8. 오후 10:25 [총무 | 가이드] 문서 제목')
        c.drawString(20, 18, 'https://intranet.example.com/print?readform 1/1')
        c.setFont('Helvetica-Bold', 13); c.drawString(50, H - 80, 'Software License Guide')
        c.setFont('Helvetica', 10)
        for i in range(6):
            c.drawString(50, H - 120 - i * 18, f'Body sentence {i} about the approval process and returns.')
        c.save()
        chunks = chunk_file(p)
        joined = '\n'.join(c.text for c in chunks) + ' ' + ' '.join(' '.join(c.heading_path) for c in chunks)
        assert 'approval process' in joined                # 본문 보존 (결함 1)
        assert 'https://' not in joined and '오후 10:25' not in joined   # 부산물 제외

    def test_청크마다_실제_쪽(self, tmp_path):
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        W, H = A4
        p = tmp_path / 'multipage.pdf'
        c = canvas.Canvas(str(p), pagesize=A4)
        c.setFont('Helvetica-Bold', 16); c.drawString(50, H - 60, 'Single Heading For Whole Document')
        for page_no in range(3):
            c.setFont('Helvetica', 10)
            for i in range(30):
                c.drawString(50, H - 100 - i * 20, f'Page {page_no + 1} line {i} with enough words to fill the budget.')
            c.showPage()
        c.save()
        chunks = chunk_file(p)
        assert len({ch.page for ch in chunks}) >= 2
        for ch in chunks:
            first = ch.text.strip().split('\n')[0]
            if first.startswith('Page '):
                assert ch.page == int(first.split()[1])

    def test_변환기는_프로세스당_하나(self):
        from rag.chunking import _docling_runtime
        a, sem_a = _docling_runtime(); b, sem_b = _docling_runtime()
        assert a is b and sem_a is sem_b
