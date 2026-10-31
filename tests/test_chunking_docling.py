"""docling PDF 경로 (#143) — 매핑 규칙은 docling 없이, 실변환은 설치 시에만.

두 층으로 나눈다:
  - `_sections_from_docling_elements` 단위 테스트: 튜플만 먹인다. 헤딩 층 복원·표·그림·NBSP.
    모델 다운로드가 없어 항상 돈다.
  - 실변환 테스트: `pytest.importorskip("docling")` — 패키지 1.4GB + 첫 실행 모델 다운로드가
    필요해 CI에선 건너뛰고 로컬·워커에서 돈다.
"""
from pathlib import Path

import pytest

from config import settings
from rag.chunking import (_PICTURE_PLACEHOLDER, _docling_elements, _docling_norm,
                          _markers_from_words, _sections_from_docling_elements,
                          _strip_footnote_markers, chunk_file,
                          count_picture_placeholders)


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

    # ── 조 제목 보정(#240·#241) — 레이아웃 모델이 같은 꼴의 조 제목을 들쭉날쭉 text로 낸다(회계 규정 35/97) ──

    def test_text로_나온_조_제목은_제목으로_올린다(self):
        secs = _sections_from_docling_elements([
            _h('규정'), _h('제1조 (목 적)'), _t('목적 본문'), _t('제2조 (적 용)'), _t('적용 본문'), _t('제3조（회계단위（본사））'), _t('단위 본문')])   # 전각·중첩 괄호도
        assert [s.heading_path for s in secs] == [['규정', '제1조 (목 적)'], ['규정', '제2조 (적 용)'], ['규정', '제3조（회계단위（본사））']]
        assert secs[1].body == '적용 본문'

    def test_장_아래에서_올린_조_제목은_조_층(self):
        secs = _sections_from_docling_elements([_h('규정'), _h('제 1 장 총칙'), _t('제 1 조(목적)'), _t('본문')])
        assert secs[0].heading_path == ['규정', '제 1 장 총칙', '제 1 조(목적)']

    def test_제목과_첫_문장이_한_요소면_떼어서_제목과_본문으로(self):
        # '제1조 (목적) 이 규정은 …' — 조 제목 뒤 간격이 좁으면 docling이 한 덩어리로 낸다(실측 간격 합성 규정 30/30, 회계 규정 3건)
        secs = _sections_from_docling_elements([
            _h('규정'), _h('제1조 (목적)'), _t('본문'), _t('제24조 (자금의 관리) 회사의 자금은 재무팀장이 총괄한다.'), _t('둘째 줄')])
        assert secs[-1].heading_path == ['규정', '제24조 (자금의 관리)']
        assert secs[-1].body == '회사의 자금은 재무팀장이 총괄한다.\n둘째 줄'

    def test_알려진_한계_괄호_뒤_공백_인용으로_시작하는_문단은_제목이_된다(self):
        # DOCX 폴백(#205)과 같은 규칙·같은 위험(#233 결정 2) — 글은 남는다. 이 동작이 바뀌면 이 단언을 고친다
        secs = _sections_from_docling_elements([_h('규정'), _h('제1조 (목적)'), _t('본문'), _t('제3조(보상) 규정에 따라 지급한다')])
        assert secs[-1].heading_path == ['규정', '제3조(보상)'] and secs[-1].body == '규정에 따라 지급한다'

    @pytest.mark.parametrize('line', [
        '제10조(보관)에 따라 보관한다',          # 닫는 괄호에 조사 — 인용
        '제 10 조 3호에 의한 특별 공로 퇴직금',   # 괄호 없음 — 인용(실문서 퇴직금 규정)
        "제98조(이하 '보상조항'이라 한다)에 …",   # 정의 꼴
    ])
    def test_조_인용_문장은_올리지_않는다(self, line):
        secs = _sections_from_docling_elements([_h('규정'), _h('제1조 (목적)'), _t('본문'), _t(line)])
        assert [s.heading_path for s in secs] == [['규정', '제1조 (목적)']] and line in secs[0].body

    def test_문서_제목보다_먼저_나온_조_제목_꼴은_올리지_않는다(self):
        # 문서 제목(첫 section_header) 전의 글은 본문으로 둔다 — 올리면 '규정'이 조 아래 헤딩으로 밀려 제목 경로가 틀어진다
        secs = _sections_from_docling_elements([_t('제1조 (목적)'), _h('규정'), _h('제2조 (적용)'), _t('본문')])
        assert [(s.heading_path, s.body) for s in secs] == [([], '제1조 (목적)'), (['규정', '제2조 (적용)'], '본문')]

    def test_닫는_괄호가_다음_요소로_떨어진_조_제목은_잇는다(self):
        # 징계규정 실문서 3건(#241) — 전에는 ')'가 따로 제목이 되어 제목 경로가 [... , ')']였다
        secs = _sections_from_docling_elements([
            _h('징계규정'), _h('제 1 장 총 칙'), _h('제 7 조 ( 징계종류'), _h(')'), _t('징계의 종류는 5 종으로 한다 .'),
            _h('제 8 조 ( 양정기준'), _t('）'), _t('양정 본문')])
        assert [s.heading_path for s in secs] == [['징계규정', '제 1 장 총 칙', '제 7 조 ( 징계종류 )'],
                                                 ['징계규정', '제 1 장 총 칙', '제 8 조 ( 양정기준 ）']]

    def test_괄호가_떨어진_조_제목이_text로_나와도_잇고_올린다(self):
        # 제목 쪽이 text로 오면(①이 올릴 꼴) ②가 먼저 잇는다 — 안 그러면 ')'가 본문 첫 줄로 남는다(리뷰 지적)
        secs = _sections_from_docling_elements([_h('규정'), _t('제 7 조 ( 징계종류'), _t(')'), _t('본문')])
        assert [(s.heading_path, s.body) for s in secs] == [(['규정', '제 7 조 ( 징계종류 )'], '본문')]

    def test_괄호_하나만_있는_요소라도_조_제목_뒤가_아니면_건드리지_않는다(self):
        secs = _sections_from_docling_elements([_h('규정'), _h('■ 안내'), _t(')'), _t('본문')])
        assert secs[0].heading_path == ['규정', '■ 안내'] and secs[0].body.startswith(')')

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

    def test_그림은_캡션이_없으면_자리표시로_남긴다(self):
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

    def test_목록_항목은_번호와_글머리를_text에_되붙인다(self):
        # docling ListItem은 '2.1'·'-' 같은 marker를 text에서 뗀다. 안 붙이면 소항목 번호가 색인에서 사라진다.
        from types import SimpleNamespace as NS
        def el(label, text, marker=None):
            return NS(label=NS(value=label), text=text, marker=marker, prov=[NS(page_no=1)])
        doc = NS(iterate_items=lambda: [
            (el('section_header', '2. 주요 개정 내용'), 0),
            (el('list_item', '얼리버드 할인율 상향', '2.1'), 1),
            (el('list_item', '정가의 15%로 상향한다.', '-'), 2),
            (el('list_item', '마커 없는 이어지는 줄', ''), 2),
            (el('text', '본문', None), 0),
        ])
        assert [t for _, t, _, _ in _docling_elements(doc)] == [
            '2. 주요 개정 내용', '2.1 얼리버드 할인율 상향', '- 정가의 15%로 상향한다.', '마커 없는 이어지는 줄', '본문']

    def test_빈_헤딩과_빈_텍스트는_건너뛴다(self):
        secs = _sections_from_docling_elements([_h('제목'), _h('\xa0'), _t(''), _t('본문')])
        assert [s.heading_path for s in secs] == [['제목']] and secs[0].body == '본문'


class TestPictureCaption:
    """그림 캡션(#162) — 캡션이 자리표시를 대체한다. 튜플만 먹이므로 VLM·모델 없이 돈다."""

    def test_캡션이_있으면_자리표시_대신_캡션이_본문에_들어간다(self):
        cap = '4단계 흐름도: 신청 접수 → 자격 심사 → 매칭 승인 → 집행·보고'
        secs = _sections_from_docling_elements([_h('제목'), _h('(1) 절차'), ('picture', cap, 1, None), _t('설명')])
        assert secs[0].body.split('\n') == [cap, '설명']

    def test_캡션이_비면_자리표시로_되돌아간다(self):
        # VLM이 죽거나 꺼져 있으면 docling이 캡션을 안 만든다(예외 없음) — 그때의 경로다.
        secs = _sections_from_docling_elements([_h('제목'), ('picture', '', 1, None)])
        assert secs[0].body == _PICTURE_PLACEHOLDER

    def test_캡션이_공백뿐이면_자리표시로_되돌아간다(self):
        secs = _sections_from_docling_elements([_h('제목'), ('picture', '\xa0  ', 1, None)])
        assert secs[0].body == _PICTURE_PLACEHOLDER

    def test_그림마다_캡션_유무가_달라도_각각_처리된다(self):
        secs = _sections_from_docling_elements([
            _h('제목'), _h('절'),
            ('picture', '그림1 설명', 1, None), _t('본문1'),
            ('picture', '', 1, None), _t('본문2'),
        ])
        assert secs[0].body.split('\n') == ['그림1 설명', '본문1', _PICTURE_PLACEHOLDER, '본문2']

    def test_캡션은_annotations에서_꺼낸다(self):
        # 캡션은 element.text가 아니라 annotations에 붙는다. text만 보면 조용히 버려진다 —
        # 실제로 그 상태였고(#162 실측) 캡션이 청크에 안 실렸다.
        from types import SimpleNamespace as NS
        def pic(annotations):
            return NS(label=NS(value='picture'), text='', marker=None,
                      annotations=annotations, prov=[NS(page_no=1)])
        doc = NS(iterate_items=lambda: [
            (pic([NS(text='막대그래프: 면 100% 40℃, 린넨 30℃')]), 0),
            (pic([]), 1),
            (pic(None), 2),
            (pic([NS(text=''), NS(text='두 번째가 살아있다')]), 3),
        ])
        assert [t for _, t, _, _ in _docling_elements(doc)] == [
            '막대그래프: 면 100% 40℃, 린넨 30℃', '', '', '두 번째가 살아있다']

    def test_자리표시_집계는_캡션_실패_수를_센다(self):
        # 지표(kms_index_picture_placeholder_total)의 입력. 캡션 성공분은 일반 본문과 구분되지
        # 않아 셀 수 없다 — 이 함수는 성공률이 아니라 실패 신호다.
        from types import SimpleNamespace as NS
        chunks = [NS(text=f'앞 {_PICTURE_PLACEHOLDER} 뒤'), NS(text='캡션이 붙은 그림 설명'),
                  NS(text=f'{_PICTURE_PLACEHOLDER}\n{_PICTURE_PLACEHOLDER}')]
        assert count_picture_placeholders(chunks) == 3
        assert count_picture_placeholders([]) == 0


docling = pytest.importorskip('docling', reason='docling 미설치 — 실변환 테스트는 설치 환경에서만')


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
        assert _docling_runtime() is _docling_runtime()

    def test_메모리_설정이_파이프라인에_실제로_전달된다(self):
        # 조용히 무시되는 설정을 잡기 위한 테스트다. docling_page_batch_size가 정확히 그랬다 —
        # 값은 들어가는데 2.126의 기본 파이프라인이 그 필드를 읽지 않아 아무 효과가 없었고,
        # 그 사실이 개발계 OOM으로 드러나기 전까지 아무도 몰랐다(#151).
        # PdfPipelineOptions는 pydantic 모델이라 필드명이 바뀌면 대입 자체가 실패한다 —
        # docling 업그레이드 때 이 테스트가 먼저 깨진다.
        from docling.datamodel.base_models import InputFormat
        from rag.chunking import _docling_runtime

        converter = _docling_runtime()
        opts = converter.format_to_options[InputFormat.PDF].pipeline_options
        assert opts.layout_batch_size == settings.docling_layout_batch_size
        assert opts.queue_max_size == settings.docling_queue_max_size
        assert opts.accelerator_options.num_threads == settings.docling_num_threads
        assert opts.do_ocr is settings.docling_do_ocr

    def test_캡션_설정이_파이프라인에_실제로_전달된다(self):
        # 위와 같은 이유(#151). 특히 scale은 조용히 틀리면 VLM이 글자를 못 읽고 **원문에 없는
        # 내용을 지어낸다**(#162 실측: scale 1.0에서 3회 모두 환각). 값이 안 걸리면 기본 2.0으로
        # 도는데 에러가 없어 아무도 모른다 — 그래서 값까지 단언한다.
        from docling.datamodel.base_models import InputFormat
        from rag.chunking import _PICTURE_CAPTION_PROMPT, _caption_headers, _docling_runtime

        converter = _docling_runtime()
        opts = converter.format_to_options[InputFormat.PDF].pipeline_options
        assert opts.do_picture_description is True
        assert opts.enable_remote_services is True   # 빠지면 OperationNotAllowed로 변환 자체가 죽는다
        api = opts.picture_description_options
        assert api.scale == settings.vlm_caption_scale
        assert api.timeout == settings.vlm_caption_timeout_seconds
        assert api.headers == _caption_headers() == {}            # 키 비면 전과 같은 요청(#290) — 기본 .env는 사내
        # temperature 0 = 결정적 캡션(#209): 3회 재변환 25/25 동일 실측. 빠지면 샘플링으로 돌아가 재인제스트마다 캡션이 달라진다
        assert api.params == {'model': settings.vlm_caption_model, 'temperature': settings.vlm_caption_temperature,
                              'max_completion_tokens': settings.vlm_caption_max_tokens}   # 새 표기(#290 결정 3) — gpt-5가 max_tokens 거부
        # 출력 상한(#247 결정 16) — 빠지면 7B 반복 고리 폭주(실측 6천 자대)가 색인되고 45초 상한을 넘겨 문서 재시도로 번진다
        assert 655 * 2 <= api.params['max_completion_tokens'] <= 4000
        # 면적 하한(#247 결정 17) — 0.05면 내용 있는 작은 캡처(1.2~4.6%)가 캡션 없이 빠졌다. 아이콘(0.39%)은 계속 제외
        assert api.picture_area_threshold == settings.vlm_caption_area_threshold == 0.01
        # XLSX만 하한 0(#249 결정 1) — 엑셀은 시트 칸 단위라 면적 비율이 그림 크기를 반영하지 못한다. DOCX는 1% 그대로
        office = {fmt: converter.format_to_options[fmt].pipeline_options.picture_description_options
                  for fmt in (InputFormat.DOCX, InputFormat.XLSX)}
        assert office[InputFormat.DOCX].picture_area_threshold == 0.01
        assert office[InputFormat.XLSX].picture_area_threshold == 0.0
        assert office[InputFormat.XLSX].prompt == _PICTURE_CAPTION_PROMPT          # 하한 말고는 같은 캡션 설정
        assert office[InputFormat.XLSX].params == api.params
        # 형식마다 캡션 감시(_PictureGuard)가 따로 붙는다 — XLSX가 DOCX 파이프라인을 공유하지 않으므로
        from rag.chunking import _PictureGuard
        guarded = [p for p in converter.initialized_pipelines.values() if any(isinstance(m, _PictureGuard) for m in p.enrichment_pipe)]
        assert len(guarded) >= 3
        assert api.params['temperature'] == 0.0 and api.timeout == 45.0   # .env.dev가 config를 덮으므로 값 자체도 단언
        # 동시 1(#223 결정 7) — 동시로 보내면 vLLM 배치 탓에 온도 0이어도 캡션이 실행마다 갈린다(실문서 실측). 올리기 전에 config 주석부터
        assert api.concurrency == settings.vlm_caption_concurrency == 1
        assert api.prompt == _PICTURE_CAPTION_PROMPT
        assert str(api.url) == settings.vlm_caption_url   # AnyUrl — str 캐스팅 후 비교

    def test_VLM이_실패하면_문서가_일시_실패한다(self, tmp_path, fake_vlm):
        """#223 결정 3 — #162 이래 VLM 실패는 자리표시로 ready가 됐다(docling이 오류를 삼켜 빈 캡션을 준다).
        그림 내용이 빠진 문서가 "완료"로 보이는 조용한 불완전 색인이라, 이제 CaptionFailedError(일시)로 올린다.
        docling 업그레이드로 실패를 삼키는 방식이 바뀌면(예: 예외로 바뀜) 이 테스트가 먼저 알려준다."""
        from rag.chunking import CaptionFailedError
        fake_vlm.status = 500
        with pytest.raises(CaptionFailedError):
            chunk_file(_picture_pdf(tmp_path))

    def test_VLM이_되면_캡션이_자리표시를_대체한다(self, tmp_path, fake_vlm):
        from tests.conftest import FAKE_CAPTION
        chunks = chunk_file(_picture_pdf(tmp_path))
        assert FAKE_CAPTION in '\n'.join(c.text for c in chunks)
        assert count_picture_placeholders(chunks) == 0


def _picture_pdf(tmp_path, pictures: int = 1):
    """그림이 든 PDF — 1×1 PNG를 300×200pt로(쪽 면적의 약 12%, 캡션 하한 5% 위). 내용은 무관하다."""
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Image as RLImage, PageBreak, SimpleDocTemplate
    png = tmp_path / 'fig.png'
    png.write_bytes(bytes.fromhex(
        '89504e470d0a1a0a0000000d494844520000000100000001080200000090'
        '7753de0000000c4944415408d763f8cfc00000030101003c2f8e3b0000000049454e44ae426082'))
    p = tmp_path / 'pic.pdf'
    story = []
    for i in range(pictures):
        story += [RLImage(str(png), width=300, height=200), PageBreak()]
    SimpleDocTemplate(str(p), pagesize=A4).build(story)
    return p


# ── 원격 OCR (#168) ─────────────────────────────────────────────────────────────────
# 표 이미지(TableItem)의 셀은 Triton에 올린 RapidOCR이 읽는다. 여기서 고정하는 것 셋:
#   ① 옵션이 실제로 파이프라인에 들어가는가(특히 scale 3.0 — KServe 옵션 기본값 2.0과 다르다)
#   ② 원격이 죽으면 문서가 **실패**하는가(조용한 빈 격자 색인 금지 — docling은 삼키므로 우리가 올린다)
#   ③ 서버가 있으면 숫자·영문 표 셀이 실제로 읽히는가(도달 가능할 때만 — 한글은 폰트 자산이 필요해 안 쓴다)

def _reset_converter(ch):
    """설정 변경이 먹으려면 컨버터 싱글턴을 비워야 한다 — 전역 캐시라 재생성이 안 된다."""
    ch._docling_converter = None


def _triton_ready(base_url: str) -> bool:
    import requests
    try:
        return requests.get(f'{base_url}/v2/health/ready', timeout=2).status_code == 200
    except Exception:            # noqa: BLE001 — 도달 불가는 스킵 사유일 뿐
        return False


def _table_image_pdf(tmp_path):
    """숫자·영문 표를 **비트맵**으로 그려 PDF에 넣는다 — 텍스트 레이어가 없어 OCR만이 읽을 수 있다."""
    from PIL import Image, ImageDraw, ImageFont
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import Image as RLImage, SimpleDocTemplate
    rows = [['Part', 'Warranty', 'Fee'], ['Motor', '3y', '48000'], ['Blade', '1y', '12000'], ['Pack', '2y', '35000']]
    cw, rh = 400, 110
    W, H = cw * 3, rh * len(rows) + 20
    im = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=44)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            x0, y0 = c * cw, 10 + r * rh
            d.rectangle([x0, y0, x0 + cw, y0 + rh], outline='black', width=4)
            d.text((x0 + 30, y0 + 28), cell, fill='black', font=font)
    png = tmp_path / 'table.png'; im.save(png)
    p = tmp_path / 'table_image.pdf'
    SimpleDocTemplate(str(p), pagesize=A4).build([RLImage(str(png), width=480, height=H * 480 / W)])
    return p, rows


class TestConversionGuard:
    """raises_on_error=True가 못 잡는 불완전 변환을 _convert_local이 올리는지 — 모델 없이 가짜 컨버터로."""

    # 가드는 변환 프로세스 안의 _convert_local에 있다 — 가짜 컨버터는 이 프로세스에서만 보이므로 그것을 직접 부른다.
    def _with_result(self, monkeypatch, status, errors):
        import types
        import rag.chunking as ch
        self.doc = types.SimpleNamespace(iterate_items=lambda: iter(()))
        res = types.SimpleNamespace(status=status, errors=errors, document=self.doc)
        conv = types.SimpleNamespace(convert=lambda *a, **k: res)
        monkeypatch.setattr(ch, '_docling_runtime', lambda: conv)
        return ch

    def test_부분_성공은_실패로_올린다(self, monkeypatch, tmp_path):
        # 문서 타임아웃이 나면 docling은 처리된 쪽까지만 담고 PARTIAL_SUCCESS로 **정상 반환**한다 —
        # 그대로 두면 뒤쪽 쪽이 빠진 문서가 ready가 된다.
        from docling.datamodel.base_models import ConversionStatus
        ch = self._with_result(monkeypatch, ConversionStatus.PARTIAL_SUCCESS, [])
        with pytest.raises(RuntimeError, match='partial_success'):
            ch._convert_local(tmp_path / 'x.pdf')

    def test_성공이어도_errors가_있으면_실패로_올린다(self, monkeypatch, tmp_path):
        # 원격 OCR 실패는 status를 안 바꾸고 errors에만 남긴다(kserve_v2_ocr_model.py) — 표가 빈 격자로 색인된다.
        from docling.datamodel.base_models import (ConversionStatus, DoclingComponentType,
                                                    ErrorItem, FailureCategory)
        err = ErrorItem(component_type=DoclingComponentType.MODEL, module_name='KserveV2OcrModel',
                        error_message='connection refused', category=FailureCategory.INFERENCE_FAILURE)
        ch = self._with_result(monkeypatch, ConversionStatus.SUCCESS, [err])
        with pytest.raises(RuntimeError, match='KserveV2OcrModel'):
            ch._convert_local(tmp_path / 'x.pdf')

    def test_완전_성공은_통과한다(self, monkeypatch, tmp_path):
        from docling.datamodel.base_models import ConversionStatus
        ch = self._with_result(monkeypatch, ConversionStatus.SUCCESS, [])
        assert ch._convert_local(tmp_path / 'x.pdf') is self.doc


class TestRemoteOcr:
    def test_OCR_설정이_파이프라인에_실제로_전달된다(self, monkeypatch):
        # 캡션 테스트와 같은 이유(#151·#162) — 조용히 무시되는 설정을 잡는다. scale이 특히 그렇다:
        # KserveV2OcrOptions 기본값은 2.0이라 빠뜨리면 에러 없이 인프로세스 실측(3.0)과 다른 결과가 나온다.
        import rag.chunking as ch
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import OcrMode
        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        _reset_converter(ch)
        try:
            converter = ch._docling_runtime()
            opts = converter.format_to_options[InputFormat.PDF].pipeline_options
            assert opts.do_ocr is True
            o = opts.ocr_options
            assert o.kind == 'kserve_v2_ocr'
            assert o.scale == ch._OCR_SCALE == 3.0
            assert o.lang == [ch._OCR_LANG] == ['korean']
            assert o.transport == 'http'
            assert o.model_name == settings.ocr_kserve_model_name
            assert o.url == settings.ocr_kserve_url
            assert o.timeout == settings.ocr_kserve_timeout_seconds
            assert o.mode == OcrMode.DEFAULT          # 비트맵 영역만 — 본문 텍스트 레이어 무손상의 근거
        finally:
            _reset_converter(ch)

    def test_원격_OCR이_죽으면_문서가_실패한다(self, tmp_path, monkeypatch):
        """VLM(#162)과 반대다 — 캡션은 없어도 진행하지만 OCR 실패는 문서를 failed로 확정한다.

        docling은 KServe 실패를 삼키고 쪽을 그대로 내보내므로(errors에만 기록) 그대로 두면 표가
        **빈 격자로 조용히 색인**된다. 그게 #168 이전 상태와 같아 보여도 '읽는다'는 약속이 깨진 것이라
        실패로 올린다. 오래 죽어 있으면 docling_do_ocr로 끄는 게 운영 절차다(config.py).
        """
        import rag.chunking as ch
        p, _ = _table_image_pdf(tmp_path)
        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        monkeypatch.setattr(settings, 'ocr_kserve_url', 'http://127.0.0.1:1')
        monkeypatch.setattr(settings, 'ocr_kserve_timeout_seconds', 2.0)
        _reset_converter(ch)
        try:
            with pytest.raises(RuntimeError, match='docling 변환 불완전'):
                chunk_file(p)
        finally:
            _reset_converter(ch)

    def test_표_이미지_셀을_원격_OCR이_읽는다(self, tmp_path, monkeypatch):
        import rag.chunking as ch
        if not _triton_ready(settings.ocr_kserve_url):
            pytest.skip(f'Triton OCR 미도달: {settings.ocr_kserve_url} (OCR_KSERVE_URL로 지정)')
        p, rows = _table_image_pdf(tmp_path)
        monkeypatch.setattr(settings, 'docling_do_ocr', True)
        _reset_converter(ch)
        try:
            body = '\n'.join(c.text for c in chunk_file(p))
        finally:
            _reset_converter(ch)
        for value in ('48000', '12000', '35000'):        # 금액 셀 — 표 이미지에서 제일 중요한 값
            assert value in body, body


class TestFootnoteMarkers:
    """위첨자 각주 참조 'N)' (#204). 열거 '1) …'와 글자 모양이 같아 **글자 크기 비율**로만 가른다.
    실측(타임레포트 매뉴얼): 각주 7.3~8.0 / 본문 11~12 → 비율 0.67~0.73. 정의 줄·열거는 1.0."""

    def test_같은_줄_본문보다_작은_N괄호_토큰만_각주(self):
        words = [('승인자', 12.0, 144.0), (':', 12.0, 144.0), ('PM', 12.0, 144.0),
                 ('2)', 8.0, 142.0), ('>팀장>사업부장', 12.0, 144.0)]   # 위첨자는 top이 2pt 위
        assert _markers_from_words(words) == {'2)'}

    def test_열거는_본문과_같은_크기라_각주가_아니다(self):
        words = [('다.', 11.0, 300.0), ('1)', 11.0, 300.0), ('직원이', 11.0, 300.0),
                 ('2)', 11.0, 300.0), ('회사의', 11.0, 300.0)]
        assert _markers_from_words(words) == set()

    def test_정의_줄은_줄_전체가_작아_걸리지_않는다(self):
        words = [('2)', 8.9, 500.0), ('PM', 8.9, 500.0), ('없는', 8.9, 500.0), ('경우', 8.9, 500.0)]
        assert _markers_from_words(words) == set()

    def test_줄_묶기는_top_허용치_밖이면_다른_줄(self):
        # 작은 글자 '3)'이 혼자 다른 줄(10pt 아래)에 있으면 비교 대상이 없어 각주가 아니다
        words = [('계획대상선택', 14.0, 70.0), ('3)', 9.4, 85.0)]
        assert _markers_from_words(words) == set()

    def test_본문에서_단어_뒤_표시만_뗀다(self):
        assert _strip_footnote_markers('( 승인자 : PM 2) > 팀장 > 사업부장 )', {'2)'}) == '( 승인자 : PM > 팀장 > 사업부장 )'
        assert _strip_footnote_markers('확정 실적은 PM 2) , 팀장 승인 전까지', {'2)'}) == '확정 실적은 PM , 팀장 승인 전까지'
        assert _strip_footnote_markers('계획대상선택 3) → [ 이전계획불러오기 ]', {'3)'}) == '계획대상선택 → [ 이전계획불러오기 ]'

    def test_정의_줄과_다른_번호_열거는_그대로(self):
        assert _strip_footnote_markers('2) PM 없는 경우 생략', {'2)'}) == '2) PM 없는 경우 생략'
        enum = '다. 1) 직원이 임원을 수행하여 출장하는 경우 2) 회사의 손님을 수행'
        assert _strip_footnote_markers(enum, set()) == enum                 # 그 쪽에 각주 표시가 없으면 무변경
        assert _strip_footnote_markers(enum, {'3)'}) == enum                # 다른 번호면 무변경

    def test_더_큰_숫자의_일부는_각주가_아니다(self):
        # 리뷰(#204): lookbehind의 \\w가 숫자를 포함해 "서식 12)"의 "2)"를 지우면 숫자가 변조된다
        assert _strip_footnote_markers('서식 12) 참조', {'2)'}) == '서식 12) 참조'
        assert _strip_footnote_markers('총 1000 2) 명', {'2)'}) == '총 1000 명'     # 공백 뒤는 각주

    def test_두_자리_각주와_뒤에_붙은_화살표(self):
        words = [('상세', 12.0, 50.0), ('10)', 8.0, 48.0), ('→', 12.0, 50.0)]
        assert _markers_from_words(words) == {'10)'}
        assert _strip_footnote_markers('상세 10)→ 다음', {'10)'}) == '상세→ 다음'

    def test_줄바꿈_너머의_정의_줄_머리는_지우지_않는다(self):
        # 요소가 합쳐진 뒤 적용돼도 "상신\n2) PM 없는 경우 생략"의 줄 머리 '2)'는 남아야 한다
        text = '6. 상신\n2) PM 없는 경우 생략'
        assert _strip_footnote_markers(text, {'2)'}) == text

    def test_요소_흐름에_쪽별로_적용된다(self, monkeypatch):
        class _El:
            def __init__(self, label, text, page):
                self.label, self.text, self.marker = label, text, ''
                self.prov = [type('P', (), {'page_no': page})()]
        class _Doc:
            def iterate_items(self):
                return [(_El('text', '승인자 : PM 2) > 팀장', 4), None), (_El('text', '열거 2) 유지', 5), None)]
        els = list(_docling_elements(_Doc(), {4: {'2)'}}))
        assert els[0][1] == '승인자 : PM > 팀장'
        assert els[1][1] == '열거 2) 유지'      # 5쪽엔 각주 표시가 없다


class TestFootnoteSidecarPdf:
    """_footnote_markers I/O 래퍼 — reportlab으로 그린 실제 PDF에서 위첨자만 잡히는지(#204). docling 불필요."""

    def _pdf(self, tmp_path):
        from reportlab.pdfgen import canvas
        p = tmp_path / 'footnote.pdf'
        c = canvas.Canvas(str(p), pagesize=(595, 842))
        c.setFont('Helvetica', 12); c.drawString(72, 700, 'approver : PM')
        c.setFont('Helvetica', 8); c.drawString(152, 703, '2)')               # 위첨자 — 작고 3pt 위
        c.setFont('Helvetica', 12); c.drawString(166, 700, '> team > division')
        c.setFont('Helvetica', 12); c.drawString(72, 650, 'case 1) employee 2) guest')   # 열거 — 본문 크기
        c.setFont('Helvetica', 9); c.drawString(72, 100, '2) omitted when no PM')        # 정의 줄 — 전체가 작음
        c.showPage(); c.save()
        return p

    def test_실제_PDF에서_위첨자만_각주로_잡힌다(self, tmp_path):
        from rag.chunking import _footnote_markers
        assert _footnote_markers(self._pdf(tmp_path)) == {1: {'2)'}}

    def test_못_여는_파일은_빈_dict와_경고(self, tmp_path, caplog):
        from rag.chunking import _footnote_markers
        import logging
        with caplog.at_level(logging.WARNING, logger='rag.chunking'):
            assert _footnote_markers(tmp_path / 'missing.pdf') == {}
        assert any('각주 사이드카 실패' in r.message for r in caplog.records)


# ── 변환 프로세스 (#223 결정 1·2) ─────────────────────────────────────────────────────
# docling에는 취소 API가 없다 — 스레드로 돌면 마감을 넘겨도 멈출 수 없어 고아 스레드가 세마포어를 쥐고 남았다.
# 그래서 변환을 자식 프로세스로 옮기고 마감에 kill한다. 여기서 고정하는 것:
#   ① 마감을 넘기면 TimeoutError이고, 자식이 교체돼 다음 변환은 정상이다
#   ② 자식이 죽어 있어도 다음 요청이 새로 띄운다
#   ③ 자식을 거쳐 온 문서가 이 프로세스에서 변환한 것과 같다(직렬화 무손실)

class TestConvertProcess:
    def test_마감을_넘기면_TimeoutError_자식_교체_다음_변환은_정상(self, tmp_path, monkeypatch, fake_vlm):
        import rag.chunking as ch
        p = _picture_pdf(tmp_path)
        deadline = settings.docling_document_timeout_seconds
        fake_vlm.delay = 30                      # 캡션 단계에서 멈춘다 — #223 전에는 어떤 마감도 이 단계를 못 끊었다
        monkeypatch.setattr(settings, 'docling_document_timeout_seconds', 3.0)
        with pytest.raises(TimeoutError, match='3초'):
            ch.docling_convert(p)
        assert ch._convert_process._proc is None          # kill하고 비웠다 — 다음 요청이 새로 띄운다
        fake_vlm.delay = 0
        monkeypatch.setattr(settings, 'docling_document_timeout_seconds', deadline)
        assert ch.docling_convert(p).pictures             # 새 자식에서 정상 변환

    def test_자식이_죽어_있으면_다음_요청이_새로_띄운다(self, tmp_path, fake_vlm):
        import rag.chunking as ch
        p = _picture_pdf(tmp_path)
        ch.docling_convert(p)
        old = ch._convert_process._proc
        old.kill(); old.wait(10)                          # OOM kill 같은 외부 종료
        assert ch.docling_convert(p).pictures
        assert ch._convert_process._proc is not old

    def test_자식을_거친_문서가_이_프로세스_변환과_같다(self, tmp_path, fake_vlm):
        # 자식은 DoclingDocument를 피클로 돌려준다. 손실이 있으면 청크가 조용히 달라진다 — 형식별로 대조한다
        # (보유 코퍼스 전수 대조는 #223 PR의 실측 — 이 테스트는 그 회귀 고정).
        from io import BytesIO
        import openpyxl
        from docx import Document
        from PIL import Image
        import rag.chunking as ch
        buf = BytesIO(); Image.new('RGB', (120, 80), (200, 30, 30)).save(buf, format='PNG')
        d = Document(); d.add_heading('제목', level=1); d.add_paragraph('본문 문단.')
        t = d.add_table(rows=1, cols=2); t.cell(0, 0).text = '칸'; d.add_picture(BytesIO(buf.getvalue()))
        docx = tmp_path / 'a.docx'; d.save(docx)
        wb = openpyxl.Workbook(); wb.active.append(['등급', '적립']); wb.active.append(['VIP', '2.0'])
        xlsx = tmp_path / 'a.xlsx'; wb.save(xlsx)
        for path, kw in ((_picture_pdf(tmp_path), {'max_num_pages': settings.docling_max_num_pages}),
                         (docx, {}), (xlsx, {})):
            assert ch.docling_convert(path, **kw).export_to_dict() == ch._convert_local(path, **kw).export_to_dict(), path

    def test_abort하면_기다리던_변환이_바로_끝나고_다음_변환은_정상(self, tmp_path, fake_vlm):
        # 색인 잡이 취소될 때(rag/ingestion.py) 부른다 — 기다리던 스레드가 마감(900초)까지 잠금을 쥐지 않게(#223 리뷰)
        import threading
        import time
        import rag.chunking as ch
        p = _picture_pdf(tmp_path)
        ch.docling_convert(p)                             # 자식 기동(모델 로딩)은 측정에서 뺀다
        fake_vlm.delay = 60
        errors = []
        t = threading.Thread(target=lambda: errors.append(_raises(ch.docling_convert, p)))
        t.start()
        time.sleep(3)                                     # 캡션 단계에 들어가 VLM을 기다리는 중
        started = time.monotonic()
        ch.abort_conversion()
        t.join(30)
        assert not t.is_alive() and time.monotonic() - started < 15
        assert isinstance(errors[0], RuntimeError) and '비정상 종료' in str(errors[0])
        fake_vlm.delay = 0
        assert ch.docling_convert(p).pictures


    def test_변환과_무관한_설정이_바뀌면_자식을_다시_띄우지_않는다(self, tmp_path, monkeypatch):
        # #267 — 설정 전체를 비교하던 때는 테스트가 임베딩 주소·한도를 바꿔 끼울 때마다 콜드 기동(8.5초)이 났다
        import rag.chunking as ch
        p = tmp_path / 'a.md'
        p.write_text('# 제목\n\n본문.\n', encoding='utf-8')
        ch.docling_convert(p)
        first = ch._convert_process._proc
        monkeypatch.setattr(settings, 'rerank_timeout', settings.rerank_timeout + 1)
        ch.docling_convert(p)
        assert ch._convert_process._proc is first
        monkeypatch.setattr(settings, 'docling_table_mode', 'fast' if settings.docling_table_mode != 'fast' else 'accurate')
        ch.docling_convert(p)
        assert ch._convert_process._proc is not first        # 변환 설정은 여전히 자식에 닿아야 한다

    def test_이_모듈이_읽는_설정은_모두_자식에_넘어간다(self):
        # 자식은 _CHILD_SETTING_PREFIXES 설정만 받는다 — 다른 접두사 설정을 자식 코드가 읽으면 부모가 바꿔도 옛 값으로 돈다
        import re
        import rag.chunking as ch
        src = ''.join(f.read_text(encoding='utf-8') for f in Path(ch.__file__).parent.glob('*.py'))  # 패키지 전체
        names = set(re.findall(r'settings\.([a-z_]+)', src)) - {'model_dump'}
        assert names and all(n.startswith(ch._CHILD_SETTING_PREFIXES) for n in names), sorted(names)


def _raises(fn, *args):
    try:
        fn(*args)
    except Exception as e:      # noqa: BLE001
        return e
    return None


# ── 표 블록 분할·쪽 넘김 잇기·칸 글자 정리 (#252) ─────────────────────────────────────

class TestTableBlocks:
    def _long(self, n=45):
        return [['번호', '품목', '단가']] + [[str(i), f'품목-{i:03d}', f'{i * 1234 % 97000 + 1000:,}원'] for i in range(1, n + 1)]

    def test_긴_표는_행_묶음으로_나뉘고_묶음마다_헤더가_반복되며_행이_잘리지_않는다(self):
        # #246 실측: 45행 표가 SentenceSplitter 경계에서 `30,` / `148원 | 5 |`로 쪼개지고 뒷 청크에 헤더가 없었다(결함 B)
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section, to_markdown_table
        rows = self._long()
        block = TableBlock(rows)
        sec = Section(['문서'], 1, '머리말\n' + block.render(), None, blocks=['머리말', block])
        chunks = _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        tables = [t.removeprefix('머리말\n') for t, *_ in chunks if '| 번호 |' in t]   # 큰 표는 묶음마다 표 위 한 줄(결정 7)
        assert len(tables) >= 2                                            # 1,000토큰 예산이라 둘 이상
        body_lines = [l for t in tables for l in t.split('\n')[2:]]        # 헤더·구분선 뺀 데이터 행
        assert body_lines == to_markdown_table(rows[0], rows[1:]).split('\n')[2:]   # 행이 하나도 안 잘리고 순서 그대로
        assert all(t.split('\n')[:2] == to_markdown_table(rows[0], []).split('\n') for t in tables)   # 묶음마다 헤더

    def test_작은_표는_앞뒤_글과_한_청크로_다시_이어진다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section, to_markdown_table
        block = TableBlock([['a', 'b'], ['1', '2']])
        body = '앞 글.\n' + block.render() + '\n뒤 글.'
        sec = Section(['문서'], 1, body, [1, 1, 1, 2, 2], blocks=['앞 글.', block, '뒤 글.'])
        chunks = _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert chunks == [(body, 0, None)]                                       # 결정 7(c): 예산 안이면 블록 사이를 다시 잇는다

    def test_쪽을_넘긴_표는_헤더가_같으면_하나로_이어진다(self):
        # #252 결정 5(E): docling은 쪽마다 표를 따로 낸다(45행 → 36 + 10). 바로 이어지고 첫 행이 같으면 뒤 표 헤더를 떼고 잇는다
        from rag.chunking import TableBlock, _sections_from_docling_elements
        rows = self._long(5)
        secs = _sections_from_docling_elements([_h('제목'), ('table', '', 1, rows[:4]), ('table', '', 2, [rows[0]] + rows[4:])])
        assert [b.rows for b in secs[0].blocks] == [rows]
        assert secs[0].body == TableBlock(rows).render()
        assert secs[0].line_pages[:1] == [1] and len(secs[0].line_pages) == len(rows) + 1

    def test_헤더가_다르거나_사이에_글이_있으면_잇지_않는다(self):
        from rag.chunking import TableBlock, _sections_from_docling_elements
        a, b = [['x', 'y'], ['1', '2']], [['x', 'z'], ['3', '4']]
        secs = _sections_from_docling_elements([_h('제목'), ('table', '', 1, a), ('table', '', 1, b)])
        assert [x.rows for x in secs[0].blocks] == [a, b]
        secs = _sections_from_docling_elements([_h('제목'), ('table', '', 1, a), _t('설명'), ('table', '', 2, a)])
        assert [x.rows if isinstance(x, TableBlock) else x for x in secs[0].blocks] == [a, '설명', a]

    def test_옛_계약_markdown_문자열_표는_글_블록으로_남는다(self):
        from rag.chunking import _sections_from_docling_elements
        md = '| 구 분 | 무상수리 |\n| --- | --- |\n| 침수 | 없음 |'
        secs = _sections_from_docling_elements([_h('제목'), ('table', '', 1, md)])
        assert secs[0].blocks == [md] and secs[0].body == md

    def test_칸_글자는_줄바꿈을_공백으로_접고_엑셀_자정_꼬리를_뗀다(self):
        from rag.chunking import _cell_text_norm
        assert _cell_text_norm('1. 요청서 수신\n2. 담당자 배정') == '1. 요청서 수신 2. 담당자 배정'     # C
        assert _cell_text_norm('2024-10-02 00:00:00') == '2024-10-02'                               # F
        assert _cell_text_norm('2024-10-02 09:30:00') == '2024-10-02 09:30:00'                      # 실제 시각은 유지
        assert _cell_text_norm('구분\xa0(원)') == '구분 (원)'

    def test_묶기는_블록을_잇고_body는_블록에서_파생된다(self):
        from rag.chunking import Section, TableBlock, _pack_sections
        block = TableBlock([['a'], ['1']])
        s1 = Section(['문서', '1장', '1.1'], None, '글', None, blocks=['글'])
        s2 = Section(['문서', '1장', '1.2'], None, block.render(), None, blocks=[block])
        packed, = _pack_sections([s1, s2])
        assert packed.blocks == ['1.1\n글\n1.2', block]
        assert packed.body == '\n'.join(['1.1', '글', '1.2', block.render()])

    def test_이어진_표의_행_묶음은_자기_쪽_번호를_가진다(self):
        # 리뷰 지적: 잇기(E)로 한 표가 되면 뒷 쪽 행 묶음이 앞 쪽 번호를 물려받는다(#137 결함 6) — row_pages로 막는다
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import _page_at, _sections_from_docling_elements, _split_section
        rows = self._long(60)
        secs = _sections_from_docling_elements([_h('제목'), ('table', '', 1, rows[:31]), ('table', '', 2, [rows[0]] + rows[31:])])
        chunks = _split_section(secs[0], SentenceSplitter(chunk_size=512, chunk_overlap=50))
        pages = [_page_at(secs[0], off) for _, off, _ in chunks]
        assert pages[0] == 1 and pages[-1] == 2 and pages == sorted(pages)      # 앞 묶음은 1쪽, 뒷 묶음은 2쪽
        assert secs[0].line_pages[:2] == [1, 1] and secs[0].line_pages[-1] == 2

    def test_표_뒤에_오는_글의_오프셋과_목록_번호_복원(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section, to_markdown_table
        block = TableBlock([['a', 'b'], ['1', '2']])
        body = block.render() + '\n1. 항목 하나\n2. 항목 둘'
        sec = Section(['문서'], None, body, None, blocks=[block, '1. 항목 하나\n2. 항목 둘'])
        chunks = _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert chunks == [(body, len(to_markdown_table(['a', 'b'], [])) + 1, None)]   # 표 + 뒤 글이 한 청크, 오프셋은 첫 데이터 행

    def test_그림이_사이에_있으면_잇지_않는다(self):
        from rag.chunking import TableBlock, _sections_from_docling_elements
        a = [['x', 'y'], ['1', '2']]
        secs = _sections_from_docling_elements([_h('제목'), ('table', '', 1, a), ('picture', '캡션', 1, None), ('table', '', 2, a)])
        assert secs[0].blocks == [TableBlock(a, [1, 1]), '캡션', TableBlock(a, [2, 2])]

    def test_작은_표는_앞_문장과_한_청크에_남는다(self):
        # 결정 7(c): 블록마다 따로 자르면 '2. 사이즈 스펙 (단위: cm)'만 혼자 남고 표가 맥락을 잃었다(실문서 diff)
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section
        block = TableBlock([['사이즈', '총장'], ['S', '68'], ['M', '70']])
        intro = '2. 품목별 사이즈 스펙 (단위: cm)'
        sec = Section(['문서'], None, intro + '\n' + block.render(), None, blocks=[intro, block])
        chunks = _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert [t for t, *_ in chunks] == [intro + '\n' + block.render()] and chunks[0][1] == 0

    def test_큰_표는_묶음마다_표_위_한_줄을_머리에_붙인다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section
        block = TableBlock(self._long(60))
        intro = '본문 문단입니다.\n별첨 1) 품목 단가표'
        sec = Section(['문서'], None, intro + '\n' + block.render(), None, blocks=[intro, block])
        chunks = [t for t, *_ in _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))]
        tables = [t for t in chunks if '| 번호 |' in t]
        assert len(tables) >= 2 and all(t.startswith('별첨 1) 품목 단가표\n| 번호 |') for t in tables)
        assert chunks[0] == intro                                          # 앞 글은 따로(큰 표와는 합치지 않음)

    def test_같은_글_블록에서_나온_조각끼리는_합치지_않는다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, _split_section
        text = '\n'.join(f'{i}번째 문장은 충분히 길어서 분할기가 여러 조각으로 나누게 만든다 반복 반복 반복.' for i in range(60))
        sec = Section(['문서'], None, text, None, blocks=[text])
        splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
        assert [t for t, *_ in _split_section(sec, splitter)] == [c for c in splitter.split_text(text)]

    def test_큰_표_위_한_줄뿐인_글은_혼자_남지_않는다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section
        block = TableBlock(self._long(60))
        intro = '[ 별표 1] 품목 단가표'
        sec = Section(['문서'], None, intro + '\n' + block.render(), None, blocks=[intro, block])
        chunks = [t for t, *_ in _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))]
        assert all(t.startswith(intro + '\n| 번호 |') for t in chunks) and len(chunks) >= 2


# ── 큰 표 묶음의 표 정보·크기 상한 (#257) ────────────────────────────────────────────

class TestTableExpansionMeta:
    def _long(self, n=60, head=('번호', '품목', '단가')):
        return [list(head)] + [[str(i), f'품목-{i:03d}', f'{i * 1234 % 97000 + 1000:,}원'] for i in range(1, n + 1)]

    def _sec(self, blocks, page=None):
        from rag.chunking import Section, TableBlock
        body = '\n'.join(b.render() if isinstance(b, TableBlock) else b for b in blocks)
        return Section(['문서'], page, body, None, blocks=blocks)

    def test_큰_표의_묶음은_표_전체를_함께_돌려주고_작은_표는_안_붙인다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import TableBlock, _split_section
        big, small = TableBlock(self._long()), TableBlock([['a', 'b'], ['1', '2']])
        pieces = _split_section(self._sec(['별첨 1) 단가표', big, '설명 문단.', small]),
                                SentenceSplitter(chunk_size=512, chunk_overlap=50))
        refs = [p.table for p in pieces if p.table is not None]
        assert len(refs) >= 2 and len({r.block for r in refs}) == 1            # 묶음 전부 같은 표
        assert refs[0].text == '별첨 1) 단가표\n' + big.render()              # 표 위 한 줄 + 헤더 한 번 + 전체 행
        assert pieces[-1].table is None and small.render() in pieces[-1].text   # 작은 표는 한 조각 안에 이미 다 있다

    def test_chunk_file은_문서_안_표_순번을_table_key로_싣는다(self, monkeypatch, tmp_path):
        import rag.chunking as C
        from rag.chunking import TableBlock
        t1, t2 = TableBlock(self._long()), TableBlock(self._long(head=('번호', '품목', '원가')))
        sections = [self._sec(['1장 본문', t1]), self._sec(['2장 본문', t2, '끝.', TableBlock([['a'], ['1']])])]
        sections[1].heading_path = ['문서', '2장']                             # 묶이지 않게 다른 경로
        monkeypatch.setitem(C._SECTION_PARSERS, '.pdf', lambda _p: sections)
        monkeypatch.setattr(C, '_pack_sections', lambda secs: secs)
        chunks = C.chunk_file(tmp_path / 'x.pdf')
        tables = [c for c in chunks if c.meta and c.meta.get('is_table')]
        assert {c.meta['table_key'] for c in tables} == {'t1', 't2'}
        assert {c.meta['table_text'] for c in tables if c.meta['table_key'] == 't2'} == {'2장 본문\n' + t2.render()}
        assert all(not (c.meta or {}).get('is_table') for c in chunks if '| a |' in c.text)   # 작은 표 청크는 표 묶음이 아니다

    def test_상한을_넘는_표는_TableTooLargeError_결정적_실패(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import TABLE_MAX_TOKENS, TableTooLargeError, TableBlock, _split_section
        from rag.outbox import is_transient
        from rag.tokens import estimate_tokens_conservative
        rows = self._long(400)
        assert estimate_tokens_conservative(TableBlock(rows).render()) > TABLE_MAX_TOKENS   # 시험 표가 실제로 상한 위
        with pytest.raises(TableTooLargeError) as e:
            _split_section(self._sec([TableBlock(rows, [7] * len(rows))], page=7),
                           SentenceSplitter(chunk_size=512, chunk_overlap=50))
        msg = str(e.value)
        assert msg.startswith("7쪽 표('번호 | 품목 | 단가')가 허용 크기의 ") and '(현재 400행 × 3열)' in msg
        assert '토큰' not in msg                                              # 엑셀 거절 문구 규칙(#222)
        assert not is_transient(e.value)                                      # 재시도 없이 바로 failed(#257 결정 3)

    def test_상한_바로_아래_표는_통과한다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import TABLE_MAX_TOKENS, TableBlock, _split_section
        from rag.tokens import estimate_tokens_conservative
        n = 60
        while estimate_tokens_conservative(TableBlock(self._long(n + 1)).render()) <= TABLE_MAX_TOKENS:
            n += 1
        pieces = _split_section(self._sec([TableBlock(self._long(n))]), SentenceSplitter(chunk_size=512, chunk_overlap=50))
        assert pieces and all(p.table is not None for p in pieces)

    def test_DOCX처럼_쪽이_없으면_표_이름만_말한다(self):
        from rag.chunking import TableTooLargeError
        assert str(TableTooLargeError(None, ['구분', '', '금액'], 150, 500, 3)).startswith("표('구분 | 금액')가 허용 크기의 150%")


# ── 표 전체에 싣는 표 앞뒤 글 (#268) ──────────────────────────────────────────────────

class TestTableContext:
    def test_앞_설명은_끝에서_최대_3줄이고_제목에서_멈춘다(self):
        from rag.chunking import _table_lead_lines
        text = '제5조(단가)\n첫째 줄.\n둘째 줄.\n셋째 줄.\n넷째 줄.'
        assert _table_lead_lines(text, set()) == ['둘째 줄.', '셋째 줄.', '넷째 줄.']
        assert _table_lead_lines('제5조(단가)\n유일한 설명.', set()) == ['유일한 설명.']        # 조 제목 꼴은 안 가져온다
        assert _table_lead_lines('2. 반품 규정\n설명.', {'2. 반품 규정'}) == ['설명.']            # 기록된 헤딩도 멈춘다

    def test_앞_설명은_합계_200자를_넘기지_않는다(self):
        from rag.chunking import _TABLE_CONTEXT_CHARS, _table_lead_lines
        assert _TABLE_CONTEXT_CHARS == 200                     # B2 실측 문단 196자를 덮는 최소선(#268 결정 3)
        long = '가' * 150
        assert _table_lead_lines(f'{long}\n{long}', set()) == [long]
        assert _table_lead_lines('가' * 196, set()) == ['가' * 196]
        assert _table_lead_lines('가' * (_TABLE_CONTEXT_CHARS + 1), set()) == []

    def test_아래_주석은_표지로_시작하는_줄이_이어지는_동안만(self):
        from rag.chunking import _table_note_lines
        text = '※ 단위는 퍼밀이다.\n* 재검사는 1회.\n주) 부가세 별도.\n(단위: 원)\n불합격 업체는 공지한다.'
        assert _table_note_lines(text) == ['※ 단위는 퍼밀이다.', '* 재검사는 1회.', '주) 부가세 별도.']   # 최대 3줄
        assert _table_note_lines('불합격 업체는 공지한다.\n※ 뒤에 와도 안 가져온다') == []
        assert _table_note_lines('단위: kg당 원\n출처: 구매팀') == ['단위: kg당 원', '출처: 구매팀']
        assert _table_note_lines('**굵은 글씨 문장**') == []                                         # 마크다운 강조는 표지가 아니다
        assert _table_note_lines(f'※ {"가" * 120}\n※ 짧은 주석') == []                              # 합계 100자 상한(#268 리뷰 1)
        assert _table_note_lines('※ 짧은 주석\n※ ' + '가' * 120) == ['※ 짧은 주석']

    def test_큰_표의_표_전체에만_앞뒤_글이_붙고_묶음_본문은_그대로다(self):
        from llama_index.core.node_parser import SentenceSplitter
        from rag.chunking import Section, TableBlock, _split_section
        rows = [['번호', '품목', '단가']] + [[str(i), f'품목-{i:03d}', f'{i * 1234 % 97000 + 1000:,}원'] for i in range(1, 61)]
        block = TableBlock(rows)
        intro, notes = '아래 단가는 2027년 1월 1일부터 적용한다.\n단가 단위는 kg당 원이다.', '※ 부가세는 포함하지 않는다.\n그 밖의 안내 문장.'
        sec = Section(['문서'], None, f'{intro}\n{block.render()}\n{notes}', None, blocks=[intro, block, notes])
        pieces = _split_section(sec, SentenceSplitter(chunk_size=512, chunk_overlap=50))
        batches = [p for p in pieces if p.table is not None]
        whole = batches[0].table.text
        assert whole == '\n'.join([*intro.split('\n'), block.render(), '※ 부가세는 포함하지 않는다.'])
        assert all(p.text.startswith('단가 단위는 kg당 원이다.\n| 번호 |') for p in batches)   # 묶음 머리는 예전처럼 한 줄
        assert not any('※' in p.text for p in batches)                                      # 주석은 묶음 본문에 안 들어간다
        assert any('※ 부가세는 포함하지 않는다.' in p.text for p in pieces if p.table is None)  # 원래 자리에도 그대로
