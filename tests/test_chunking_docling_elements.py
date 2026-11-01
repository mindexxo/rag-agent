"""docling 요소→섹션 매핑 — 제목 층 복원·요소별 본문화·그림 캡션. 튜플·가짜 요소만 먹이므로 docling 없이 돈다."""

import pytest

from rag.chunking import count_picture_placeholders
from rag.chunking.docling_elements import (
    PICTURE_PLACEHOLDER,
    docling_elements,
    docling_norm,
    sections_from_docling_elements,
)


def _h(text, page=1):
    return ('section_header', text, page, None)


def _t(text, page=1):
    return ('text', text, page, None)


class TestHeadingLevels:
    """docling은 장·조를 같은 레벨로 낸다 — 한국 규정 번호 체계로 층을 복원해야 조항끼리 묶인다."""

    def test_제목_장_조_3단(self):
        secs = sections_from_docling_elements(
            [
                _h('주식회사 인티큐브 취업규칙'),
                _h('제 1 장 총 칙'),
                _h('제 1 조(목 적)'),
                _t('본문 A'),
                _h('제 2 조(적용범위)'),
                _t('본문 B'),
                _h('제 2 장 인 사'),
                _h('제 3 조(채용)'),
                _t('본문 C'),
            ]
        )
        assert [s.heading_path for s in secs] == [
            ['주식회사 인티큐브 취업규칙', '제 1 장 총 칙', '제 1 조(목 적)'],
            ['주식회사 인티큐브 취업규칙', '제 1 장 총 칙', '제 2 조(적용범위)'],
            ['주식회사 인티큐브 취업규칙', '제 2 장 인 사', '제 3 조(채용)'],
        ]

    def test_장이_없는_가이드_문서는_제목_아래_2단(self):
        secs = sections_from_docling_elements(
            [
                _h('[ 총무 | 가이드 ] 노트북 대여'),
                _h('■ 신청 절차'),
                _t('본문'),
                _h('■ 참고 사항'),
                _t('본문2'),
            ]
        )
        assert [s.heading_path for s in secs] == [
            ['[ 총무 | 가이드 ] 노트북 대여', '■ 신청 절차'],
            ['[ 총무 | 가이드 ] 노트북 대여', '■ 참고 사항'],
        ]

    def test_같은_장_아래_형제_조항은pack_sections가_묶을_수_있다(self):
        # joinable = 공통 조상 깊이 ≥2. 층 복원이 없으면(전부 L1) 공통 조상이 []라 아무것도 안 묶인다.
        from rag.chunking.sections import _common_prefix

        secs = sections_from_docling_elements(
            [_h('제목'), _h('제 1 장 총칙'), _h('제 1 조'), _t('a'), _h('제 2 조'), _t('b')]
        )
        assert len(_common_prefix([s.heading_path for s in secs])) == 2

    # ── 조 제목 보정 — 레이아웃 모델이 같은 꼴의 조 제목을 들쭉날쭉 text로 낸다 ──

    def test_text로_나온_조_제목은_제목으로_올린다(self):
        secs = sections_from_docling_elements(
            [
                _h('규정'),
                _h('제1조 (목 적)'),
                _t('목적 본문'),
                _t('제2조 (적 용)'),
                _t('적용 본문'),
                _t('제3조（회계단위（본사））'),
                _t('단위 본문'),
            ]
        )  # 전각·중첩 괄호도
        assert [s.heading_path for s in secs] == [
            ['규정', '제1조 (목 적)'],
            ['규정', '제2조 (적 용)'],
            ['규정', '제3조（회계단위（본사））'],
        ]
        assert secs[1].body == '적용 본문'

    def test_장_아래에서_올린_조_제목은_조_층(self):
        secs = sections_from_docling_elements([_h('규정'), _h('제 1 장 총칙'), _t('제 1 조(목적)'), _t('본문')])
        assert secs[0].heading_path == ['규정', '제 1 장 총칙', '제 1 조(목적)']

    def test_제목과_첫_문장이_한_요소면_떼어서_제목과_본문으로(self):
        # '제1조 (목적) 이 규정은 …' — 조 제목 뒤 간격이 좁으면 docling이 한 덩어리로 낸다
        secs = sections_from_docling_elements(
            [
                _h('규정'),
                _h('제1조 (목적)'),
                _t('본문'),
                _t('제24조 (자금의 관리) 회사의 자금은 재무팀장이 총괄한다.'),
                _t('둘째 줄'),
            ]
        )
        assert secs[-1].heading_path == ['규정', '제24조 (자금의 관리)']
        assert secs[-1].body == '회사의 자금은 재무팀장이 총괄한다.\n둘째 줄'

    def test_알려진_한계_괄호_뒤_공백_인용으로_시작하는_문단은_제목이_된다(self):
        # DOCX 폴백과 같은 규칙·같은 위험 — 글은 남는다. 이 동작이 바뀌면 이 단언을 고친다
        secs = sections_from_docling_elements(
            [_h('규정'), _h('제1조 (목적)'), _t('본문'), _t('제3조(보상) 규정에 따라 지급한다')]
        )
        assert secs[-1].heading_path == ['규정', '제3조(보상)'] and secs[-1].body == '규정에 따라 지급한다'

    @pytest.mark.parametrize(
        'line',
        [
            '제10조(보관)에 따라 보관한다',  # 닫는 괄호에 조사 — 인용
            '제 10 조 3호에 의한 특별 공로 퇴직금',  # 괄호 없음 — 인용(실문서 퇴직금 규정)
            "제98조(이하 '보상조항'이라 한다)에 …",  # 정의 꼴
        ],
    )
    def test_조_인용_문장은_올리지_않는다(self, line):
        secs = sections_from_docling_elements([_h('규정'), _h('제1조 (목적)'), _t('본문'), _t(line)])
        assert [s.heading_path for s in secs] == [['규정', '제1조 (목적)']] and line in secs[0].body

    def test_문서_제목보다_먼저_나온_조_제목_꼴은_올리지_않는다(self):
        # 문서 제목(첫 section_header) 전의 글은 본문으로 둔다 — 올리면 '규정'이 조 아래 헤딩으로 밀려 제목 경로가 틀어진다
        secs = sections_from_docling_elements([_t('제1조 (목적)'), _h('규정'), _h('제2조 (적용)'), _t('본문')])
        assert [(s.heading_path, s.body) for s in secs] == [([], '제1조 (목적)'), (['규정', '제2조 (적용)'], '본문')]

    def test_닫는_괄호가_다음_요소로_떨어진_조_제목은_잇는다(self):
        # 잇지 않으면 ')'가 따로 제목이 되어 제목 경로가 [... , ')']가 된다
        secs = sections_from_docling_elements(
            [
                _h('징계규정'),
                _h('제 1 장 총 칙'),
                _h('제 7 조 ( 징계종류'),
                _h(')'),
                _t('징계의 종류는 5 종으로 한다 .'),
                _h('제 8 조 ( 양정기준'),
                _t('）'),
                _t('양정 본문'),
            ]
        )
        assert [s.heading_path for s in secs] == [
            ['징계규정', '제 1 장 총 칙', '제 7 조 ( 징계종류 )'],
            ['징계규정', '제 1 장 총 칙', '제 8 조 ( 양정기준 ）'],
        ]

    def test_괄호가_떨어진_조_제목이_text로_나와도_잇고_올린다(self):
        # 제목 쪽이 text로 오면(①이 올릴 꼴) ②가 먼저 잇는다 — 안 그러면 ')'가 본문 첫 줄로 남는다
        secs = sections_from_docling_elements([_h('규정'), _t('제 7 조 ( 징계종류'), _t(')'), _t('본문')])
        assert [(s.heading_path, s.body) for s in secs] == [(['규정', '제 7 조 ( 징계종류 )'], '본문')]

    def test_괄호_하나만_있는_요소라도_조_제목_뒤가_아니면_건드리지_않는다(self):
        secs = sections_from_docling_elements([_h('규정'), _h('■ 안내'), _t(')'), _t('본문')])
        assert secs[0].heading_path == ['규정', '■ 안내'] and secs[0].body.startswith(')')

    def test_장_패턴_변형(self):
        from rag.chunking.sections import CHAPTER_RE

        assert CHAPTER_RE.match('제 3 장 복 무') and CHAPTER_RE.match('제3장 총칙') and CHAPTER_RE.match('제 12 장')
        assert not CHAPTER_RE.match('제 3 조(용어)') and not CHAPTER_RE.match('장기근속휴가')


class TestElementMapping:
    def test_표는_markdown_그대로_본문에_들어간다(self):
        md = '| 구 분 | 무상수리 |\n| --- | --- |\n| 침수 | 없음 |'
        secs = sections_from_docling_elements([_h('제목'), _h('■ 기준'), ('table', '', 1, md)])
        assert secs[0].body == md
        assert secs[0].line_pages == [1, 1, 1]  # 표 3줄 ↔ 쪽 3개 — 1:1 대응 (결함 6)

    def test_그림은_캡션이_없으면_자리표시로_남긴다(self):
        secs = sections_from_docling_elements([_h('제목'), _h('(1) 프로세스'), ('picture', '', 1, None), _t('설명')])
        assert secs[0].body.split('\n') == [PICTURE_PLACEHOLDER, '설명']

    def test_헤더_푸터는_요소로_오지_않는다는_전제(self):
        # docling은 page_header/page_footer를 iterate_items()에 넣지 않는다 — 그 전제만 고정한다.
        secs = sections_from_docling_elements([_h('제목'), _t('본문')])
        assert secs[0].body == '본문'

    def test_NBSP_정규화(self):
        assert docling_norm('■ 신청\xa0절차') == '■ 신청 절차'
        assert docling_norm('\xa0\xa0개인부담\xa0') == '개인부담'

    def test_목록_항목은_번호와_글머리를_text에_되붙인다(self):
        # docling ListItem은 '2.1'·'-' 같은 marker를 text에서 뗀다. 안 붙이면 소항목 번호가 색인에서 사라진다.
        from types import SimpleNamespace as NS

        def el(label, text, marker=None):
            return NS(label=NS(value=label), text=text, marker=marker, prov=[NS(page_no=1)])

        doc = NS(
            iterate_items=lambda: [
                (el('section_header', '2. 주요 개정 내용'), 0),
                (el('list_item', '얼리버드 할인율 상향', '2.1'), 1),
                (el('list_item', '정가의 15%로 상향한다.', '-'), 2),
                (el('list_item', '마커 없는 이어지는 줄', ''), 2),
                (el('text', '본문', None), 0),
            ]
        )
        assert [t for _, t, _, _ in docling_elements(doc)] == [
            '2. 주요 개정 내용',
            '2.1 얼리버드 할인율 상향',
            '- 정가의 15%로 상향한다.',
            '마커 없는 이어지는 줄',
            '본문',
        ]

    def test_빈_헤딩과_빈_텍스트는_건너뛴다(self):
        secs = sections_from_docling_elements([_h('제목'), _h('\xa0'), _t(''), _t('본문')])
        assert [s.heading_path for s in secs] == [['제목']] and secs[0].body == '본문'


class TestPictureCaption:
    """그림 캡션이 자리표시를 대체한다. 튜플만 먹이므로 VLM·모델 없이 돈다."""

    def test_캡션이_있으면_자리표시_대신_캡션이_본문에_들어간다(self):
        cap = '4단계 흐름도: 신청 접수 → 자격 심사 → 매칭 승인 → 집행·보고'
        secs = sections_from_docling_elements([_h('제목'), _h('(1) 절차'), ('picture', cap, 1, None), _t('설명')])
        assert secs[0].body.split('\n') == [cap, '설명']

    def test_캡션이_비면_자리표시로_되돌아간다(self):
        # VLM이 죽거나 꺼져 있으면 docling이 캡션을 안 만든다(예외 없음) — 그때의 경로다.
        secs = sections_from_docling_elements([_h('제목'), ('picture', '', 1, None)])
        assert secs[0].body == PICTURE_PLACEHOLDER

    def test_캡션이_공백뿐이면_자리표시로_되돌아간다(self):
        secs = sections_from_docling_elements([_h('제목'), ('picture', '\xa0  ', 1, None)])
        assert secs[0].body == PICTURE_PLACEHOLDER

    def test_그림마다_캡션_유무가_달라도_각각_처리된다(self):
        secs = sections_from_docling_elements(
            [
                _h('제목'),
                _h('절'),
                ('picture', '그림1 설명', 1, None),
                _t('본문1'),
                ('picture', '', 1, None),
                _t('본문2'),
            ]
        )
        assert secs[0].body.split('\n') == ['그림1 설명', '본문1', PICTURE_PLACEHOLDER, '본문2']

    def test_캡션은_annotations에서_꺼낸다(self):
        # 캡션은 element.text가 아니라 annotations에 붙는다. text만 보면 조용히 버려진다.
        from types import SimpleNamespace as NS

        def pic(annotations):
            return NS(label=NS(value='picture'), text='', marker=None, annotations=annotations, prov=[NS(page_no=1)])

        doc = NS(
            iterate_items=lambda: [
                (pic([NS(text='막대그래프: 면 100% 40℃, 린넨 30℃')]), 0),
                (pic([]), 1),
                (pic(None), 2),
                (pic([NS(text=''), NS(text='두 번째가 살아있다')]), 3),
            ]
        )
        assert [t for _, t, _, _ in docling_elements(doc)] == [
            '막대그래프: 면 100% 40℃, 린넨 30℃',
            '',
            '',
            '두 번째가 살아있다',
        ]

    def test_자리표시_집계는_캡션_실패_수를_센다(self):
        # 지표(kms_index_picture_placeholder_total)의 입력. 캡션 성공분은 일반 본문과 구분되지
        # 않아 셀 수 없다 — 이 함수는 성공률이 아니라 실패 신호다.
        from types import SimpleNamespace as NS

        chunks = [
            NS(text=f'앞 {PICTURE_PLACEHOLDER} 뒤'),
            NS(text='캡션이 붙은 그림 설명'),
            NS(text=f'{PICTURE_PLACEHOLDER}\n{PICTURE_PLACEHOLDER}'),
        ]
        assert count_picture_placeholders(chunks) == 3
        assert count_picture_placeholders([]) == 0
