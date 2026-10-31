"""#244 배치 유형 명세 — 합성 PDF의 **정답 읽기 순서**의 정의점. 내용은 전부 지어낸 것이다.

실문서에는 다단·그림 감싸기·가로 쪽 혼합 배치가 없어(개발계 PDF 99건 전수 렌더 육안, 2026-10-22) 합성으로 검증한다. 상담센터에
올라올 법한 꼴을 고른다(사용자 지시 "흔한 것 포함"). 문단마다 고유 표식(`[P07]`)을 글 앞에 박아 docling 출력에서 순서를 되찾는다 —
표식은 본문처럼 읽히지 않게 짧고, 문장 안에 들어가 레이아웃 판정을 흔들지 않도록 문단 첫머리에만 둔다.

조판 수치는 실문서 규정(#240 실측)에 맞춘다: 11.6pt·줄 간격 13.8pt·문단 앞 6pt. #240 교훈 — 합성 PDF는 조판에 따라 docling 결과가
크게 갈리므로, 가장 흔한 2단만 단 간격을 좁게(6mm)·넓게(14mm) 두 변형으로 둔다.

각 spec: id / title / layout(렌더 종류) / blocks — 정답 순서대로 나열한 블록. 블록은 ('p', 글) 문단, ('h', 글) 소제목,
('table', rows) 표, ('img', alt) 그림, ('break',) 다음 프레임(단)으로. 표·그림은 순서 대조에서 "위치"만 본다(글 대조는 표 축 #246이 한다).
"""

from itertools import count

_REGIONS = ['서울', '경기', '강원', '충청', '전라', '경상', '제주']


def _paras(n: int, topic: str, start: int) -> list[tuple[str, str]]:
    """n개 문단 — 각 문단은 표식 + 두 문장(합성 규정 문체)."""
    out = []
    for k in range(n):
        i = start + k
        out.append(
            (
                'p',
                f'[P{i:02d}] {topic}에 관한 업무는 담당 부서가 맡으며, 처리 기한은 사유가 생긴 날부터 {i % 9 + 3}일 이내로 한다. '
                f'다만 금액이 {(i % 5 + 1) * 100}만 원을 넘으면 사무국장의 승인을 받는다.',
            )
        )
    return out


def _faq(n: int, start: int, qstart: int = 1) -> list[tuple[str, str]]:
    """Q/A 쌍 n개 — 표식은 질문·답변에 하나씩 연번(첫 측정에서 같은 번호를 둘에 붙여 거짓 '섞임'이 났다)."""
    out = []
    for k in range(n):
        i = start + 2 * k
        q = qstart + k
        out.append(
            (
                'h',
                f'[P{i:02d}] 질문 {q}. 정기 배송 {["건너뛰기", "해지", "결제일 변경", "주소 변경", "수량 변경", "일시정지"][(q - 1) % 6]}는 어떻게 하나요?',
            )
        )
        out.append(
            (
                'p',
                f'[P{i + 1:02d}] 앱의 구독 관리 화면에서 신청할 수 있으며, 배송 예정일 {q % 4 + 2}일 전까지 요청해야 당월에 반영된다. '
                f'그 뒤에 요청하면 다음 회차부터 적용된다.',
            )
        )
    return out


def _table(rows: int, start: int) -> tuple[str, list[list[str]]]:
    body = [
        [f'HN-{3000 + i}', f'품목 {i}호', _REGIONS[i % 7], f'{(i * 37) % 900 + 1200:,}']
        for i in range(start, start + rows)
    ]
    return ('table', [['품목 코드', '품목명', '산지', '납품 단가']] + body)


def _two_col(gap_mm: int, tag: str) -> dict:
    """① 2단 본문 — 왼쪽 단을 다 읽고 오른쪽 단으로. 쪽 2장."""
    c = count(1)
    blocks = []
    for page in range(2):
        for col in range(2):
            n = next(c)
            blocks.append(('h', f'[P{n:02d}] {page * 2 + col + 1}. 절 제목 {page * 2 + col + 1}'))
            blocks += _paras(4, f'항목 {page * 2 + col + 1}', next(c))
            for _ in range(3):
                next(c)
            blocks.append(('break',))
    return {
        'id': f'L01_two_col_{tag}',
        'title': '2단 본문 규정',
        'layout': 'two_col',
        'gap_mm': gap_mm,
        'blocks': blocks,
    }


def _three_col() -> dict:
    """② 3단 브로슈어 — 단마다 소제목 + 짧은 문단 셋. 쪽 1장(가로)."""
    c = count(1)
    blocks = []
    for col in range(3):
        n = next(c)
        blocks.append(('h', f'[P{n:02d}] {["상품 소개", "이용 방법", "문의 안내"][col]}'))
        blocks += _paras(3, ['제품', '신청', '상담'][col], next(c))
        for _ in range(2):
            next(c)
        blocks.append(('break',))
    return {'id': 'L02_three_col', 'title': '상품 안내 리플릿', 'layout': 'three_col', 'blocks': blocks}


def _image_wrap() -> dict:
    """③ 그림 옆으로 글이 흐르는 배치 — 그림(오른쪽) + 문단 셋이 옆에, 그 뒤 전폭 문단."""
    blocks = [('h', '[P01] 제품 구성'), ('img', '제품 구성도'), *_paras(3, '구성품', 2), *_paras(2, '보관', 5)]
    return {'id': 'L03_image_wrap', 'title': '제품 안내서', 'layout': 'image_wrap', 'blocks': blocks}


def _mixed_orientation() -> dict:
    """④ 세로 쪽 + 가로 쪽(넓은 표) + 세로 쪽 — 요금표가 가로 쪽에 붙은 안내서."""
    blocks = [
        ('h', '[P01] 1. 안내'),
        *_paras(3, '요금', 2),
        ('break',),
        ('h', '[P05] 2. 요금표'),
        ('p', '[P06] 아래 표는 품목별 납품 단가다.'),
        _table(12, 1),
        ('break',),
        ('h', '[P07] 3. 문의'),
        *_paras(2, '문의', 8),
    ]  # 표식 연번 P01~P09
    return {'id': 'L04_mixed_orient', 'title': '납품 단가 안내', 'layout': 'mixed_orient', 'blocks': blocks}


def _two_col_table() -> dict:
    """⑤ 2단 사이에 단을 가로지르는 표 — 위 2단 → 전폭 표 → 아래 2단."""
    blocks = [
        ('h', '[P01] 1. 개요'),
        *_paras(2, '개요', 2),
        ('break',),
        *_paras(2, '범위', 4),
        ('break',),
        ('p', '[P06] 품목별 단가는 다음과 같다.'),
        _table(6, 1),
        ('break',),
        ('h', '[P07] 2. 절차'),
        *_paras(2, '절차', 8),
        ('break',),
        *_paras(2, '예외', 10),
        ('break',),
    ]
    return {'id': 'L05_two_col_table', 'title': '납품 절차 규정', 'layout': 'two_col_table', 'blocks': blocks}


def _two_col_faq() -> dict:
    """⑥ 2단 FAQ — Q/A 쌍이 단 안에서 이어져야 한다(섞이면 Q와 A가 엇갈림)."""
    blocks = [('h', '[P01] 자주 묻는 질문'), *_faq(3, 2), ('break',), *_faq(3, 8, qstart=4), ('break',)]
    return {'id': 'L06_two_col_faq', 'title': '정기 배송 FAQ', 'layout': 'two_col', 'gap_mm': 10, 'blocks': blocks}


def _header_footer() -> dict:
    """⑦ 머리글·바닥글·쪽 번호가 있는 단일 단 — 머리글이 본문에 섞이면 안 된다(docling은 furniture로 뺀다)."""
    blocks = [('h', '[P01] 1. 목적'), *_paras(5, '목적', 2), ('h', '[P07] 2. 적용'), *_paras(6, '적용', 8)]
    return {
        'id': 'L07_header_footer',
        'title': '운영 지침',
        'layout': 'header_footer',
        'blocks': blocks,
        'furniture': ['사내 운영 지침 · 대외비', '쪽'],
    }


def _sidebar() -> dict:
    """⑧ 표 옆 설명 상자(사이드바) — 본문 2/3 + 오른쪽 1/3 상자. 상자는 본문 뒤에 와야 한다."""
    blocks = [
        ('h', '[P01] 요금 안내'),
        *_paras(3, '요금', 2),
        _table(5, 1),
        *_paras(2, '정산', 5),
        ('break',),
        ('h', '[P07] 주의'),
        *_paras(2, '주의 사항', 8),
    ]
    return {'id': 'L08_sidebar', 'title': '요금 안내서', 'layout': 'sidebar', 'blocks': blocks}


def _two_col_list() -> dict:
    """⑨ 2단 + 번호 목록이 단 경계를 넘는다 — 번호 순서가 지켜져야 한다."""
    items = [
        (
            'p',
            f'[P{i:02d}] {i - 1}. 단계 {i - 1}: 신청서를 확인하고 담당자에게 전달한 뒤 처리 결과를 {i % 3 + 1}일 안에 회신한다.',
        )
        for i in range(2, 12)
    ]
    blocks = [('h', '[P01] 처리 절차'), *items[:5], ('break',), *items[5:], ('break',)]
    return {'id': 'L09_two_col_list', 'title': '접수 처리 절차', 'layout': 'two_col', 'gap_mm': 10, 'blocks': blocks}


SPECS = [
    _two_col(6, 'narrow'),
    _two_col(14, 'wide'),
    _three_col(),
    _image_wrap(),
    _mixed_orientation(),
    _two_col_table(),
    _two_col_faq(),
    _header_footer(),
    _sidebar(),
    _two_col_list(),
    # 단 간격 경계값 — 2단 본문을 간격만 바꿔(리뷰: 경계 실험도 명세에 있어야 결과 파일로 남는다)
    _two_col(8, 'gap8'),
    _two_col(10, 'gap10'),
    _two_col(12, 'gap12'),
]
BY_ID = {s['id']: s for s in SPECS}


def paragraph_bodies(spec: dict) -> dict[str, str]:
    """표식 → 문단 본문(표식 뺀 글). 채점이 "본문이 요소·조각 안에 연속으로 남아 있나"를 보는 데 쓴다 — 표식 순서만 보면 단 두 개가
    가로로 한 줄씩 이어 읽힌 꼴(좁은 2단 실측)을 놓친다(리뷰 지적 1)."""
    import re

    return {
        m.group(1): b[1][m.end() :].strip()
        for b in spec['blocks']
        if b[0] in ('p', 'h')
        for m in [re.match(r'\[(P\d+)\]\s*', b[1])]
    }


def expected_order(spec: dict) -> list[str]:
    """정답 순서의 표식 목록 — 문단·소제목만(표·그림은 글 대조 밖)."""
    import re

    out = []
    for b in spec['blocks']:
        if b[0] in ('p', 'h'):
            out.append(re.match(r'\[(P\d+)\]', b[1]).group(1))
    return out
