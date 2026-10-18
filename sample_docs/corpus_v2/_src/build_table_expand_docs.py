"""#257 큰 표 확장 문서 생성 — '표 전체를 봐야 답이 맞는' 질문용 합성 문서 2건(재현 가능한 정본).

실문서의 큰 표(묶음 2개 이상)는 약 10개이고 전부 규정·서식이라 최댓값·개수·전체 나열 질문의 재료가 모자라다(#257 실측).
그래서 상담 현장에 실제로 있을 법한 '목록형 큰 표'를 앞뒤 문맥과 함께 만든다:
  - aromanica_17 (PDF):  오프라인 매장 60곳 — 쪽을 넘기는 표(쪽마다 헤더 반복 → 한 표로 이어진 뒤 행 묶음 4개 안팎)
  - homeplus_18 (DOCX):  문화센터 강좌 60개 — 행 묶음 4개 안팎
2차(측정 보강): 위 둘은 묶음이 top-5 안에 다 들어가 '이전'도 표 전체를 봤다(효과가 안 드러남). 그래서 효과가 나는 두 조건을 만든다:
  - summers_17 (PDF):    A/S 접수처 96곳 — 상한(6,000토큰) 근처라 묶음 6개(top-5에 다 못 들어감). 같은 테넌트의 품질보증·A/S 기준
                         (summers_04)·각인 주문(summers_06)이 질문 어휘('A/S'·'각인'·'수리 기간')에 같이 걸려 칸을 다툰다
  - harim_16 (DOCX):     협력농가 96곳 — 같은 조건. 협력농가 점검 일정(harim_15)·품질인증(harim_07)이 '협력농가'·'인증'에 같이 걸린다
정답이 되는 행은 **서로 다른 묶음에 흩어 둔다** — 묶음 하나만 보면 틀리게(최댓값이 뒷 묶음, 개수 대상이 묶음마다 하나씩).
정답의 유일성(최댓값이 하나뿐 등)은 아래 assert가 지킨다. 골드 문항은 eval/gold_set_v2.jsonl의 *_te00N.

표 그리기는 eval/table_parsing/build.py의 함수를 그대로 쓴다(#252 build_table_docs.py와 같은 방식).

사용: python sample_docs/corpus_v2/_src/build_table_expand_docs.py   → sample_docs/corpus_v2/<tenant>/ 에 산출
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from docx import Document as DocxDocument                          # noqa: E402
from reportlab.lib.pagesizes import A4                             # noqa: E402
from reportlab.lib.styles import ParagraphStyle                    # noqa: E402
from reportlab.lib.units import mm                                 # noqa: E402
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer  # noqa: E402

from eval.table_parsing.build import _BODY, _docx_fill, _rl_table   # noqa: E402

OUT = Path(__file__).resolve().parent.parent
_H1 = ParagraphStyle('h1', fontName='HYGothic-Medium', fontSize=16, leading=22, spaceAfter=6)
_H2 = ParagraphStyle('h2', fontName='HYGothic-Medium', fontSize=13, leading=18, spaceBefore=8, spaceAfter=4)
_CAP = ParagraphStyle('cap', fontName='HYGothic-Medium', fontSize=9, leading=12, alignment=1, spaceBefore=2, spaceAfter=6)

# ── aromanica_17 매장 목록 ─────────────────────────────────────────────────────────

_STORES = [   # (지역, 매장명) — 지역별로 모아 적는 실제 매장 안내의 순서
    ('서울', '성수'), ('서울', '연남'), ('서울', '잠실'), ('서울', '강남역'), ('서울', '여의도'), ('서울', '신촌'),
    ('서울', '건대'), ('서울', '목동'), ('서울', '노원'), ('서울', '은평'), ('서울', '합정'), ('서울', '망원'),
    ('서울', '이태원'), ('경기', '판교'), ('경기', '광교'), ('경기', '일산'), ('경기', '정자'), ('경기', '동탄'),
    ('경기', '평택'), ('경기', '하남'), ('경기', '김포'), ('경기', '의정부'), ('경기', '광명'), ('경기', '안양'),
    ('경기', '수지'), ('인천', '송도'), ('인천', '청라'), ('인천', '부평'), ('인천', '구월'), ('부산', '서면'),
    ('부산', '해운대'), ('부산', '광안리'), ('부산', '센텀'), ('부산', '동래'), ('부산', '남포'), ('대구', '동성로'),
    ('대구', '수성'), ('대구', '칠곡'), ('광주', '충장로'), ('광주', '상무'), ('대전', '둔산'), ('대전', '유성'),
    ('울산', '삼산'), ('세종', '나성'), ('강원', '춘천'), ('강원', '강릉'), ('강원', '원주'), ('충북', '청주'),
    ('충남', '천안'), ('충남', '아산'), ('전북', '전주'), ('전남', '여수'), ('전남', '순천'), ('경북', '포항'),
    ('경북', '경주'), ('경남', '창원'), ('경남', '김해'), ('경남', '진주'), ('제주', '제주시청'), ('제주', '서귀포'),
]
_OUTLET = {'의정부', '동래', '칠곡', '김해'}                 # 아울렛 입점 4곳 — 묶음마다 흩어 둔다
_DEPT = {'잠실', '여의도', '목동', '판교', '광교', '송도', '센텀', '수성', '둔산', '천안', '창원'}
_TUESDAY = {'신촌', '부평', '전주'}                          # 매주 화요일 휴무 3곳
_LATEST = '해운대'                                          # 평일 마감이 가장 늦은 매장(22:30) — 하나뿐
_NO_PARKING = {'연남', '망원', '광안리', '남포', '제주시청'}   # 부산에서는 광안리·남포 둘
_LATE_2200 = {'강남역', '홍대', '건대', '서면', '동성로', '충장로'}


def store_rows() -> list[list[str]]:
    rows = [['번호', '매장명', '지역', '유형', '평일 영업시간', '정기 휴무', '주차']]
    for i, (region, name) in enumerate(_STORES, start=1):
        kind = '아울렛 입점' if name in _OUTLET else '백화점 입점' if name in _DEPT else '직영'
        if name == _LATEST:
            hours = '10:30~22:30'
        elif name in _LATE_2200:
            hours = '10:30~22:00'
        elif kind == '백화점 입점':
            hours = '10:30~20:00'
        elif kind == '아울렛 입점':
            hours = '10:30~21:00'
        else:
            hours = '11:00~21:00'
        holiday = ('매주 화요일' if name in _TUESDAY else '백화점 휴점일' if kind == '백화점 입점'
                   else '아울렛 휴점일' if kind == '아울렛 입점' else '연중무휴')
        parking = ('불가' if name in _NO_PARKING else '건물 주차장 3시간 무료' if kind != '직영'
                   else '구매 시 1시간 무료')
        rows.append([str(i), f'아로마니카 {name}점', region, kind, hours, holiday, parking])
    body = rows[1:]
    assert len(body) == 60
    assert [r[4] for r in body].count('10:30~22:30') == 1                         # 최댓값 유일
    assert sum(r[5] == '매주 화요일' for r in body) == 3
    assert sum(r[3] == '아울렛 입점' for r in body) == 4
    return rows


def build_aromanica() -> Path:
    p = OUT / 'aromanica' / 'aromanica_17_오프라인매장안내.pdf'
    P = lambda t: Paragraph(t, _BODY)                                # noqa: E731
    story = [
        Paragraph('오프라인 매장 이용 안내 (ARO-CS-017)', _H1),
        P('개정 2026. 10. 1. / 주관: 리테일운영팀 / 고객센터 상담용'),
        Paragraph('1. 안내 목적', _H2),
        P('이 문서는 고객이 오프라인 매장의 위치, 영업시간, 휴무일, 주차 가능 여부를 문의할 때 상담사가 같은 기준으로 안내하도록 '
          '매장 정보를 한곳에 정리한 것이다. 매장 정보가 바뀌면 리테일운영팀이 매월 1일에 이 문서를 갱신한다.'),
        Paragraph('2. 매장 유형별 운영 원칙', _H2),
        P('직영 매장은 본사가 직접 운영하며 연중무휴를 원칙으로 한다. 백화점·아울렛 입점 매장은 입점한 점포의 영업시간과 휴점일을 '
          '따르므로, 휴점일 문의에는 해당 점포의 공지를 함께 확인하도록 안내한다. 매장별로 영업시간이 다르므로 고객에게는 반드시 '
          '아래 목록의 시간을 안내하고, 임의로 일반적인 시간을 말하지 않는다.'),
        Paragraph('3. 매장 목록', _H2),
        P('2026년 10월 기준 운영 중인 매장은 다음과 같다.'),          # 큰 표의 도입 줄 — 표 바로 앞 마지막 줄
        Spacer(1, 4),
        _rl_table(store_rows()),                                    # 쪽마다 헤더 반복(repeatRows=1)
        Paragraph('[표 1] 오프라인 매장 목록 (2026년 10월)', _CAP),
        P('주차 지원은 매장 영수증을 제시한 고객에게만 적용하며, 주차 "불가" 매장은 인근 공영주차장을 안내한다.'),
        Paragraph('4. 임시 휴무와 영업시간 변경', _H2),
        P('명절 당일과 매장 공사 기간에는 임시로 휴무하거나 영업시간을 줄일 수 있다. 임시 변경은 공식 홈페이지 공지사항과 매장 '
          '안내문으로 알리며, 상담사는 공지사항을 먼저 확인한 뒤 안내한다.'),
        Paragraph('5. 문의', _H2),
        P('매장 정보가 목록과 다르다는 고객 제보를 받으면 상담 이력에 매장명과 내용을 남기고 리테일운영팀에 전달한다.'),
    ]
    SimpleDocTemplate(str(p), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, title='오프라인 매장 이용 안내').build(story)
    return p


# ── homeplus_18 문화센터 강좌 ──────────────────────────────────────────────────────

_BRANCHES = ['잠실점', '강서점', '영등포점', '일산점', '수원점', '부산센텀점']
_CLASSES = [   # (강좌명, 대상, 기본 수강료) — 3개월 기준
    ('수채화 드로잉', '성인', 120000), ('왕초보 기타', '성인', 150000), ('홈베이킹', '성인', 165000),
    ('플라워 클래스', '성인', 180000), ('필라테스', '성인', 135000), ('요가', '성인', 105000),
    ('창의 미술', '유아(4~7세)', 99000), ('발레', '유아(4~7세)', 117000), ('키즈 쿠킹', '유아(4~7세)', 126000),
    ('로봇 코딩', '초등', 159000), ('보드게임 수학', '초등', 108000), ('주말 축구', '초등', 96000),
    ('스마트폰 활용', '시니어', 60000), ('라인댄스', '시니어', 75000),
]
_DAYS = ['월', '화', '수', '목', '금']
_TIMES = ['10:30~12:00', '13:00~14:30', '15:00~16:30', '19:00~20:30']


def class_rows() -> list[list[str]]:
    rows = [['강좌번호', '강좌명', '점포', '대상', '요일', '시간', '수강료(3개월)', '정원']]
    n = 0
    for b, branch in enumerate(_BRANCHES):
        for c in range(10):
            n += 1
            name, target, fee = _CLASSES[(b * 3 + c) % len(_CLASSES)]
            day = _DAYS[(b + c) % 5]
            # 초등 강좌의 토요일 반은 정확히 3개 — 점포가 달라 묶음이 갈린다(아래 덮어쓰기)
            cap = 6 if target == '유아(4~7세)' else 12 if target == '초등' else 15
            rows.append([f'HC-{1000 + n}', name, branch, target, day, _TIMES[(b + c) % 4], f'{fee + b * 3000:,}원', f'{cap}명'])
    # 정답 행을 심는다 — 서로 다른 점포(=다른 묶음)에 흩는다
    rows[2] = ['HC-1002', '주말 축구', '잠실점', '초등', '토', '10:30~12:00', '96,000원', '12명']
    rows[25] = ['HC-1025', '로봇 코딩', '영등포점', '초등', '토', '13:00~14:30', '165,000원', '12명']
    rows[53] = ['HC-1053', '보드게임 수학', '부산센텀점', '초등', '토', '10:30~12:00', '123,000원', '12명']
    rows[47] = ['HC-1047', '와인 소믈리에 입문', '수원점', '성인', '금', '19:00~20:30', '210,000원', '10명']   # 수강료 최고(유일)
    rows[14] = ['HC-1014', '1:2 골프 레슨', '강서점', '성인', '수', '19:00~20:30', '195,000원', '2명']        # 정원 최소(유일)
    body = rows[1:]
    fees = [int(r[6].replace(',', '').removesuffix('원')) for r in body]
    caps = [int(r[7].removesuffix('명')) for r in body]
    assert len(body) == 60 and len({r[0] for r in body}) == 60 and all(r[0] == f"HC-{1000 + i}" for i, r in enumerate(body, 1))
    assert fees.count(max(fees)) == 1 and max(fees) == 210000
    assert caps.count(min(caps)) == 1 and min(caps) == 2
    assert sum(r[3] == '초등' and r[4] == '토' for r in body) == 3
    return rows


def build_homeplus() -> Path:
    p = OUT / 'homeplus' / 'homeplus_18_문화센터강좌안내.docx'
    d = DocxDocument()
    d.add_heading('문화센터 가을학기 강좌 안내 (HP-CUL-018)', level=1)
    d.add_paragraph('학기: 2026년 10월 ~ 12월(3개월) | 주관: 문화센터운영팀 | 고객센터 상담용')
    d.add_heading('1. 수강 신청', level=2)
    d.add_paragraph('수강 신청은 홈플러스 앱의 문화센터 메뉴 또는 점포 문화센터 데스크에서 할 수 있다. 정원이 차면 대기 접수를 받고, '
                    '취소자가 생기면 대기 순서대로 문자로 안내한다. 같은 강좌를 다른 점포에서 동시에 신청할 수는 없다.')
    d.add_heading('2. 강좌 목록', level=2)
    d.add_paragraph('점포별 가을학기 강좌는 아래와 같다. 수강료는 3개월 기준이며 재료비는 강좌별로 별도이다.')
    rows = class_rows()
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    _docx_fill(t, rows)
    cap = d.add_paragraph('[표 1] 2026 가을학기 문화센터 강좌 목록')
    cap.alignment = 1
    d.add_heading('3. 환불 기준', level=2)
    d.add_paragraph('개강 전 취소는 수강료 전액을 환불한다. 개강 후에는 1개월 경과 전 3분의 2, 2개월 경과 전 3분의 1을 환불하며, '
                    '2개월이 지나면 환불하지 않는다. 재료비는 재료를 받기 전에 취소한 경우에만 환불한다.')
    d.add_heading('4. 휴강', level=2)
    d.add_paragraph('강사 사정으로 휴강하면 보강일을 문자로 안내한다. 점포 휴무일에는 강좌도 쉰다.')
    d.save(p)
    return p


# ── summers_17 A/S 접수처 (상한 근처 · 칸 다툼) ─────────────────────────────────────

_AS_REGIONS = ['서울'] * 22 + ['경기'] * 18 + ['인천'] * 6 + ['강원'] * 5 + ['충북'] * 4 + ['충남'] * 5 + ['대전'] * 4 + \
              ['전북'] * 5 + ['전남'] * 5 + ['광주'] * 4 + ['경북'] * 5 + ['대구'] * 4 + ['경남'] * 5 + ['부산'] * 2 + ['제주'] * 2
_AS_ENGRAVE = {7, 33, 61, 88}           # 각인 수리 가능 협력 수선점 — 묶음 1·2·4·6에 하나씩
_AS_DEPT_SAT = {12, 47, 79}             # 토요일 접수하는 백화점 A/S 데스크
_AS_LONGEST = 91                        # 평균 수리 기간 최장(14일) — 마지막 묶음


def as_rows() -> list[list[str]]:
    rows = [['번호', '접수처', '지역', '유형', '토요일 접수', '각인 수리', '평균 수리 기간']]
    for i, region in enumerate(_AS_REGIONS, start=1):
        kind = '백화점 A/S 데스크' if i % 6 == 0 or i in _AS_DEPT_SAT else '협력 수선점' if i % 3 == 1 else '직영 매장'
        sat = '접수' if (kind != '백화점 A/S 데스크' and i % 2 == 0) or i in _AS_DEPT_SAT else '미접수'
        engrave = '가능' if i in _AS_ENGRAVE else '불가'
        days = '14일' if i == _AS_LONGEST else f'{(i * 7) % 6 + 5}일'          # 그 밖은 5~10일
        rows.append([str(i), f'썸머스 {region} A/S {i:02d}', region, kind, sat, engrave, days])
    for i in _AS_ENGRAVE:
        rows[i][3] = '협력 수선점'
    body = rows[1:]
    assert len(body) == 96 and [r[6] for r in body].count('14일') == 1
    assert sum(r[5] == '가능' for r in body) == 4 and all(r[3] == '협력 수선점' for r in body if r[5] == '가능')
    assert sorted(int(r[0]) for r in body if r[3] == '백화점 A/S 데스크' and r[4] == '접수') == sorted(_AS_DEPT_SAT)
    assert [r[0] for r in body if r[2] == '제주'] == ['95', '96']
    return rows


def build_summers() -> Path:
    p = OUT / 'summers' / 'summers_17_AS접수처안내.pdf'
    P = lambda t: Paragraph(t, _BODY)                                # noqa: E731
    story = [
        Paragraph('A/S 접수처 안내 (SUM-CS-017)', _H1),
        P('개정 2026. 10. 1. / 주관: 고객지원팀 / 고객센터 상담용'),
        Paragraph('1. 안내 목적', _H2),
        P('제품 수리를 원하는 고객에게 가까운 A/S 접수처를 안내하기 위한 목록이다. 수리 기준·보증 기간은 품질보증 및 A/S 기준을, '
          '각인 신규 주문은 각인 주문 제작 안내를 따른다. 이 목록은 접수처의 운영 정보만 다룬다.'),
        Paragraph('2. 접수처 유형', _H2),
        P('직영 매장은 접수만 하고 수리는 본사 수선실로 보낸다. 협력 수선점은 현장에서 직접 수리하며, 일부는 각인 부위 수리까지 한다. '
          '백화점 A/S 데스크는 백화점 영업일에 접수하며 토요일 접수 여부는 데스크마다 다르다.'),
        Paragraph('3. 접수처 목록', _H2),
        P('2026년 10월 기준 A/S 접수처는 다음과 같다.'),
        Spacer(1, 4),
        _rl_table(as_rows()),
        Paragraph('[표 1] A/S 접수처 목록 (2026년 10월)', _CAP),
        P('평균 수리 기간은 접수일부터 수리 완료 안내까지의 영업일 기준 평균이며, 부품 수급에 따라 늘어날 수 있다.'),
        Paragraph('4. 접수처 정보 변경', _H2),
        P('접수처가 문을 닫거나 운영 정보가 바뀌면 고객지원팀이 매월 1일에 이 목록을 갱신한다.'),
    ]
    SimpleDocTemplate(str(p), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm, title='A/S 접수처 안내').build(story)
    return p


# ── harim_16 협력농가 현황 (상한 근처 · 칸 다툼) ────────────────────────────────────

_FARM_AREAS = ['전북 정읍', '전북 익산', '전북 김제', '전남 나주', '충남 논산', '충남 홍성', '경북 상주', '경북 의성', '경기 이천', '강원 횡성',
               '충북 음성', '경남 거창']
_FARM_WELFARE = {9, 36, 58, 90}         # 동물복지 인증 — 묶음마다 흩는다
_FARM_DUCK = {15, 52, 83}               # 오리 공급 농가
_FARM_TOP = 77                          # 연간 공급량 최대(3,480톤) — 하나뿐


def farm_rows() -> list[list[str]]:
    rows = [['농가 코드', '농가명', '지역', '주요 품목', '인증', '연간 공급량']]
    names = ['산들', '햇살', '들녘', '바람', '솔빛', '은하', '푸른', '다솜', '가람', '해오름', '초록', '새벽']
    for i in range(1, 97):
        area = _FARM_AREAS[(i * 5) % len(_FARM_AREAS)]
        item = '오리' if i in _FARM_DUCK else '육계' if i % 4 else '토종닭'
        cert = '동물복지' if i in _FARM_WELFARE else 'HACCP·무항생제' if i % 5 == 0 else 'HACCP'
        amount = '3,480톤' if i == _FARM_TOP else f'{(i * 37) % 1900 + 400:,}톤'          # 그 밖은 400~2,299톤
        rows.append([f'HF-{200 + i}', f'{area.split()[1]} {names[i % 12]}농장 {i:02d}', area, item, cert, amount])
    body = rows[1:]
    amounts = [int(r[5].removesuffix('톤').replace(',', '')) for r in body]
    assert len(body) == 96 and amounts.count(max(amounts)) == 1 and max(amounts) == 3480
    assert sum(r[4] == '동물복지' for r in body) == 4 and sum(r[3] == '오리' for r in body) == 3
    return rows


def build_harim() -> Path:
    p = OUT / 'harim' / 'harim_16_협력농가현황.docx'
    d = DocxDocument()
    d.add_heading('협력농가 현황 (HR-QA-016)', level=1)
    d.add_paragraph('기준일: 2026년 10월 1일 | 주관: 구매품질팀 | 고객센터 상담용')
    d.add_heading('1. 안내 목적', level=2)
    d.add_paragraph('원료육을 공급하는 협력농가의 지역·품목·인증 현황을 정리한 것이다. 고객이 원료 산지나 인증을 문의하면 이 목록으로 '
                    '안내한다. 위생 점검 일정은 협력농가 위생 점검 일정표를, 인증 제도 설명은 품질 인증 안전 관리 기준을 따른다.')
    d.add_heading('2. 협력농가 목록', level=2)
    d.add_paragraph('2026년 10월 기준 협력농가는 다음과 같다. 연간 공급량은 2025년 실적이다.')
    rows = farm_rows()
    t = d.add_table(rows=len(rows), cols=len(rows[0]))
    t.style = 'Table Grid'
    _docx_fill(t, rows)
    cap = d.add_paragraph('[표 1] 협력농가 현황 (2026년 10월)')
    cap.alignment = 1
    d.add_heading('3. 변경 관리', level=2)
    d.add_paragraph('신규 계약·계약 종료·인증 변경이 있으면 구매품질팀이 분기마다 목록을 갱신한다.')
    d.save(p)
    return p


if __name__ == '__main__':
    for build in (build_aromanica, build_homeplus, build_summers, build_harim):
        print(build())
