"""#247 합성 그림 문서 생성 — 실문서 29장에 없던 그림 케이스(차트 6·수식 2·복잡한 흐름도 2·작은 그림 3·영한 화면·표 칸 안 그림)를
**실제 사내 문서 모양**(문서번호·목적·절별 본문·그림 참조 문장·그림 제목 줄·해설·자료 주석)으로 homeplus 문서 3건에 넣는다.
그림과 정답은 eval/doc_fidelity/picture_specs.py.

앞뒤 문맥을 두는 이유(10/18 사용자 지적): 그림만 덩그러니 있으면 그림이 문맥과 같은 청크로 묶이는지, 캡션만 따로 떨어지는지를
볼 수 없다 — 생성은 청크에 함께 실린 문맥에 좌우된다. 그림 아래 '[그림 N] …' 제목 줄은 실제 문서에 흔하고, docling이 그 줄을
어떻게 다루는지(캡션 요소로 붙이는지·본문으로 두는지·버리는지)도 확인 대상이다.

**본문에는 그림 속 숫자·사실을 쓰지 않는다** — 정보가 그림에만 있어야 캡션 오류·누락이 검색·답변에서 그대로 드러난다.
PDF의 작은 그림(area)은 쪽 면적 대비 목표 비율로 크기를 맞춘다 — 면적 하한(settings.vlm_caption_area_threshold 1%, #247 결정 17)
양쪽에서 캡션이 붙는지 본다. 두 상자 사이에는 본문 문단을 둔다 — 붙여 두면 docling이 그림 하나로 합쳤다(10/18 실측).
XLSX는 시트를 짧게 두고 데이터 표는 넣지 않는다 — 엑셀은 보통 데이터 옆에 차트를 두지만 그러면 정답이 표에 드러난다.
시트가 길면 그림 면적 비율이 하한 미만으로 계산돼 캡션이 안 붙는다(#249, 범위 밖).

사용: python sample_docs/corpus_v2/_src/build_picture_docs.py   → sample_docs/corpus_v2/homeplus/ 에 산출
"""
import io
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import openpyxl                                                       # noqa: E402
from docx import Document as DocxDocument                             # noqa: E402
from docx.shared import Mm, Pt                                        # noqa: E402
from openpyxl.drawing.image import Image as XlImage                   # noqa: E402
from reportlab.lib.pagesizes import A4                                # noqa: E402
from reportlab.lib.styles import ParagraphStyle                       # noqa: E402
from reportlab.lib.units import mm                                    # noqa: E402
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402
from reportlab.lib import colors                                      # noqa: E402

from eval.doc_fidelity.picture_specs import DOC_NAMES, PICTURES      # noqa: E402
from eval.table_parsing.build import _BODY                            # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'homeplus'
_H1 = ParagraphStyle('h1', fontName='HYGothic-Medium', fontSize=17, leading=24, spaceAfter=4)
_H2 = ParagraphStyle('h2', fontName='HYGothic-Medium', fontSize=13, leading=18, spaceBefore=12, spaceAfter=4)
_H3 = ParagraphStyle('h3', fontName='HYGothic-Medium', fontSize=11.5, leading=16, spaceBefore=6, spaceAfter=2)
_CAP = ParagraphStyle('cap', fontName='HYGothic-Medium', fontSize=9, leading=12, alignment=1, spaceBefore=2, spaceAfter=8)
_NOTE = ParagraphStyle('note', fontName='HYGothic-Medium', fontSize=8.5, leading=12, textColor=colors.HexColor('#555555'))
PAGE_AREA = A4[0] * A4[1]


def _png(key) -> bytes:
    buf = io.BytesIO()
    PICTURES[key]['draw']().save(buf, format='PNG')
    return buf.getvalue()


def _rl(key, width_mm=165):
    im = PICTURES[key]['draw']()
    ratio = im.height / im.width
    if PICTURES[key].get('area'):                                    # 쪽 면적 대비 목표 비율 → 너비(pt)
        w = math.sqrt(PICTURES[key]['area'] * PAGE_AREA / ratio)
    else:
        w = width_mm * mm
    return RLImage(io.BytesIO(_png(key)), width=w, height=w * ratio)


def _meta_table(rows):
    t = Table(rows, colWidths=[35 * mm, 135 * mm])
    t.setStyle(TableStyle([('FONT', (0, 0), (-1, -1), 'HYGothic-Medium', 9), ('GRID', (0, 0), (-1, -1), 0.4, colors.grey),
                           ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#eef2f7'))]))
    return t


def build_pdf() -> Path:
    p = OUT / DOC_NAMES['pdf']
    P = lambda t: Paragraph(t, _BODY)                                # noqa: E731
    C = lambda t: Paragraph(t, _CAP)                                 # noqa: E731
    N = lambda t: Paragraph(t, _NOTE)                                # noqa: E731
    story = [
        Paragraph('매장 운영 지표 및 근무 기준 안내', _H1),
        _meta_table([['문서번호', 'HP-OPS-015'], ['시행일', '2026년 10월 1일'], ['주관 부서', '영업관리팀 (근무 기준: 인사팀, 장비: 시설관리팀)'],
                     ['적용 대상', '전 직영 매장 점장·부점장·근무 편성 담당자']]),
        Spacer(1, 8),
        Paragraph('1. 목적', _H2),
        P('이 안내서는 매장 운영 담당자가 월간 운영 지표를 해석하고, 근무 수당·장비 고장 처리·근무표 시스템·지급 물품에 관한 사내 기준을 '
          '일관되게 적용하도록 돕기 위해 작성되었다. 지표 수치와 산식은 각 그림에 정리되어 있으며, 본문은 해석 방법과 적용 원칙을 설명한다.'),
        P('이 안내서와 개별 규정이 다를 경우 개별 규정이 우선하며, 해석에 이견이 있으면 주관 부서에 문의한다.'),
        Paragraph('2. 운영 지표', _H2),
        Paragraph('2.1 매장 방문 현황', _H3),
        P('매장 방문객 수는 출입구 계수기 집계를 기준으로 하며, 동일 고객의 재입장도 각각 1회로 센다. 매장별 월 방문 현황은 [그림 1]과 같다.'),
        _rl('P01_bar'), C('[그림 1] 매장별 월 방문객 (2026년 9월)'),
        P('방문객 수가 전월보다 크게 줄어든 매장은 점장이 원인 분석 보고서를 다음 달 5일까지 영업관리팀에 제출한다. '
          '행사·공사 등 일시적 요인은 보고서에 함께 적는다.'),
        N('※ 자료: 영업관리팀 출입 계수 시스템'),
        Paragraph('2.2 매출 현황', _H3),
        P('분기 매출은 재무팀 마감 기준이며, 반품·에누리를 뺀 순매출이다. 전년 동기와의 비교는 [그림 2]를 참고한다.'),
        _rl('P08_grouped'), C('[그림 2] 분기별 매출 비교 (전년 대비)'),
        P('4분기 수치는 연말 결산 후 갱신하며, 갱신 전까지는 3분기까지의 누계로 연간 목표 달성률을 판단한다.'),
        P('온라인 채널 매출은 이커머스팀이 월 단위로 집계한다. 월별 추이와 전월 대비 증감률은 [그림 3]과 같다.'),
        _rl('P09_combo'), C('[그림 3] 월별 온라인 매출 추이'),
        P('증감률이 두 달 연속 감소하면 이커머스팀과 매장이 공동으로 판촉 계획을 다시 세운다.'),
        PageBreak(),
        Paragraph('3. 근무 수당', _H2),
        Paragraph('3.1 야간 근무 수당', _H3),
        P('야간 근무는 사전에 근무표에 편성된 경우에만 수당 지급 대상이 된다. 편성되지 않은 연장·야간 근무는 사후 승인을 받아야 하며, '
          '승인 전에는 수당이 계산되지 않는다. 야간 근무 수당의 계산식은 [산식 1]과 같다.'),
        _rl('P02_formula'), C('[산식 1] 야간수당 계산식'),
        P('통상시급의 산정 기준이 되는 기본급에는 직책수당이 포함되지 않는다.'),
        Paragraph('3.2 주말 근무', _H3),
        P('주말 근무는 점장 승인 후 근무표에 반영한다. 주말 근무에 적용하는 가산 기준은 아래 참고 상자에 정리했다.'),
        _rl('P04_small'),
        P('가산 기준은 공휴일 근무에는 적용하지 않으며, 공휴일 근무는 인사팀의 휴일근무 기준을 따른다.'),
        Paragraph('4. 장비 고장 처리', _H2),
        P('매장 장비(냉장 진열대·포스 단말기·저울 등)가 고장 나면 담당자는 즉시 시설관리팀 헬프데스크에 신고한다. '
          '처리 절차는 [그림 4]와 같으며, 각 단계의 처리 결과는 헬프데스크 시스템에 기록한다.'),
        _rl('P03_flow', 140), C('[그림 4] 매장 장비 고장 처리 절차'),
        P('수리 기간 동안 대체 장비가 필요하면 시설관리팀에 임대를 요청할 수 있다. 임대 비용은 매장 운영비에서 처리한다.'),
        PageBreak(),
        Paragraph('5. 근무표 시스템', _H2),
        P('근무표는 매장 운영 시스템(Store Ops Console)에서 조회하고 제출한다. 근무 편성 담당자는 매주 금요일까지 다음 주 근무표를 '
          '제출해야 하며, 제출 후 변경은 점장 승인이 필요하다. 근무표 화면 구성은 [그림 5]와 같다.'),
        _rl('P05_screen'), C('[그림 5] 근무표 화면'),
        P('시스템 접속 장애 시에는 종이 근무표로 대체하고, 복구 후 24시간 안에 시스템에 다시 입력한다.'),
        Paragraph('6. 지급 물품', _H2),
        P('매장 직원에게는 업무용 물품을 지급한다. 지급 수량과 주기는 아래 기준을 따르며, 분실·훼손 시 재지급은 점장 승인 후 1회에 한한다.'),
        _rl('P06_edge05'),
        P('안전화는 매장 바닥 작업이 있는 직원에게만 지급하며, 지급 대상은 점장이 정한다.'),
        _rl('P07_edge15'),
        P('퇴직 시에는 지급 물품을 반납하지 않아도 된다. 다만 사원증과 출입 카드는 반드시 반납한다.'),
        N('※ 문의: 인사팀(근무·수당), 시설관리팀(장비), 영업관리팀(지표)'),
    ]
    SimpleDocTemplate(str(p), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, title='매장 운영 지표 및 근무 기준 안내').build(story)
    return p


def build_docx() -> Path:
    p = OUT / DOC_NAMES['docx']
    d = DocxDocument()
    d.styles['Normal'].font.size = Pt(10)

    def cap(text):
        para = d.add_paragraph(text)
        para.alignment = 1
        para.runs[0].font.size = Pt(9)

    d.add_heading('매장 시스템 사용 및 업무 절차 안내', level=1)
    d.add_paragraph('문서번호: HP-SYS-016   |   시행일: 2026년 10월 1일   |   주관 부서: 고객지원팀·물류운영팀·인사팀')
    d.add_heading('1. 목적과 적용 범위', level=2)
    d.add_paragraph('이 안내서는 매장 운영 시스템을 쓰는 직원이 고객 문의 현황, 매장 인력 구성, 연장 근무 수당, 재고 이동 절차를 '
                    '같은 기준으로 이해하도록 돕는다. 수치는 각 그림에 정리되어 있으며, 본문은 해석과 적용 원칙을 설명한다.')
    d.add_heading('2. 고객 문의 현황', level=2)
    d.add_paragraph('고객 문의는 매장 방문 문의와 콜센터 이관 문의를 합산하며, 같은 고객이 같은 날 여러 번 문의해도 1건으로 센다. '
                    '분기별 문의 건수 추이는 [그림 1]과 같다.')
    d.add_picture(io.BytesIO(_png('D01_bar')), width=Mm(150))
    cap('[그림 1] 분기별 고객 문의 건수')
    d.add_paragraph('문의가 많았던 분기에는 원인을 유형별(배송·교환·결제)로 분석해 다음 분기 응대 교육에 반영한다.')
    d.add_heading('3. 매장 인력 현황', level=2)
    d.add_paragraph('매장 인력은 정규직과 계약직으로 나뉘며, 단시간 근로자는 이 통계에 넣지 않는다. 매장별 고용 형태 구성은 [그림 2]와 같다.')
    d.add_picture(io.BytesIO(_png('D05_stacked')), width=Mm(150))
    cap('[그림 2] 매장별 직원 구성 (2026년 9월 말 기준)')
    d.add_paragraph('계약직 비중이 높은 매장은 정규직 전환 심사 대상자를 분기마다 인사팀에 추천할 수 있다.')
    d.add_heading('4. 연장 근무 수당', level=2)
    d.add_paragraph('연장 근무는 점장의 사전 승인을 받은 경우에만 인정한다. 사전 승인 없이 근무한 경우 다음 날까지 사유서를 제출해야 하며, '
                    '사유서가 승인되면 수당을 지급한다. 계산식은 [산식 2]와 같다.')
    d.add_picture(io.BytesIO(_png('D02_formula')), width=Mm(160))
    cap('[산식 2] 연장수당 계산식')
    d.add_paragraph('연장 근무와 야간 근무가 겹치면 두 수당을 모두 지급한다.')
    d.add_heading('5. 매장 간 재고 이동', level=2)
    d.add_paragraph('재고가 부족한 상품은 본사 발주 전에 인근 매장 재고를 먼저 확인한다. 이동 절차는 [그림 3]과 같으며, '
                    '이동 내역은 물류 시스템에 당일 등록한다.')
    d.add_picture(io.BytesIO(_png('D03_flow')), width=Mm(150))
    cap('[그림 3] 매장 간 재고 이동 절차')
    d.add_paragraph('냉장·냉동 상품은 매장 간 이동을 하지 않고 본사 발주로만 처리한다.')
    d.add_heading('6. 화면 안내', level=2)
    d.add_paragraph('자주 쓰는 메뉴의 위치는 아래 표와 같다.')
    t = d.add_table(rows=3, cols=2)
    t.style = 'Table Grid'
    t.cell(0, 0).text, t.cell(0, 1).text = '화면', '설명'
    t.cell(1, 0).paragraphs[0].add_run().add_picture(io.BytesIO(_png('D04_cell')), width=Mm(55))
    t.cell(1, 1).text = '근무표 메뉴 — 이번 주 근무 일정을 조회한다'
    t.cell(2, 0).text, t.cell(2, 1).text = '급여 명세', '상단 메뉴 [급여] > [명세서]에서 월별 명세서를 내려받는다'
    d.add_paragraph('메뉴 위치는 시스템 개편 시 바뀔 수 있으며, 변경 사항은 사내 공지로 안내한다.')
    d.save(p)
    return p


def build_xlsx() -> Path:
    p = OUT / DOC_NAMES['xlsx']
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '요약'
    for row in [['카테고리 매출 비중·교육 이수 현황 (분기 보고)'], [],
                ['구분', '담당 부서', '갱신 주기', '비고'],
                ['카테고리 매출 비중', '상품기획팀', '분기', '순매출 기준, 차트 참고'],
                ['법정교육 이수율', '인재개발팀', '월', '미이수자는 팀장에게 개별 통보']]:
        ws.append(row)
    for key, anchor in (('X01_pie', 'F2'), ('X02_hbar', 'F26')):
        img = XlImage(io.BytesIO(_png(key)))
        img.width, img.height = img.width * 0.55, img.height * 0.55
        ws.add_image(img, anchor)
    wb.save(p)
    return p


if __name__ == '__main__':
    for build in (build_pdf, build_docx, build_xlsx):
        print(build())
