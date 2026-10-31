"""합성 그림 정의 — **정답(그림에 넣은 글자·숫자·짝·순서)이 이 정의 자체다**(#247 결정 9, 실문서 29장에 없던 케이스 보강).

실문서 그림은 화면 캡처 25·단순 흐름도 3·영수증 1뿐이라(10/18 전수 추출) 차트·수식·복잡한 흐름도·작은데 중요한 그림·표 칸 안
그림·영한 혼합·면적 하한(1%, #247 결정 17) 경계를 여기서 만든다. 문서 생성은 sample_docs/corpus_v2/_src/build_picture_docs.py, 채점은 synthetic.py.
소재는 homeplus 기존 문서(반품·배송·멤버십 포인트·PB·픽업)와 겹치지 않는다(#253 교훈 "한 사실은 한 문서에만").

그림은 도형(축·막대·박스·화살표)을 넣어 '그림'으로 보이게 그린다 — 글자만 있는 이미지는 PDF에서 docling이 글자로 인식해
VLM 경로를 타지 않는다(10/18 프로브). 분류가 어떻게 되는지도 결과의 일부로 남긴다.

정답 항목:
  texts — 캡션에 있어야 할 글자(공백 무시) · numbers — 그중 숫자·금액·비율 · pairs — 같은 줄에 있어야 할 (항목, 값) 짝
  sequence — 캡션에 이 순서로 나와야 할 단계(흐름도) · branches — (조건, 예 →, 아니오 →): 자동 채점 밖, 판정자·사람이 본다
"""

from PIL import Image, ImageDraw

from eval.table_parsing.render_image import font

W, H = 1200, 720
NAVY, GRAY, LINE_C = '#1f3b63', '#5b6573', '#c9ced6'
PALETTE = ['#1f4e79', '#2e75b6', '#9dc3e6', '#bdd7ee']  # 사내 보고서 톤(남색 계열)


def _mpl():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    from eval.table_parsing.render_image import FONT_PATH

    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def _fig_to_image(fig):
    import io

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return Image.open(buf).convert('RGB')


def _canvas(w=W, h=H):
    im = Image.new('RGB', (w, h), 'white')
    return im, ImageDraw.Draw(im)


def _center_text(d, xy, text, size, fill='black'):
    f = font(size)
    lines = text.split('\n')
    lh = size + 6
    y = (xy[1] + xy[3]) / 2 - lh * len(lines) / 2
    for line in lines:
        tw = d.textlength(line, font=f)
        d.text(((xy[0] + xy[2]) / 2 - tw / 2, y), line, fill=fill, font=f)
        y += lh


def bar_chart(title, items, unit, ylabel, source):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels, values = zip(*items)
    bars = ax.bar(labels, values, color=PALETTE[1], width=0.55, zorder=3)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02, f'{v:,}', ha='center', va='bottom', fontsize=11)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    ax.set_ylabel(f'{ylabel}({unit})', fontsize=10, color=GRAY)
    ax.grid(axis='y', color='#e3e6ea', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(values) * 1.18)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def grouped_bar(title, cats, series, unit, ylabel, source):
    """묶은 막대 — series: [(이름, [값…])]."""
    plt = _mpl()
    import numpy as np

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(cats))
    w = 0.36
    for i, (name, vals) in enumerate(series):
        shown = [(xi, v) for xi, v in zip(x, vals) if v is not None]  # None = 집계 전 — 막대·숫자를 그리지 않는다
        bars = ax.bar(
            [xi + (i - 0.5) * w for xi, _ in shown],
            [v for _, v in shown],
            w,
            label=name,
            color=[PALETTE[2], PALETTE[0]][i],
            zorder=3,
        )
        for b, (_, v) in zip(bars, shown):
            ax.text(b.get_x() + b.get_width() / 2, v + 4, f'{v:,}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x, cats)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    ax.set_ylabel(f'{ylabel}({unit})', fontsize=10, color=GRAY)
    ax.grid(axis='y', color='#e3e6ea', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left', ncol=2, fontsize=10)
    ax.set_ylim(0, max(max(x for x in v if x is not None) for _, v in series) * 1.22)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def combo_bar_line(title, cats, bars_v, line_v, bar_name, line_name, source):
    """막대(왼쪽 축) + 꺾은선(오른쪽 보조축)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    b = ax.bar(cats, bars_v, color=PALETTE[1], width=0.5, zorder=3, label=bar_name)
    for r, v in zip(b, bars_v):
        ax.text(
            r.get_x() + r.get_width() / 2, v * 0.5, f'{v}', ha='center', color='white', fontsize=10, fontweight='bold'
        )
    ax.set_ylabel(bar_name, fontsize=10, color=GRAY)
    ax.set_ylim(0, max(bars_v) * 1.3)
    ax.grid(axis='y', color='#e3e6ea', zorder=0)
    ax.spines[['top']].set_visible(False)
    ax2 = ax.twinx()
    ax2.plot(cats, line_v, color='#c55a11', marker='o', linewidth=2, label=line_name)
    for xi, v in zip(cats, line_v):
        ax2.annotate(
            f'{v:+.1f}%', (xi, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=9, color='#c55a11'
        )
    ax2.set_ylabel(line_name, fontsize=10, color='#c55a11')
    ax2.set_ylim(min(line_v) - 6, max(line_v) + 6)
    ax2.spines[['top']].set_visible(False)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc='upper left', fontsize=10)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def stacked_bar(title, cats, series, unit, source):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bottom = [0] * len(cats)
    for i, (name, vals) in enumerate(series):
        bars = ax.bar(cats, vals, 0.55, bottom=bottom, label=name, color=[PALETTE[0], PALETTE[2]][i], zorder=3)
        for r, v, b0 in zip(bars, vals, bottom):
            ax.text(
                r.get_x() + r.get_width() / 2,
                b0 + v / 2,
                f'{v}',
                ha='center',
                va='center',
                fontsize=10,
                color='white' if i == 0 else 'black',
            )
        bottom = [b0 + v for b0, v in zip(bottom, vals)]
    for xi, t in zip(cats, bottom):
        ax.text(xi, t + 0.6, f'계 {t}', ha='center', fontsize=9, color=GRAY)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    ax.set_ylabel(f'인원({unit})', fontsize=10, color=GRAY)
    ax.grid(axis='y', color='#e3e6ea', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper right', fontsize=10)
    ax.set_ylim(0, max(bottom) * 1.2)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def hbar_chart(title, items, unit, source):
    """가로 막대(순위형) — 큰 값이 위."""
    plt = _mpl()
    items = sorted(items, key=lambda x: x[1])
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    labels, vals = zip(*items)
    bars = ax.barh(labels, vals, color=PALETTE[1], height=0.55, zorder=3)
    for r, v in zip(bars, vals):
        ax.text(v + 1, r.get_y() + r.get_height() / 2, f'{v}{unit}', va='center', fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    ax.grid(axis='x', color='#e3e6ea', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def pie_chart(title, items, source):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels, values = zip(*items)
    wedges, _ = ax.pie(
        values, colors=PALETTE, startangle=90, counterclock=False, wedgeprops=dict(width=0.42, edgecolor='white')
    )
    ax.legend(
        wedges,
        [f'{l}  {v}%' for l, v in items],
        loc='center left',
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
        fontsize=11,
    )
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def formula(title, expr, notes):
    """사내 규정집의 '산식' 안내 상자 — 제목 띠 + 산식 + 비고."""
    h = 230 + 42 * len(notes)
    im, d = _canvas(1200, h)
    d.rectangle([20, 20, 1180, h - 20], outline=LINE_C, width=2)
    d.rectangle([20, 20, 1180, 72], fill=NAVY)
    d.text((44, 32), title, fill='white', font=font(26))
    d.rectangle([60, 100, 1140, 190], fill='#f4f7fb', outline='#9fb3cc', width=2)
    _center_text(d, (60, 100, 1140, 190), expr, 32, fill=NAVY)
    for i, n in enumerate(notes):
        d.text((70, 215 + i * 42), f'※ {n}', fill='#333', font=font(22))
    return im


def flow(title, nodes, edges, size=(1200, 860), owner=None):
    """표준 기호 흐름도 — 시작·종료(둥근 사각형), 처리(사각형), 판단(마름모, 텍스트 끝 '?')."""
    im, d = _canvas(*size)
    d.text((30, 18), title, fill=NAVY, font=font(28))
    d.line([(30, 62), (size[0] - 30, 62)], fill=NAVY, width=2)
    if owner:
        d.text((size[0] - 300, 24), owner, fill=GRAY, font=font(18))
    for k, (x0, y0, x1, y1, text) in nodes.items():
        if text.endswith('?'):
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            d.polygon([(cx, y0 - 14), (x1 + 20, cy), (cx, y1 + 14), (x0 - 20, cy)], fill='#fff4e5', outline='#c07a1c')
            d.line([(cx, y0 - 14), (x1 + 20, cy), (cx, y1 + 14), (x0 - 20, cy), (cx, y0 - 14)], fill='#c07a1c', width=2)
            _center_text(d, (x0, y0, x1, y1), text, 20)
        elif k in ('start', 'end') or text in ('처리 종료', '본사 발주', '매장 간 이동'):
            d.rounded_rectangle((x0, y0, x1, y1), radius=28, fill='#e7eef7', outline=NAVY, width=2)
            _center_text(d, (x0, y0, x1, y1), text, 20)
        else:
            d.rectangle((x0, y0, x1, y1), fill='white', outline=NAVY, width=2)
            _center_text(d, (x0, y0, x1, y1), text, 20)

    def mid(k, side):
        x0, y0, x1, y1, t = nodes[k]
        pad = 20 if t.endswith('?') else 0
        padv = 14 if t.endswith('?') else 0
        return {
            'b': ((x0 + x1) / 2, y1 + padv),
            't': ((x0 + x1) / 2, y0 - padv),
            'r': (x1 + pad, (y0 + y1) / 2),
            'l': (x0 - pad, (y0 + y1) / 2),
        }[side]

    for a, sa, b, sb, label in edges:
        _arrow(d, mid(a, sa), mid(b, sb), label)
    return im


def _arrow(d, a, b, label=None):
    import math

    d.line([a, b], fill='#44546a', width=2)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    d.polygon(
        [
            b,
            (b[0] - 13 * math.cos(ang - 0.4), b[1] - 13 * math.sin(ang - 0.4)),
            (b[0] - 13 * math.cos(ang + 0.4), b[1] - 13 * math.sin(ang + 0.4)),
        ],
        fill='#44546a',
    )
    if label:
        d.text(((a[0] + b[0]) / 2 + 8, (a[1] + b[1]) / 2 - 24), label, fill='#a14a00', font=font(18))


def small_card(title, body, w, h):
    """본문 옆 작은 안내 박스 — 규정 문서의 '참고' 상자."""
    im, d = _canvas(w, h)
    d.rectangle([2, 2, w - 3, h - 3], outline='#7f9cc0', width=2, fill='#f5f8fc')
    d.rectangle([2, 2, w - 3, 2 + h // 3], fill='#dfe8f3')
    d.text((14, 6), title, fill=NAVY, font=font(max(14, h // 6)))
    _center_text(d, (8, 2 + h // 3, w - 8, h - 6), body, max(14, h // 6))
    return im


def screen(title, menu, crumbs, fields, buttons, rows=None):
    """사내 웹 시스템 화면 — 상단 바·좌측 메뉴·경로·입력 폼·버튼."""
    im, d = _canvas(1200, 760)
    d.rectangle([0, 0, 1200, 56], fill=NAVY)
    d.text((24, 14), title, fill='white', font=font(24))
    d.text((1000, 18), '김도윤 님 | 로그아웃', fill='#c9d6ea', font=font(16))
    d.rectangle([0, 56, 230, 760], fill='#f1f3f6')
    for i, m in enumerate(menu):
        sel = i == 1
        if sel:
            d.rectangle([0, 80 + i * 50, 230, 122 + i * 50], fill='#dce6f2')
        d.text((28, 88 + i * 50), m, fill=NAVY if sel else '#333', font=font(20))
    d.text((260, 74), crumbs, fill=GRAY, font=font(16))
    d.text((260, 104), menu[1], fill='black', font=font(26))
    d.line([(260, 146), (1170, 146)], fill=LINE_C, width=2)
    for i, (k, v) in enumerate(fields):
        y = 172 + i * 64
        d.rectangle([260, y, 470, y + 46], fill='#f6f7f9', outline=LINE_C)
        d.text((276, y + 11), k, fill='#333', font=font(19))
        d.rectangle([470, y, 1170, y + 46], fill='white', outline=LINE_C)
        d.text((488, y + 11), v, fill='black', font=font(19))
    if rows:
        y0 = 172 + len(fields) * 64 + 24
        d.text((260, y0), '이번 주 근무표', fill='black', font=font(20))
        cols = [260, 400, 560, 760, 960, 1170]
        y = y0 + 34
        for ri, row in enumerate(rows):
            d.rectangle([260, y, 1170, y + 38], fill='#e9eef5' if ri == 0 else 'white', outline=LINE_C)
            for ci, cell in enumerate(row):
                d.text((cols[ci] + 12, y + 8), cell, fill='black', font=font(17))
            for cx in cols[1:-1]:
                d.line([(cx, y), (cx, y + 38)], fill=LINE_C)
            y += 38
    x = 1170
    for label, primary in reversed(buttons):
        tw = d.textlength(label, font=font(18)) + 40
        d.rounded_rectangle([x - tw, 700, x, 744], radius=6, fill='#2e75b6' if primary else 'white', outline='#2e75b6')
        d.text((x - tw + 20, 710), label, fill='white' if primary else '#2e75b6', font=font(18))
        x -= tw + 12
    return im


def button_shot(label):
    im, d = _canvas(420, 150)
    d.rectangle([0, 0, 419, 149], fill='#f1f3f6', outline=LINE_C, width=2)
    d.rectangle([0, 0, 419, 30], fill=NAVY)
    d.text((10, 5), '근무 관리', fill='white', font=font(16))
    d.rounded_rectangle([90, 60, 330, 112], radius=6, fill='#2e75b6')
    tw = d.textlength(label, font=font(20))
    d.text((210 - tw / 2, 74), label, fill='white', font=font(20))
    return im


FLOW_REPAIR = dict(
    title='매장 장비 고장 처리 절차',
    owner='담당: 시설관리팀',
    nodes={
        'start': (460, 90, 740, 150, '고장 신고'),
        'b': (460, 190, 740, 250, '원격 점검'),
        'c': (480, 300, 720, 360, '원격 해결?'),
        'end': (860, 300, 1120, 360, '처리 종료'),
        'e': (480, 430, 720, 490, '보증 기간 내?'),
        'f': (120, 560, 400, 620, '무상 수리'),
        'g': (800, 560, 1080, 620, '유상 견적'),
        'h': (120, 700, 400, 760, '수리 검수'),
    },
    edges=[
        ('start', 'b', 'b', 't', None),
        ('b', 'b', 'c', 't', None),
        ('c', 'r', 'end', 'l', '예'),
        ('c', 'b', 'e', 't', '아니오'),
        ('e', 'l', 'f', 't', '예'),
        ('e', 'r', 'g', 't', '아니오'),
        ('f', 'b', 'h', 't', None),
        ('h', 'r', 'f', 'r', '불합격 시 재수리'),
    ],
)
FLOW_STOCK = dict(
    title='매장 간 재고 이동 절차',
    owner='담당: 물류운영팀',
    nodes={
        'start': (460, 90, 740, 150, '재고 이동 요청'),
        'b': (460, 190, 740, 250, '점장 확인'),
        'c': (480, 300, 720, 360, '재고 충분?'),
        'd': (860, 300, 1120, 360, '출고'),
        'e': (480, 450, 720, 510, '인근 매장 재고?'),
        'f': (120, 600, 400, 660, '매장 간 이동'),
        'g': (800, 600, 1080, 660, '본사 발주'),
        'h': (860, 190, 1120, 250, '입고 확인'),
    },
    edges=[
        ('start', 'b', 'b', 't', None),
        ('b', 'b', 'c', 't', None),
        ('c', 'r', 'd', 'l', '예'),
        ('c', 'b', 'e', 't', '아니오'),
        ('e', 'l', 'f', 't', '예'),
        ('e', 'r', 'g', 't', '아니오'),
        ('d', 't', 'h', 'b', None),
        ('h', 'l', 'b', 'r', '수량 불일치'),
    ],
)

# 그림 정의 — id: (문서 키, 그리는 함수, 정답). area는 PDF에 넣을 때 쪽 면적 대비 목표 비율(면적 하한 경계 시험용 —
# P06 0.5%는 하한 1% 아래라 캡션이 없어야, P07 1.5%는 위라 붙어야 맞다. 결정 17 전에는 4%·6%로 5% 경계를 쟀다).
PICTURES = {
    'P01_bar': dict(
        doc='pdf',
        draw=lambda: bar_chart(
            '매장별 월 방문객',
            [('상암', 42), ('일산', 37), ('수원', 55), ('부천', 29)],
            '천 명',
            '방문객',
            '출처: 영업관리팀, 2026.09',
        ),
        texts=['매장별 월 방문객', '상암', '일산', '수원', '부천'],
        numbers=['42', '37', '55', '29'],
        pairs=[('상암', '42'), ('일산', '37'), ('수원', '55'), ('부천', '29')],
    ),
    'P02_formula': dict(
        doc='pdf',
        draw=lambda: formula(
            '[산식 1] 야간수당',
            '야간수당 = 통상시급 × 0.5 × 야간근로시간',
            ['야간근로: 22:00 ~ 익일 06:00 사이의 근로', '통상시급은 기본급 ÷ 209시간으로 산정'],
        ),
        texts=['야간수당', '통상시급', '야간근로시간', '22:00', '06:00', '209시간'],
        numbers=['0.5', '209'],
        sequence=['야간수당', '통상시급', '0.5', '야간근로시간'],
    ),
    'P03_flow': dict(
        doc='pdf',
        draw=lambda: flow(**FLOW_REPAIR),
        texts=[
            '고장 신고',
            '원격 점검',
            '원격 해결',
            '처리 종료',
            '보증 기간 내',
            '무상 수리',
            '유상 견적',
            '수리 검수',
            '재수리',
        ],
        numbers=[],
        sequence=['고장 신고', '원격 점검', '원격 해결'],
        branches=[
            ('원격 해결?', '처리 종료', '보증 기간 내?'),
            ('보증 기간 내?', '무상 수리', '유상 견적'),
            ('수리 검수(불합격)', '무상 수리', None),
        ],
    ),
    'P04_small': dict(
        doc='pdf',
        area=0.03,
        draw=lambda: small_card('참고', '주말 근무 가산율 1.5배\n(토·일 08:00~22:00)', 420, 150),
        texts=['주말 근무 가산율', '토·일', '08:00', '22:00'],
        numbers=['1.5'],
    ),
    'P05_screen': dict(
        doc='pdf',
        draw=lambda: screen(
            'Store Ops Console',
            ['Dashboard', '근무표', 'Payroll', '설정'],
            'Home > 근무 관리 > 근무표',
            [('Shift Code', 'N-2 (야간 2조)'), ('Break Time', '60분'), ('Manager', '김도윤')],
            [('취소', False), ('Submit 제출', True)],
            [
                ['일자', '요일', 'Shift', '근무 시간', '비고'],
                ['10/20', 'MON', 'D-1', '09:00~18:00', ''],
                ['10/21', 'TUE', 'N-2', '22:00~07:00', '야간'],
                ['10/22', 'WED', 'OFF', '-', '휴무'],
            ],
        ),
        texts=[
            'Store Ops Console',
            'Dashboard',
            '근무표',
            'Payroll',
            'Shift Code',
            'N-2',
            '야간 2조',
            'Break Time',
            '60분',
            '김도윤',
            'Submit',
            '이번 주 근무표',
            '10/21',
            '22:00~07:00',
        ],
        numbers=['60'],
        pairs=[('Shift Code', 'N-2'), ('Break Time', '60분'), ('10/21', 'N-2')],
    ),
    'P06_edge05': dict(
        doc='pdf',
        area=0.005,
        draw=lambda: small_card('지급 기준', '유니폼 지급: 연 2벌', 420, 150),
        texts=['유니폼 지급', '연 2벌'],
        numbers=['2'],
    ),
    'P07_edge15': dict(
        doc='pdf',
        area=0.015,
        draw=lambda: small_card('지급 기준', '안전화 지급: 2년마다 1켤레', 420, 150),
        texts=['안전화 지급', '2년마다 1켤레'],
        numbers=['2', '1'],
    ),
    'P08_grouped': dict(
        doc='pdf',
        draw=lambda: grouped_bar(
            '분기별 매출 비교(전년 대비)',
            ['1분기', '2분기', '3분기', '4분기'],
            [('2025년', [310, 288, 335, 362]), ('2026년', [342, 301, 369, None])],
            '억 원',
            '매출',
            '출처: 재무팀, 2026.10 (4분기는 집계 전)',
        ),
        texts=['분기별 매출 비교', '2025년', '2026년', '1분기', '2분기', '3분기', '4분기'],
        numbers=['310', '288', '335', '362', '342', '301', '369'],
        pairs=[
            ('1분기', '310'),
            ('1분기', '342'),
            ('2분기', '288'),
            ('2분기', '301'),
            ('3분기', '335'),
            ('3분기', '369'),
        ],
    ),
    'P09_combo': dict(
        doc='pdf',
        draw=lambda: combo_bar_line(
            '월별 온라인 매출 추이',
            ['5월', '6월', '7월', '8월', '9월'],
            [48, 52, 61, 57, 66],
            [2.1, 8.3, 17.3, -6.6, 15.8],
            '매출(억 원)',
            '전월 대비 증감률(%)',
            '출처: 이커머스팀, 2026.10',
        ),
        texts=['월별 온라인 매출 추이', '5월', '6월', '7월', '8월', '9월'],
        numbers=['48', '52', '61', '57', '66', '8.3', '17.3', '-6.6', '15.8'],
        pairs=[('7월', '61'), ('8월', '57'), ('8월', '-6.6'), ('9월', '66')],
    ),
    'D01_bar': dict(
        doc='docx',
        draw=lambda: bar_chart(
            '분기별 고객 문의 건수',
            [('1분기', 120), ('2분기', 95), ('3분기', 143), ('4분기', 110)],
            '건',
            '문의 건수',
            '출처: 고객지원팀, 2026.09',
        ),
        texts=['분기별 고객 문의 건수', '1분기', '2분기', '3분기', '4분기'],
        numbers=['120', '95', '143', '110'],
        pairs=[('1분기', '120'), ('2분기', '95'), ('3분기', '143'), ('4분기', '110')],
    ),
    'D02_formula': dict(
        doc='docx',
        draw=lambda: formula(
            '[산식 2] 연장수당',
            '연장수당 = 통상시급 × 1.5 × 연장근로시간',
            ['연장근로: 1일 8시간 초과분', '휴일 연장은 별도 기준(인사팀 문의)'],
        ),
        texts=['연장수당', '통상시급', '연장근로시간', '8시간 초과'],
        numbers=['1.5'],
        sequence=['연장수당', '통상시급', '1.5', '연장근로시간'],
    ),
    'D03_flow': dict(
        doc='docx',
        draw=lambda: flow(**FLOW_STOCK),
        texts=[
            '재고 이동 요청',
            '점장 확인',
            '재고 충분',
            '출고',
            '인근 매장 재고',
            '매장 간 이동',
            '본사 발주',
            '입고 확인',
            '수량 불일치',
        ],
        numbers=[],
        sequence=['재고 이동 요청', '점장 확인', '재고 충분'],
        branches=[
            ('재고 충분?', '출고', '인근 매장 재고?'),
            ('인근 매장 재고?', '매장 간 이동', '본사 발주'),
            ('입고 확인(수량 불일치)', '점장 확인', None),
        ],
    ),
    'D05_stacked': dict(
        doc='docx',
        draw=lambda: stacked_bar(
            '매장별 직원 구성',
            ['상암', '일산', '수원', '부천'],
            [('정규직', [24, 19, 31, 15]), ('계약직', [11, 8, 13, 9])],
            '명',
            '출처: 인사팀, 2026.09 기준',
        ),
        texts=['매장별 직원 구성', '정규직', '계약직', '상암', '일산', '수원', '부천'],
        numbers=['24', '19', '31', '15', '11', '8', '13', '9', '35', '27', '44'],
        pairs=[('상암', '24'), ('상암', '11'), ('수원', '31'), ('수원', '13')],
    ),
    'D04_cell': dict(
        doc='docx',
        in_table=True,
        draw=lambda: button_shot('근무표 조회'),
        texts=['근무표 조회'],
        numbers=[],
        row_label='근무표 메뉴',
    ),
    'X01_pie': dict(
        doc='xlsx',
        draw=lambda: pie_chart(
            '카테고리별 매출 비중',
            [('신선', 38), ('가공', 27), ('생활', 21), ('기타', 14)],
            '출처: 상품기획팀, 2026년 3분기',
        ),
        texts=['카테고리별 매출 비중', '신선', '가공', '생활', '기타'],
        numbers=['38', '27', '21', '14'],
        pairs=[('신선', '38'), ('가공', '27'), ('생활', '21'), ('기타', '14')],
    ),
    'X02_hbar': dict(
        doc='xlsx',
        draw=lambda: hbar_chart(
            '부서별 법정교육 이수율',
            [('영업', 92), ('물류', 85), ('고객지원', 78), ('시설', 71)],
            '%',
            '출처: 인재개발팀, 2026.09',
        ),
        texts=['부서별 법정교육 이수율', '영업', '물류', '고객지원', '시설'],
        numbers=['92', '85', '78', '71'],
        pairs=[('영업', '92'), ('물류', '85'), ('고객지원', '78'), ('시설', '71')],
    ),
}
DOC_NAMES = {
    'pdf': 'homeplus_15_매장운영지표안내.pdf',
    'docx': 'homeplus_16_매장시스템사용안내.docx',
    'xlsx': 'homeplus_17_카테고리매출비중표.xlsx',
}


# ── 홀드아웃(#247 결정 15 과적합 점검, 10/18) — 프롬프트 변형을 고를 때 **보지 않은** 그림. 튜닝에 쓰지 말 것 ─────────────
def line_chart(title, items, unit, source):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels, vals = zip(*items)
    ax.plot(labels, vals, marker='o', color=PALETTE[0], linewidth=2)
    for x, v in zip(labels, vals):
        ax.annotate(f'{v}{unit}', (x, v), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=10)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    ax.grid(axis='y', color='#e3e6ea')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, max(vals) * 1.35)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


def signed_hbar(title, items, unit, source):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    labels, vals = zip(*items)
    ax.barh(labels, vals, color=[PALETTE[1] if v >= 0 else '#c55a11' for v in vals], height=0.5, zorder=3)
    for i, v in enumerate(vals):
        ax.text(
            v + (0.6 if v >= 0 else -0.6),
            i,
            f'{v:+d}{unit}',
            va='center',
            ha='left' if v >= 0 else 'right',
            fontsize=10,
        )
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlim(min(vals) - 6, max(vals) + 6)
    ax.set_title(title, fontsize=14, fontweight='bold', loc='left', color=NAVY, pad=12)
    ax.grid(axis='x', color='#e3e6ea', zorder=0)
    ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.99, 0.01, source, ha='right', fontsize=8, color=GRAY)
    return _fig_to_image(fig)


ORG = dict(
    title='조직도',
    owner='2026년 10월 기준',
    nodes={
        'a': (480, 90, 720, 150, '대표이사'),
        'b': (100, 240, 340, 300, '경영지원본부'),
        'c': (480, 240, 720, 300, '영업본부'),
        'd': (860, 240, 1100, 300, '기술본부'),
        'e': (330, 400, 570, 460, '국내영업팀'),
        'f': (630, 400, 870, 460, '해외영업팀'),
    },
    edges=[
        ('a', 'b', 'b', 't', None),
        ('a', 'b', 'c', 't', None),
        ('a', 'b', 'd', 't', None),
        ('c', 'b', 'e', 't', None),
        ('c', 'b', 'f', 't', None),
    ],
    size=(1200, 520),
)
FLOW_LEAVE = dict(
    title='휴가 신청 처리 절차',
    owner='담당: 인사팀',
    nodes={
        'start': (460, 90, 740, 150, '휴가 신청'),
        'b': (480, 210, 720, 270, '팀장 승인?'),
        'c': (860, 210, 1120, 270, '반려 통보'),
        'd': (460, 340, 740, 400, '인사팀 등록'),
        'end': (460, 460, 740, 520, '휴가 확정'),
    },
    edges=[
        ('start', 'b', 'b', 't', None),
        ('b', 'r', 'c', 'l', '아니오'),
        ('b', 'b', 'd', 't', '예'),
        ('d', 'b', 'end', 't', None),
    ],
    size=(1200, 580),
)

HOLDOUT = {
    'H01_line': dict(
        draw=lambda: line_chart(
            '월별 이직률',
            [('1월', 2.1), ('2월', 1.8), ('3월', 2.4), ('4월', 3.0), ('5월', 2.7), ('6월', 2.2)],
            '%',
            '출처: 인사팀, 2026.07',
        ),
        texts=['월별 이직률', '1월', '2월', '3월', '4월', '5월', '6월'],
        numbers=['2.1', '1.8', '2.4', '3.0', '2.7', '2.2'],
        pairs=[('1월', '2.1'), ('3월', '2.4'), ('4월', '3.0'), ('6월', '2.2')],
    ),
    'H02_signed': dict(
        draw=lambda: signed_hbar(
            '지역별 매출 증감(전년 대비)',
            [('서울', 12), ('부산', -4), ('대구', 3), ('광주', -7)],
            '%',
            '출처: 영업기획팀, 2026.09',
        ),
        texts=['지역별 매출 증감', '서울', '부산', '대구', '광주'],
        numbers=['12', '-4', '3', '-7'],
        pairs=[('서울', '12'), ('부산', '-4'), ('광주', '-7')],
    ),
    'H03_org': dict(
        draw=lambda: flow(**ORG),
        texts=['대표이사', '경영지원본부', '영업본부', '기술본부', '국내영업팀', '해외영업팀'],
        numbers=[],
        edges=[
            ('대표이사', '경영지원본부'),
            ('대표이사', '영업본부'),
            ('대표이사', '기술본부'),
            ('영업본부', '국내영업팀'),
            ('영업본부', '해외영업팀'),
        ],
    ),
    'H04_leave': dict(
        draw=lambda: flow(**FLOW_LEAVE),
        texts=['휴가 신청', '팀장 승인', '반려 통보', '인사팀 등록', '휴가 확정'],
        numbers=[],
        edges=[
            ('휴가 신청', '팀장 승인'),
            ('팀장 승인', '반려 통보'),
            ('팀장 승인', '인사팀 등록'),
            ('인사팀 등록', '휴가 확정'),
        ],
    ),
    'H05_pie': dict(
        draw=lambda: pie_chart(
            '비용 구성(2026년 상반기)',
            [('인건비', 52), ('임차료', 18), ('마케팅', 17), ('기타', 13)],
            '출처: 재무팀, 2026.07',
        ),
        texts=['비용 구성', '인건비', '임차료', '마케팅', '기타'],
        numbers=['52', '18', '17', '13'],
        pairs=[('인건비', '52'), ('임차료', '18'), ('마케팅', '17'), ('기타', '13')],
    ),
}


# ── 홀드아웃 2(10/18) — V13(계층형 지시) 검증용. HOLDOUT의 조직도를 보고 문구를 고쳤으므로 그와 다른 새 그림으로 잰다 ─────
ORG2 = dict(
    title='고객지원본부 조직',
    owner='2026년 9월 개편',
    nodes={
        'a': (450, 90, 750, 150, '고객지원본부장'),
        'b': (60, 240, 330, 300, '콜센터운영실'),
        'c': (465, 240, 735, 300, '품질관리실'),
        'd': (870, 240, 1140, 300, '고객분석실'),
        'e': (20, 400, 230, 460, '인바운드팀'),
        'f': (250, 400, 460, 460, '아웃바운드팀'),
        'g': (870, 400, 1140, 460, '데이터분석팀'),
    },
    edges=[
        ('a', 'b', 'b', 't', None),
        ('a', 'b', 'c', 't', None),
        ('a', 'b', 'd', 't', None),
        ('b', 'b', 'e', 't', None),
        ('b', 'b', 'f', 't', None),
        ('d', 'b', 'g', 't', None),
    ],
    size=(1200, 520),
)
TREE = dict(
    title='문서 분류 체계',
    owner='총무팀 문서관리 기준',
    nodes={
        'a': (480, 90, 720, 150, '사내 문서'),
        'b': (140, 240, 380, 300, '규정'),
        'c': (820, 240, 1060, 300, '양식'),
        'd': (20, 400, 230, 460, '인사 규정'),
        'e': (290, 400, 500, 460, '회계 규정'),
        'f': (820, 400, 1060, 460, '신청서'),
    },
    edges=[
        ('a', 'b', 'b', 't', None),
        ('a', 'b', 'c', 't', None),
        ('b', 'b', 'd', 't', None),
        ('b', 'b', 'e', 't', None),
        ('c', 'b', 'f', 't', None),
    ],
    size=(1200, 520),
)
FLOW_PURCHASE = dict(
    title='비품 구매 절차',
    owner='담당: 구매팀',
    nodes={
        'start': (460, 90, 740, 150, '구매 요청'),
        'b': (480, 210, 720, 270, '예산 범위 내?'),
        'c': (860, 210, 1120, 270, '예산 추가 품의'),
        'd': (460, 340, 740, 400, '견적 비교'),
        'end': (460, 460, 740, 520, '발주'),
    },
    edges=[
        ('start', 'b', 'b', 't', None),
        ('b', 'r', 'c', 'l', '아니오'),
        ('b', 'b', 'd', 't', '예'),
        ('d', 'b', 'end', 't', None),
    ],
    size=(1200, 580),
)
HOLDOUT2 = {
    'K01_org': dict(
        draw=lambda: flow(**ORG2),
        texts=[
            '고객지원본부장',
            '콜센터운영실',
            '품질관리실',
            '고객분석실',
            '인바운드팀',
            '아웃바운드팀',
            '데이터분석팀',
        ],
        numbers=[],
        edges=[
            ('고객지원본부장', '콜센터운영실'),
            ('고객지원본부장', '품질관리실'),
            ('고객지원본부장', '고객분석실'),
            ('콜센터운영실', '인바운드팀'),
            ('콜센터운영실', '아웃바운드팀'),
            ('고객분석실', '데이터분석팀'),
        ],
    ),
    'K02_tree': dict(
        draw=lambda: flow(**TREE),
        texts=['사내 문서', '규정', '양식', '인사 규정', '회계 규정', '신청서'],
        numbers=[],
        edges=[
            ('사내 문서', '규정'),
            ('사내 문서', '양식'),
            ('규정', '인사 규정'),
            ('규정', '회계 규정'),
            ('양식', '신청서'),
        ],
    ),
    'K03_flow': dict(
        draw=lambda: flow(**FLOW_PURCHASE),
        texts=['구매 요청', '예산 범위 내', '예산 추가 품의', '견적 비교', '발주'],
        numbers=[],
        edges=[
            ('구매 요청', '예산 범위'),
            ('예산 범위', '예산 추가 품의'),
            ('예산 범위', '견적 비교'),
            ('견적 비교', '발주'),
        ],
    ),
    'K04_bar': dict(
        draw=lambda: grouped_bar(
            '직급별 평균 교육 시간',
            ['사원', '대리', '과장', '차장'],
            [('2025년', [42, 38, 31, 25]), ('2026년', [45, 40, None, 28])],
            '시간',
            '교육 시간',
            '출처: 인재개발팀, 2026.09 (과장 2026년은 집계 중)',
        ),
        texts=['직급별 평균 교육 시간', '사원', '대리', '과장', '차장', '2025년', '2026년'],
        numbers=['42', '38', '31', '25', '45', '40', '28'],
        pairs=[('사원', '42'), ('사원', '45'), ('차장', '25'), ('차장', '28')],
    ),
}
