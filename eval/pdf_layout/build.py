"""#244 배치 유형별 합성 PDF 렌더 — specs.SPECS를 reportlab로 그린다(정본). 글꼴은 표 축과 같은 CID 한글(HYGothic-Medium).

다단은 BaseDocTemplate + Frame 여러 개, 단 넘김은 FrameBreak. 그림 옆 글 흐름은 ImageAndFlowables. 가로 쪽은 PageTemplate(pagesize=landscape).
머리글·바닥글은 onPage 콜백으로 그린다(본문 프레임 밖). 사이드바는 본문 2/3·상자 1/3 프레임.
용법: python -m eval.pdf_layout.build [--out DIR]   → DIR/<id>.pdf
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.pdf_layout.specs import SPECS  # noqa: E402

SIZE, LEADING = 11.6, 13.8  # 실문서 규정 실측(#240)


def _styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont

    pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
    p = ParagraphStyle('p', fontName='HYGothic-Medium', fontSize=SIZE, leading=LEADING, spaceBefore=6)
    h = ParagraphStyle('h', parent=p, fontSize=13, leading=17, spaceBefore=12, spaceAfter=4)
    t = ParagraphStyle('t', parent=p, fontSize=16, leading=22, spaceAfter=12)
    c = ParagraphStyle('c', fontName='HYGothic-Medium', fontSize=9.5, leading=12)
    return p, h, t, c


def _png(tmp: Path) -> Path:
    """내용이 있는 그림 1장 — 단색 사각형은 레이아웃 모델이 그림으로 잡지 않는다(첫 측정: pictures 0, 그래서 '그림 옆 글 흐름 정상'이
    자명한 통과였다, 리뷰 지적 4). 상자·화살표·라벨이 든 구성도 꼴로 그린다."""
    from PIL import Image, ImageDraw

    f = tmp / 'fig.png'
    if not f.exists():
        im = Image.new('RGB', (600, 400), 'white')
        d = ImageDraw.Draw(im)
        for i, (x, label) in enumerate([(40, 'A'), (230, 'B'), (420, 'C')]):
            d.rectangle([x, 120, x + 140, 280], outline='black', width=4, fill=(220, 230, 250))
            d.text((x + 60, 190), label, fill='black')
            if i < 2:
                d.line([x + 140, 200, x + 190, 200], fill='black', width=4)
                d.polygon([(x + 190, 190), (x + 205, 200), (x + 190, 210)], fill='black')
        d.ellipse([250, 20, 350, 90], outline='black', width=4)
        im.save(f)
    return f


def _flow(block, styles, tmp: Path):
    from reportlab.lib import colors
    from reportlab.platypus import FrameBreak, Image, Paragraph, Table, TableStyle

    p, h, t, c = styles
    kind = block[0]
    if kind == 'p':
        return Paragraph(block[1], p)
    if kind == 'h':
        return Paragraph(block[1], h)
    if kind == 'break':
        return FrameBreak()
    if kind == 'img':
        return Image(str(_png(tmp)), width=170, height=113)
    if kind == 'table':
        tb = Table([[Paragraph(x, c) for x in row] for row in block[1]], repeatRows=1)
        tb.setStyle(
            TableStyle(
                [('GRID', (0, 0), (-1, -1), 0.5, colors.black), ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey)]
            )
        )
        return tb
    raise ValueError(kind)


def _columns(w, h, n, gap, margin=15 * 2.835):
    from reportlab.platypus import Frame

    cw = (w - 2 * margin - gap * (n - 1)) / n
    return [Frame(margin + i * (cw + gap), margin, cw, h - 2 * margin, id=f'c{i}') for i in range(n)]


def build(spec: dict, out_dir: Path) -> Path:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        ImageAndFlowables,
        NextPageTemplate,
        PageTemplate,
        Paragraph,
        SimpleDocTemplate,
    )

    styles = _styles()
    p, h, t, _c = styles
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{spec['id']}.pdf"
    W, H = A4
    layout = spec['layout']
    blocks = spec['blocks']
    title = Paragraph(spec['title'], t)

    if layout == 'image_wrap':
        story = [title, Paragraph(blocks[0][1], h)]
        from reportlab.platypus import Image

        side = [Paragraph(b[1], p) for b in blocks[2:5]]
        story.append(
            ImageAndFlowables(
                Image(str(_png(out_dir)), width=170, height=113), side, imageSide='right', imageLeftPadding=8
            )
        )
        story += [Paragraph(b[1], p) for b in blocks[5:]]
        SimpleDocTemplate(str(path), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm).build(story)
        return path

    if layout == 'header_footer':
        head, _ = spec['furniture']

        def on_page(canvas, doc):
            canvas.saveState()
            canvas.setFont('HYGothic-Medium', 9)
            canvas.drawString(15 * mm, H - 12 * mm, head)
            canvas.drawRightString(W - 15 * mm, 10 * mm, f'- {doc.page} -')
            canvas.restoreState()

        doc = BaseDocTemplate(str(path), pagesize=A4)
        doc.addPageTemplates([PageTemplate('one', [Frame(15 * mm, 20 * mm, W - 30 * mm, H - 40 * mm)], onPage=on_page)])
        doc.build([title] + [_flow(b, styles, out_dir) for b in blocks])
        return path

    doc = BaseDocTemplate(str(path), pagesize=A4)
    if layout == 'two_col':
        gap = spec.get('gap_mm', 10) * mm
        doc.addPageTemplates([PageTemplate('two', _columns(W, H - 25 * mm, 2, gap))])
        story = [title]  # 제목은 첫 단 머리에 — 단 전폭 제목 프레임은 두지 않는다(흔한 워드 2단 조판)
    elif layout == 'three_col':
        doc = BaseDocTemplate(str(path), pagesize=landscape(A4))
        LW, LH = landscape(A4)
        doc.addPageTemplates([PageTemplate('three', _columns(LW, LH, 3, 8 * mm))])
        story = [title]
    elif layout == 'mixed_orient':
        LW, LH = landscape(A4)
        doc.addPageTemplates(
            [
                PageTemplate('port', [Frame(15 * mm, 15 * mm, W - 30 * mm, H - 30 * mm)], pagesize=A4),
                PageTemplate('land', [Frame(15 * mm, 15 * mm, LW - 30 * mm, LH - 30 * mm)], pagesize=landscape(A4)),
            ]
        )
        story = [title]
    elif layout == 'two_col_table':
        gap, m = 10 * mm, 15 * mm
        cw = (W - 2 * m - gap) / 2
        top = [
            Frame(m + i * (cw + gap), 190 * mm, cw, H - m - 190 * mm, id=f'top{i}') for i in range(2)
        ]  # 위 2단: 190mm~상단
        tbl = Frame(m, 110 * mm, W - 2 * m, 75 * mm, id='tbl')  # 전폭 표
        bot = [Frame(m + i * (cw + gap), m, cw, 90 * mm, id=f'bot{i}') for i in range(2)]  # 아래 2단
        doc.addPageTemplates([PageTemplate('mix', top + [tbl] + bot)])
        story = [title]
    elif layout == 'sidebar':
        main = Frame(15 * mm, 15 * mm, (W - 30 * mm) * 0.64, H - 30 * mm, id='main')
        side = Frame(
            15 * mm + (W - 30 * mm) * 0.68, 15 * mm, (W - 30 * mm) * 0.32, H - 30 * mm, id='side', showBoundary=1
        )
        doc.addPageTemplates([PageTemplate('sb', [main, side])])
        story = [title]
    else:
        raise ValueError(layout)

    page_switch = iter(['land', 'port']) if layout == 'mixed_orient' else None
    for b in blocks:
        if b[0] == 'break' and page_switch is not None:
            story += [NextPageTemplate(next(page_switch)), _flow(('break',), styles, out_dir)]
            continue
        story.append(_flow(b, styles, out_dir))
    doc.build(story)
    return path


def build_all(out_dir: Path) -> list[Path]:
    return [build(s, out_dir) for s in SPECS]


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='eval/results/pdf_layout/docs')
    for pth in build_all(Path(ap.parse_args().out)):
        print(pth)
