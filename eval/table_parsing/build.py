"""표 정의 → 4경로 문서 생성. 정의 하나에서 img(표 이미지 PDF)·pdf(텍스트 PDF)·docx·xlsx를 같은 값으로 만든다.

용법: python -m eval.table_parsing.build <out_dir>
  → <out_dir>/<id>.<path>.(pdf|docx|xlsx) + <out_dir>/png/<id>.png(이미지 원본)

경로별 표현 한계(정의의 `paths`로 제한한 축 외에 여기서 조용히 바뀌는 것):
- pdf(reportlab): 셀 안 줄바꿈은 Paragraph, 병합은 SPAN, 긴 표는 쪽을 넘기며 헤더 반복(repeatRows=1), 표 안의 표는 Table 중첩.
  음영은 BACKGROUND. 글꼴은 CID 한글(HYGothic-Medium) — 글자 모양이 이미지 경로(AppleSDGothicNeo)와 다르다.
- docx(python-docx): 병합은 cell.merge, 줄바꿈은 문단 추가, 표 안의 표는 cell.add_table. 음영·세로쓰기는 안 넣는다.
- xlsx(openpyxl): 병합은 merge_cells, 줄바꿈은 wrap_text, 세로쓰기는 textRotation=90, 날짜·숫자는 **값 타입 그대로**(xlsx_types 축 — 수식은 openpyxl이 캐시값을 못 써 제외).
  '나란히' 축은 한 시트에 빈 열 2개를 두고 두 표를 놓는다(docling gap_tolerance=0이 둘을 가른다 — #220 결정 9).
"""
import datetime as dt
import sys
from pathlib import Path

from docx import Document as DocxDocument
from docx.shared import Pt
import openpyxl
from openpyxl.styles import Alignment, PatternFill
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from eval.table_parsing.render_image import cell_text, render_spec
from eval.table_parsing.specs import SPECS, paths_of, tables_of

pdfmetrics.registerFont(UnicodeCIDFont('HYGothic-Medium'))
_BODY = ParagraphStyle('b', fontName='HYGothic-Medium', fontSize=10, leading=14)
_CELL = ParagraphStyle('c', fontName='HYGothic-Medium', fontSize=9, leading=12)
_INTRO = '아래 표는 사내 기준을 정리한 것이다.'   # 쪽에 글자가 표만 있으면 레이아웃 모델이 흔들려 짧은 본문을 둔다


def _span_cells(rows):
    """(r, c, rs, cs) 목록 — 병합 원점만."""
    out = []
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if isinstance(cell, dict) and (cell.get('rs', 1) > 1 or cell.get('cs', 1) > 1):
                out.append((r, c, cell.get('rs', 1), cell.get('cs', 1)))
    return out


# ── img: PNG를 PDF에 붙인다 ─────────────────────────────────────────────────────

def build_img_pdf(spec, out: Path, png_dir: Path) -> Path:
    im = render_spec(spec)
    png = png_dir / f"{spec['id']}.png"
    im.save(png)
    width = A4[0] * 0.8
    height = width * im.height / im.width
    if height > A4[1] * 0.75:                       # 긴 표 이미지는 쪽 높이에 맞춘다
        height = A4[1] * 0.75
        width = height * im.width / im.height
    p = out / f"{spec['id']}.img.pdf"
    SimpleDocTemplate(str(p), pagesize=A4).build([Paragraph(_INTRO, _BODY), Spacer(1, 6), RLImage(str(png), width=width, height=height)])
    return p


# ── pdf: reportlab 네이티브 표 ──────────────────────────────────────────────────

def _rl_table(rows, *, nested_ok=True):
    data = []
    for row in rows:
        line = []
        for cell in row:
            if isinstance(cell, dict) and cell.get('nested') and nested_ok:
                line.append(_rl_table(cell['nested'], nested_ok=False))
            else:
                line.append(Paragraph(cell_text(cell).replace('\n', '<br/>'), _CELL))
        data.append(line)
    t = Table(data, repeatRows=1)
    style = [('FONT', (0, 0), (-1, -1), 'HYGothic-Medium', 9), ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
             ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]
    for r, c, rs, cs in _span_cells(rows):
        style.append(('SPAN', (c, r), (c + cs - 1, r + rs - 1)))
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if isinstance(cell, dict) and cell.get('bg'):
                style.append(('BACKGROUND', (c, r), (c, r), colors.HexColor(cell['bg'])))
    t.setStyle(TableStyle(style))
    return t


def build_text_pdf(spec, out: Path) -> Path:
    story = [Paragraph(_INTRO, _BODY), Spacer(1, 6)]
    if spec.get('title'):
        story.append(Paragraph(spec['title'], _BODY))
    if spec.get('side_by_side'):
        inner = [_rl_table(t) for t in spec['side_by_side']]
        wrap = Table([inner], hAlign='LEFT')
        wrap.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 12)]))
        story.append(wrap)
    else:
        story.append(_rl_table(spec['rows']))
    if spec.get('note'):
        story += [Spacer(1, 4), Paragraph(spec['note'], _BODY)]
    p = out / f"{spec['id']}.pdf.pdf"
    SimpleDocTemplate(str(p), pagesize=A4, leftMargin=15 * mm, rightMargin=15 * mm).build(story)
    return p


# ── docx ────────────────────────────────────────────────────────────────────────

def _docx_fill(table, rows):
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            tc = table.cell(r, c)
            if isinstance(cell, dict) and cell.get('nested'):
                n = cell['nested']
                inner = tc.add_table(rows=len(n), cols=max(len(x) for x in n))
                inner.style = 'Table Grid'
                _docx_fill(inner, n)
                continue
            lines = cell_text(cell).split('\n')
            tc.paragraphs[0].text = lines[0]
            for extra in lines[1:]:
                tc.add_paragraph(extra)
    for r, c, rs, cs in _span_cells(rows):
        table.cell(r, c).merge(table.cell(r + rs - 1, c + cs - 1))


def build_docx(spec, out: Path) -> Path:
    d = DocxDocument()
    d.add_paragraph(_INTRO)
    if spec.get('title'):
        d.add_paragraph(spec['title'])
    rows = spec['rows']
    t = d.add_table(rows=len(rows), cols=max(len(r) for r in rows))
    t.style = 'Table Grid'
    _docx_fill(t, rows)
    if spec.get('note'):
        d.add_paragraph(spec['note'])
    p = out / f"{spec['id']}.docx.docx"
    d.save(p)
    return p


# ── xlsx ────────────────────────────────────────────────────────────────────────

def _xlsx_value(cell, typed: bool):
    if cell is None:
        return None
    raw = cell.get('t') if isinstance(cell, dict) else cell
    if typed and isinstance(raw, str) and len(raw) == 10 and raw[4] == '-' and raw[7] == '-':
        return dt.date.fromisoformat(raw)               # 날짜 타입 셀
    return raw if raw != '' else None


def _xlsx_fill(ws, rows, *, typed: bool, col_offset: int = 0):
    for r, row in enumerate(rows, start=1):
        for c, cell in enumerate(row, start=1 + col_offset):
            v = _xlsx_value(cell, typed)
            if v is None:
                continue
            x = ws.cell(row=r, column=c, value=v)
            if isinstance(cell, dict):
                if cell.get('bg'):
                    x.fill = PatternFill('solid', fgColor=cell['bg'].lstrip('#'))
                if cell.get('v'):
                    x.alignment = Alignment(textRotation=90)
            if isinstance(v, str) and '\n' in v:
                x.alignment = Alignment(wrap_text=True)
    for r, c, rs, cs in _span_cells(rows):
        ws.merge_cells(start_row=r + 1, start_column=c + 1 + col_offset, end_row=r + rs, end_column=c + cs + col_offset)


def build_xlsx(spec, out: Path) -> Path:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '표'
    typed = bool(spec.get('xlsx_types'))
    if spec.get('side_by_side'):
        off = 0
        for t in spec['side_by_side']:
            _xlsx_fill(ws, t, typed=typed, col_offset=off)
            off += max(len(r) for r in t) + 2
    else:
        _xlsx_fill(ws, spec['rows'], typed=typed)
    p = out / f"{spec['id']}.xlsx.xlsx"
    wb.save(p)
    return p


BUILDERS = {'img': lambda s, o: build_img_pdf(s, o, o / 'png'), 'pdf': build_text_pdf,
            'docx': build_docx, 'xlsx': build_xlsx}


def build_all(out: Path, only: set[str] | None = None, paths: set[str] | None = None) -> list[tuple[str, str, Path]]:
    out.mkdir(parents=True, exist_ok=True)
    (out / 'png').mkdir(exist_ok=True)
    made = []
    for spec in SPECS:
        if only and spec['id'] not in only:
            continue
        for path in paths_of(spec):
            if paths and path not in paths:
                continue
            made.append((spec['id'], path, BUILDERS[path](spec, out)))
    return made


if __name__ == '__main__':
    made = build_all(Path(sys.argv[1]))
    print(f'{len(made)}개 생성 — 정의 {len(SPECS)}개, 표 {sum(len(tables_of(s)) for s in SPECS)}개')
