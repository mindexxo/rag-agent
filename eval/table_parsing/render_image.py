"""표 정의 → PNG(표 이미지 경로의 원본). PIL로 격자·병합·음영·세로쓰기·중첩·기울임·JPEG 열화를 그린다.

폰트는 맥 기본 한글 폰트(AppleSDGothicNeo). 리눅스에서 돌리려면 FONT_PATH를 나눔고딕 등으로 바꾼다 —
글꼴이 바뀌면 OCR 결과도 바뀌므로 측정 보고에 글꼴을 적는다.
"""

import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
PAD, LINE = 14, 1.35
Rows = list[list[Any]]  # 정의 격자 — 칸은 str/int/dict/None


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size)


def cell_text(c: Any) -> str:
    if c is None:
        return ''
    return str(c) if not isinstance(c, dict) else c.get('t', '')


_MEASURE = ImageDraw.Draw(Image.new('RGB', (10, 10)))


def _text_size(text: str, f: ImageFont.FreeTypeFont, vertical: bool = False) -> tuple[float, float]:
    lines = text.split('\n') or ['']
    w = max(_MEASURE.textlength(line, font=f) for line in lines)
    h = len(lines) * f.size * LINE
    return (h, w) if vertical else (w, h)


def _span(cell: Any) -> tuple[int, int]:
    """(rs, cs) — 병합 칸의 행·열 폭."""
    if isinstance(cell, dict):
        return cell.get('rs', 1), cell.get('cs', 1)
    return 1, 1


def _nested_images(rows: Rows, font_size: int) -> dict[tuple[int, int], Image.Image]:
    return {
        (r, c): render_table(cell['nested'], font_size=font_size - 2)
        for r, row in enumerate(rows)
        for c, cell in enumerate(row)
        if isinstance(cell, dict) and cell.get('nested')
    }


def _grid_sizes(
    rows: Rows, nested_imgs: dict[tuple[int, int], Image.Image], f: ImageFont.FreeTypeFont
) -> tuple[list[float], list[float]]:
    """열 너비·행 높이 — 병합 안 된 칸만 크기를 정한다."""
    col_w: list[float] = [0] * max(len(r) for r in rows)
    row_h: list[float] = [0] * len(rows)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            rs, cs = _span(cell)
            if (r, c) in nested_imgs:
                w, h = nested_imgs[(r, c)].size
            else:
                w, h = _text_size(cell_text(cell), f, isinstance(cell, dict) and cell.get('v'))
            if cs == 1:
                col_w[c] = max(col_w[c], w + 2 * PAD)
            if rs == 1:
                row_h[r] = max(row_h[r], h + 2 * PAD)
    return [max(w, 40) for w in col_w], [max(h, 30) for h in row_h]


def _edges(sizes: list[float]) -> list[float]:
    out: list[float] = [1]
    for s in sizes:
        out.append(out[-1] + s)
    return out


def _draw_frame(d: ImageDraw.ImageDraw, box: list[float], bg: str | None, style: str) -> None:
    if bg:
        d.rectangle(box, fill=bg)
    x0, y0, x1, y1 = box
    if style == 'grid':
        d.rectangle(box, outline='black', width=1)
    elif style == 'hlines':
        d.line([x0, y1, x1, y1], fill='black', width=1)
        d.line([x0, y0, x1, y0], fill='black', width=1)
    elif style == 'striped':
        d.line([x0, y1, x1, y1], fill='#999', width=1)


def _draw_text(im: Image.Image, cell: Any, box: list[float], bg: str | None, f: ImageFont.FreeTypeFont) -> None:
    """칸 글자 — 세로쓰기('v')는 가로로 그린 뒤 90도 돌려 칸 가운데에 붙인다."""
    x0, y0, x1, y1 = box
    text = cell_text(cell)
    if isinstance(cell, dict) and cell.get('v'):
        tw, th = _text_size(text, f)
        t = Image.new('RGB', (int(tw) + 4, int(th) + 4), bg or 'white')
        ImageDraw.Draw(t).text((2, 2), text, fill='black', font=f)
        t = t.rotate(90, expand=True)
        im.paste(t, (int((x0 + x1 - t.width) / 2), int((y0 + y1 - t.height) / 2)))
    else:
        ImageDraw.Draw(im).multiline_text((x0 + PAD, y0 + PAD), text, fill='black', font=f, spacing=4)


def render_table(rows: Rows, *, font_size: int = 16, style: str = 'grid', row_header: bool = False) -> Image.Image:
    f = font(font_size)
    nested_imgs = _nested_images(rows, font_size)
    col_w, row_h = _grid_sizes(rows, nested_imgs, f)
    W, Hh = int(sum(col_w)) + 2, int(sum(row_h)) + 2
    im = Image.new('RGB', (W, Hh), 'white')
    d = ImageDraw.Draw(im)
    xs, ys = _edges(col_w), _edges(row_h)
    covered = set()
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None or (r, c) in covered:
                continue
            rs, cs = _span(cell)
            covered |= {(r + dr, c + dc) for dr in range(rs) for dc in range(cs) if (dr, dc) != (0, 0)}
            box = [xs[c], ys[r], xs[c + cs], ys[r + rs]]
            bg = cell.get('bg') if isinstance(cell, dict) else None
            if style == 'striped' and r > 0 and r % 2 == 0 and not bg:
                bg = '#f2f2f2'
            _draw_frame(d, box, bg, style)
            if (r, c) in nested_imgs:
                im.paste(nested_imgs[(r, c)], (int(box[0] + PAD), int(box[1] + PAD)))
            else:
                _draw_text(im, cell, box, bg, f)
    d.rectangle([0, 0, W - 1, Hh - 1], outline='black', width=1)
    return im


def render_spec(spec: dict[str, Any]) -> Image.Image:
    """정의 하나 → 표 이미지(제목·주석·나란히·기울임·축소·JPEG 열화 포함)."""
    fs = spec.get('font_size', 16)
    if spec.get('side_by_side'):
        parts = [render_table(t, font_size=fs) for t in spec['side_by_side']]
        W = sum(p.width for p in parts) + 40 * (len(parts) + 1)
        Hh = max(p.height for p in parts) + 40
        im = Image.new('RGB', (W, Hh), 'white')
        x = 40
        for p in parts:
            im.paste(p, (x, 20))
            x += p.width + 40
        return im
    table = render_table(
        spec['rows'], font_size=fs, style=spec.get('style', 'grid'), row_header=spec.get('row_header', False)
    )
    title, note = spec.get('title'), spec.get('note')
    extra = (fs * 2 if title else 0) + (fs * 2 if note else 0)
    tmp = ImageDraw.Draw(Image.new('RGB', (10, 10)))
    tw = max([table.width] + [int(tmp.textlength(t, font=font(fs))) for t in (title, note) if t])
    im = Image.new('RGB', (tw + 40, table.height + 40 + extra), 'white')
    d = ImageDraw.Draw(im)
    y = 20
    if title:
        d.text((20, y), title, fill='black', font=font(fs))
        y += fs * 2
    im.paste(table, (20, y))
    y += table.height + 8
    if note:
        d.text((20, y), note, fill='black', font=font(fs - 2))
    if spec.get('skew'):
        im = im.rotate(spec['skew'], expand=True, fillcolor='white', resample=Image.BICUBIC)
    if spec.get('scale'):
        im = im.resize((int(im.width * spec['scale']), int(im.height * spec['scale'])), Image.LANCZOS)
    if spec.get('jpeg_quality'):
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=spec['jpeg_quality'])
        im = Image.open(buf).convert('RGB')
    return im
