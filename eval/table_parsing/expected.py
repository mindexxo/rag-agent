"""표 정의 → 기대 결과. 두 층(#246 결정 3):
  L1 `expected_cells`: docling이 복원해야 할 셀 격자 — (행, 열, 세로 병합, 가로 병합, 글자). 경로와 무관하다.
  L2 `expected_rows`: 우리 렌더(rag/chunking)가 내놓아야 할 markdown 행 — **경로마다 규칙이 다르다**(아래).

L2 규칙은 코드의 현재 **설계**를 옮긴 것이다 — 정답이 아니라 "지금 설계대로면 이렇게 나온다"다. 규칙 자체가 나쁘면
(예: 가로 머리글 표의 첫 행을 헤더로 취급) 그것도 결과에 드러나야 하므로 여기서 보정하지 않는다.
- pdf·img (`_docling_table_markdown`, export_to_dataframe): 헤더 행들을 '.'로 이어 열 이름으로, 병합 칸은 덮인 자리마다
  같은 글자를 반복. header_rows=0(가로 머리글)이면 dataframe 열 이름이 0,1,2…가 된다.
- docx·xlsx (`docling_grid_markdown`): 첫 행이 헤더, 가로 병합은 첫 열에만 글자, 세로 병합은 행마다 반복, 값이 하나도 없는
  행은 뺌(#220 결정 10).
- 셀 안 줄바꿈은 **공백으로 접은 것을 기대**한다(markdown 표는 행 = 한 줄이므로 그래야 행이 보존된다). 현재 코드는
  docx·xlsx에서 `\n`을 그대로 내보내 행이 깨진다 — 그 차이가 L2 결함 C(#252)로 집계되는 것이 의도다.
"""
import re
from dataclasses import dataclass

from eval.table_parsing.render_image import cell_text


@dataclass(frozen=True)
class Cell:
    row: int
    col: int
    row_span: int
    col_span: int
    text: str


def norm(text) -> str:
    """비교용 정규화 — NBSP·연속 공백·줄바꿈을 공백 하나로, 앞뒤 공백 제거. 글자 자체는 바꾸지 않는다."""
    return re.sub(r'[\s ]+', ' ', str(text if text is not None else '')).strip()


def is_numeric(text: str) -> bool:
    """숫자·금액·날짜·비율 칸 — 한 글자라도 틀리면 실패로 따로 세는 대상(이슈 완료 기준)."""
    return bool(re.search(r'\d', text))


def flat(rows) -> tuple[int, int]:
    return len(rows), max(len(r) for r in rows)


def expected_cells(rows) -> list[Cell]:
    """정의 격자 → 병합 원점 셀 목록. 표 안의 표(nested)는 바깥 칸을 빈 글자로 두고, 안쪽 표는 별도 표로 센다(expected_tables)."""
    out = []
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            if cell is None:
                continue
            rs = cell.get('rs', 1) if isinstance(cell, dict) else 1
            cs = cell.get('cs', 1) if isinstance(cell, dict) else 1
            text = '' if isinstance(cell, dict) and cell.get('nested') else cell_text(cell)
            out.append(Cell(r, c, rs, cs, norm(text)))
    return out


def expected_tables(spec) -> list[list[list]]:
    """정의 하나가 담는 표(들)의 격자 목록 — 나란히 축은 2개, 중첩 축은 바깥 + 안쪽."""
    if spec.get('side_by_side'):
        return list(spec['side_by_side'])
    tables = [spec['rows']]
    for row in spec['rows']:
        for cell in row:
            if isinstance(cell, dict) and cell.get('nested'):
                tables.append(cell['nested'])
    return tables


def _filled_grid(rows) -> list[list[str]]:
    """병합 칸을 덮인 자리마다 반복한 글자 격자(docling grid와 같은 모양)."""
    n, m = flat(rows)
    g = [[''] * m for _ in range(n)]
    for cell in expected_cells(rows):
        for dr in range(cell.row_span):
            for dc in range(cell.col_span):
                g[cell.row + dr][cell.col + dc] = cell.text
    return g


def expected_rows(spec, rows, path: str) -> list[list[str]]:
    """L2 — 경로별 렌더 규칙을 적용한 markdown 행(첫 행 = 헤더)."""
    g = _filled_grid(rows)
    if path in ('pdf', 'img'):
        h = spec.get('header_rows', 1)
        if h == 0:
            header = [str(i) for i in range(len(g[0]))]
            body = g
        else:
            header = [''] * len(g[0])          # docling export_to_dataframe의 열 이름 규칙 그대로 — 두 번째 행부터 '.'로 잇는다
            for i in range(h):                 # (세로 병합 헤더는 grid가 반복하므로 '직급.직급'이 된다 — 현재 동작)
                for j, name in enumerate(g[i]):
                    header[j] += f'.{name}' if header[j] != '' else name
            body = g[h:]
        return [header] + body
    # docx·xlsx: grid 규칙
    out = []
    for r, row in enumerate(rows):
        line = []
        for c, cell in enumerate(row):
            if cell is None:
                # 덮인 자리: 세로 병합이면 원점 글자 반복, 가로 병합이면 빈 칸
                origin = _origin_of(rows, r, c)
                line.append(g[r][c] if origin and origin[1] == c else '')
            else:
                line.append(norm('' if isinstance(cell, dict) and cell.get('nested') else cell_text(cell)))
        if any(line):
            out.append(line)
    return out


def _origin_of(rows, r, c):
    for cell in expected_cells(rows):
        if cell.row <= r < cell.row + cell.row_span and cell.col <= c < cell.col + cell.col_span:
            return cell.row, cell.col
    return None
