"""표 정의 → 기대 결과. 두 층(#246 결정 3):
  L1 `expected_cells`: docling이 복원해야 할 셀 격자 — (행, 열, 세로 병합, 가로 병합, 글자). 경로와 무관하다.
  L2 `expected_rows`: 우리 렌더(rag/chunking)가 내놓아야 할 markdown 행 — **경로마다 규칙이 다르다**(아래).

L2 규칙은 코드의 현재 **설계**를 옮긴 것이다 — 정답이 아니라 "지금 설계대로면 이렇게 나온다"다. 규칙 자체가 나쁘면
그것도 결과에 드러나야 하므로 여기서 보정하지 않는다.
- 전 경로 공통(#252 결정 2로 PDF·이미지도 `docling_grid_rows`): 첫 행이 헤더, 병합 칸은 가로·세로 모두
  칸마다 반복(#252 결정 9 — 그 전엔 가로 병합은 첫 열에만), 값이 하나도 없는 행은 뺌(#220 결정 10). 셀 안 줄바꿈은 공백으로(#252 C), 엑셀 날짜 `00:00:00` 꼬리 없음(F).
- 표 안의 표는 어느 경로든 **평탄화**(flatten_nested — PDF는 TableFormer가, DOCX는 docx.docx_table_rows가 같은 모양으로).
- (#252 전: PDF는 export_to_dataframe 경로라 2단 헤더를 '.'로 잇고 column_header가 없으면 열 이름이 0,1,2…였다 — 실문서
  PDF 표 55%가 숫자 헤더로 색인되던 결함 A의 원인. 그 규칙은 지웠다.)
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


def flatten_nested(rows) -> list[list]:
    """표 안의 표를 **docling PDF(TableFormer)가 내는 모양**으로 편다(#253 확인, 2026-10-17): 안쪽 표가 든 열은 안쪽 열 수만큼
    넓어지고, 안쪽 표가 든 행은 안쪽 행 수만큼 늘어난다. 그 행의 다른 칸은 세로 병합(안쪽 행 수), 다른 행의 그 열 칸은
    가로 병합(안쪽 열 수). 예: 3×2(안쪽 3×2) → 5×3 — '평가 항목'이 안쪽 3행에 반복, '세부'·'매년 12월'이 2열에 걸침.
    값은 전부 보존되고 바깥 칸 글자가 안쪽 행마다 붙는다 — 따로 떼는 것보다 검색에 낫다(수정 대상 아님)."""
    nested = [(r, c, cell['nested']) for r, row in enumerate(rows) for c, cell in enumerate(row)
              if isinstance(cell, dict) and cell.get('nested')]
    if not nested:
        return rows
    r0, c0, inner = nested[0]                      # 중첩 하나만 지원(축 15)
    R, C = len(inner), max(len(x) for x in inner)
    out = []
    for r, row in enumerate(rows):
        if r != r0:
            new = list(row[:c0]) + [dict(t=cell_text(row[c0]), cs=C) if row[c0] is not None else None] + [None] * (C - 1) + list(row[c0 + 1:])
            out.append(new)
            continue
        for i, irow in enumerate(inner):
            new = []
            for c, cell in enumerate(row):
                if c == c0:
                    new.extend(list(irow) + [None] * (C - len(irow)))
                elif i == 0:
                    new.append(dict(t=cell_text(cell), rs=R) if cell is not None else None)
                else:
                    new.append(None)
            out.append(new)
    return out


def expected_tables(spec, path: str | None = None, layer: str = 'l2') -> list[list[list]]:
    """정의 하나가 담는 표(들)의 격자 목록 — 나란히 축은 2개. 중첩 축은 층·경로에 따라 다르다:
    - L2(색인 결과)는 어느 경로든 평탄화된 1개(flatten_nested — PDF는 TableFormer가, DOCX는 rag/chunking/docx.docx_table_rows가
      같은 모양으로, #252 결정 4).
    - L1(docling 추출)은 docling이 내는 모양 그대로: PDF·이미지는 TableFormer 평탄화 1개, DOCX는 바깥(칸 비움) + 안쪽 2개."""
    if spec.get('side_by_side'):
        return list(spec['side_by_side'])
    has_nested = any(isinstance(cell, dict) and cell.get('nested') for row in spec['rows'] for cell in row)
    if not has_nested:
        return [spec['rows']]
    if layer == 'l1' and path == 'docx':
        return [spec['rows']] + [cell['nested'] for row in spec['rows'] for cell in row if isinstance(cell, dict) and cell.get('nested')]
    return [flatten_nested(spec['rows'])]


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
    """L2 — 렌더 규칙(전 경로 공통 grid 규칙)을 적용한 markdown 행(첫 행 = 헤더). path는 호환용(규칙이 같다)."""
    g = _filled_grid(rows)
    out = []
    for r, row in enumerate(rows):
        line = []
        for c, cell in enumerate(row):
            if cell is None:
                line.append(g[r][c])          # 덮인 자리: 가로·세로 병합 모두 원점 글자 반복(#252 결정 9)
            else:
                line.append(norm('' if isinstance(cell, dict) and cell.get('nested') else cell_text(cell)))
        if any(line):
            out.append(line)
    return out
