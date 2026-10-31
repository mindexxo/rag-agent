"""채점 — 두 층(#246 결정 3).
  L1 `score_doc`: docling 문서의 표 셀 격자(병합 범위·글자) vs 정의. 추출 정확도. docling·OCR·TableFormer의 책임.
  L2 `score_chunks`: chunk_file 결과의 markdown 행 vs 경로별 기대 행(expected.expected_rows). 색인 결과 정확도. 우리 렌더까지 포함.

오류 유형 5종(이슈 완료 기준): 미검출(표를 못 찾음) · 구조(행·열 수 불일치 — 칸 밀림 포함) · 병합(범위 불일치) ·
누락(기대 글자가 있는데 빈 칸) · 오독(둘 다 글자가 있는데 다름). 구조가 어긋나면 자리 비교가 무의미하므로
**글자 존재율**(기대 글자가 표 어딘가에 있는 비율)을 따로 내 "밀렸을 뿐 읽긴 했다"를 가른다.
"""

from dataclasses import dataclass, field

from eval.table_parsing.expected import Cell, is_numeric, norm


@dataclass
class ActualTable:
    cells: list[Cell]
    n_rows: int
    n_cols: int
    header_rows: int = 0  # docling이 column_header로 표시한 앞 행 수(L1만)

    @property
    def texts(self) -> set[str]:
        return {c.text for c in self.cells if c.text}


@dataclass
class Score:
    detected: bool = True
    dims_ok: bool = True
    exp_dims: tuple[int, int] = (0, 0)
    act_dims: tuple[int, int] = (0, 0)
    cells_total: int = 0
    cells_ok: int = 0
    numeric_total: int = 0
    numeric_ok: int = 0
    span_total: int = 0
    span_ok: int = 0
    presence_total: int = 0
    presence_ok: int = 0
    header_rows: int | None = None
    mismatches: list[dict] = field(default_factory=list)  # {'r','c','kind','expected','actual'}

    @property
    def kinds(self) -> list[str]:
        special = sorted({m['kind'] for m in self.mismatches if m['kind'] in ('변환실패', '청킹실패')})
        if not self.detected:
            return special or ['미검출']
        out = ['구조'] if not self.dims_ok else []
        out += sorted({m['kind'] for m in self.mismatches})
        return out

    @property
    def clean(self) -> bool:
        """✅ 기준 — 표를 찾았고 행·열 수가 맞고 병합 범위·모든 칸이 일치."""
        return self.detected and self.dims_ok and self.span_ok == self.span_total and self.cells_ok == self.cells_total


def docling_tables(doc) -> list[ActualTable]:
    """DoclingDocument → 표 목록(읽기 순서). 병합 원점만 남긴다 — grid는 덮인 자리마다 같은 셀을 반복한다."""
    from docling_core.types.doc import TableItem  # 지연 import — docling은 톱레벨에 안 들인다

    out = []
    for item, _ in doc.iterate_items():
        if not isinstance(item, TableItem):
            continue
        seen, cells = set(), []
        for c in item.data.table_cells:
            key = (c.start_row_offset_idx, c.start_col_offset_idx)
            if key in seen:
                continue
            seen.add(key)
            cells.append(
                Cell(
                    c.start_row_offset_idx,
                    c.start_col_offset_idx,
                    c.end_row_offset_idx - c.start_row_offset_idx,
                    c.end_col_offset_idx - c.start_col_offset_idx,
                    norm(c.text),
                )
            )
        header_rows = 0  # export_to_dataframe과 같은 셈(원점 행의 column_header만)
        for row_idx, row in enumerate(item.data.grid):
            if any(c.column_header and c.start_row_offset_idx == row_idx for c in row):
                header_rows += 1
            else:
                break
        out.append(ActualTable(cells, item.data.num_rows, item.data.num_cols, header_rows))
    return out


def _loose(text: str) -> str:
    """띄어쓰기를 뺀 글자 — '읽긴 했다'(존재율·표 짝짓기)의 기준. OCR 띄어쓰기 소실을 유실과 가르기 위해서다."""
    return text.replace(' ', '')


def score_cells(
    expected: list[Cell], actual: ActualTable | None, exp_dims: tuple[int, int], presence_pool: set[str] | None = None
) -> Score:
    """presence_pool — 존재율을 잴 글자 집합(띄어쓰기 뺀 것). 주면 짝지은 표 밖(쪽 넘김으로 갈린 다른 표 등)까지 포함한다."""
    s = Score(exp_dims=exp_dims)
    exp_texts = [c.text for c in expected if c.text]
    s.presence_total = len(exp_texts)
    pool = presence_pool if presence_pool is not None else ({_loose(t) for t in actual.texts} if actual else set())
    s.presence_ok = sum(1 for t in exp_texts if _loose(t) in pool)
    if actual is None:
        s.detected = False
        s.cells_total = len(expected)
        s.numeric_total = sum(1 for c in expected if is_numeric(c.text))
        s.span_total = sum(1 for c in expected if c.row_span > 1 or c.col_span > 1)
        return s
    s.act_dims = (actual.n_rows, actual.n_cols)
    s.header_rows = actual.header_rows
    s.dims_ok = s.exp_dims == s.act_dims
    by_pos = {(c.row, c.col): c for c in actual.cells}
    for e in expected:
        s.cells_total += 1
        numeric = is_numeric(e.text)
        s.numeric_total += numeric
        spanned = e.row_span > 1 or e.col_span > 1
        s.span_total += spanned
        a = by_pos.get((e.row, e.col))
        if a is not None and a.text != norm(a.text):  # 호출부가 정규화를 빠뜨려도 비교는 같은 기준으로
            a = Cell(a.row, a.col, a.row_span, a.col_span, norm(a.text))
        if a is None:
            if e.text:
                s.mismatches.append(dict(r=e.row, c=e.col, kind='누락', expected=e.text, actual=''))
            else:
                s.cells_ok += 1
            continue
        if spanned and (a.row_span, a.col_span) == (e.row_span, e.col_span):
            s.span_ok += 1
        elif spanned or (a.row_span, a.col_span) != (1, 1):  # 기대 병합이 깨졌거나, 기대엔 없는 병합이 생겼거나
            s.mismatches.append(
                dict(
                    r=e.row,
                    c=e.col,
                    kind='병합',
                    expected=f'{e.row_span}x{e.col_span}',
                    actual=f'{a.row_span}x{a.col_span}',
                )
            )
        if a.text == e.text:
            s.cells_ok += 1
            s.numeric_ok += numeric
        elif not a.text:
            s.mismatches.append(dict(r=e.row, c=e.col, kind='누락', expected=e.text, actual=''))
        elif not e.text:
            s.mismatches.append(dict(r=e.row, c=e.col, kind='추가', expected='', actual=a.text))
        else:
            s.mismatches.append(dict(r=e.row, c=e.col, kind='오독', expected=e.text, actual=a.text))
    return s


def _pair(expected_text_sets: list[set[str]], actual_text_sets: list[set[str]]) -> list[int | None]:
    """기대 표마다 글자(띄어쓰기 무시)가 가장 많이 겹치는 실제 표의 번호를 짝짓는다(중복 없이).
    겹침이 0이어도 기대 표가 하나뿐이고 남은 실제 표가 있으면 가장 큰 것을 짝짓는다 — OCR이 전부 틀린 표를
    '미검출'로 잘못 세지 않기 위해서다(그 경우 오독으로 집계된다)."""
    taken, out = set(), []
    for exp in expected_text_sets:
        best, best_n = None, 0
        for i, act in enumerate(actual_text_sets):
            if i in taken:
                continue
            n = len(exp & act)
            if n > best_n:
                best, best_n = i, n
        if best is None and len(expected_text_sets) == 1:
            rest = [i for i in range(len(actual_text_sets)) if i not in taken]
            if rest:
                best = max(rest, key=lambda i: len(actual_text_sets[i]))
        if best is not None:
            taken.add(best)
        out.append(best)
    return out


def match_tables(expected_sets: list[list[Cell]], actuals: list[ActualTable]) -> list[ActualTable | None]:
    idx = _pair(
        [{_loose(c.text) for c in exp if c.text} for exp in expected_sets],
        [{_loose(t) for t in a.texts} for a in actuals],
    )
    return [actuals[i] if i is not None else None for i in idx]


def all_texts(actuals: list[ActualTable]) -> set[str]:
    """존재율 풀 — 문서의 모든 표 글자(띄어쓰기 뺀 것)."""
    return {_loose(t) for a in actuals for t in a.texts}


# ── L2: 청크 markdown ──────────────────────────────────────────────────────────


def chunk_tables(chunks, *, xlsx: bool = False) -> list[list[list[str]]]:
    """청크 목록 → markdown 표(행 목록)들. 구분선('---')이 나오면 그 윗줄이 헤더인 새 표 — 단 **직전 표와 헤더가 같으면 같은
    표의 다음 행 묶음**으로 잇는다(#252: 긴 표는 행 묶음마다 헤더를 반복한다). 긴 표가 청크 경계에서 갈려 뒷 청크가 '|' 행으로
    바로 시작하면 앞 표에 잇는다. xlsx는 시트 완성본(meta.table_text)이 전체라 그것만 읽는다. PDF·DOCX 큰 표 묶음도 #257부터
    table_text를 갖지만 읽지 않는다 — 이 하네스가 재는 것이 묶음 분할(헤더 반복·행 경계)이라 묶음에서 다시 잇는다. 형식은 청크로
    가르지 않고 호출부가 알려 준다: 표만 있는 PDF는 모든 청크가 표 묶음이라 청크 모양으로는 xlsx와 구별되지 않는다(#257 리뷰)."""
    if xlsx and chunks and chunks[0].meta and chunks[0].meta.get('table_text'):
        texts = []
        seen = set()
        for c in chunks:  # 시트마다 완성본 하나
            st = c.meta.get('table_text')
            if st and st not in seen:
                seen.add(st)
                texts.append(st)
        text = '\n'.join(texts)
    else:
        text = '\n'.join(c.text for c in chunks)
    tables, cur, prev = [], None, None  # cur = 지금 잇는 표, prev = 직전 '|' 행(새 표의 헤더 후보)
    for line in text.split('\n'):
        if not line.startswith('|'):
            prev = None  # 표 아닌 줄 — cur는 유지한다(긴 표가 청크 경계에서 갈린 뒤 이어지도록)
            continue
        cells = [norm(x) for x in line.strip().strip('|').split('|')]
        if cells and all(set(x) <= set('-: ') for x in cells) and any('-' in x for x in cells):
            if prev is not None:  # 구분선: 직전 행이 헤더 → 새 표. 그 행이 cur에 붙어 있었으면 뗀다
                if cur and cur[-1] is prev:
                    cur.pop()
                if cur is not None and cur and cur[0] == prev:
                    pass  # 같은 헤더가 반복되면 같은 표의 다음 행 묶음(#252 — batch_table_rows가 헤더를 반복) → 잇는다
                else:
                    cur = [prev]
                    tables.append(cur)
            prev = None
            continue
        if cur is not None:
            cur.append(cells)
        prev = cells
    return tables


def _rows_texts(rows) -> set[str]:
    return {_loose(norm(t)) for row in rows for t in row if norm(t)}


def score_rows(
    expected: list[list[str]], actual: list[list[str]] | None, presence_pool: set[str] | None = None
) -> Score:
    """L2 — 기대 행 vs 실제 행, 자리 비교. 구조(행·열 수)가 다르면 글자 존재율로만.
    **분할기 흔적까지 포함해 잰다**(#246 리뷰 결정 a): 긴 표가 청크 경계에서 잘리거나 겹침(50토큰)으로 행이 중복되면
    그것도 색인에 들어간 그대로이므로 오류로 집계한다 — 결함 B(#252)의 근거."""
    exp_cells = [Cell(r, c, 1, 1, norm(t)) for r, row in enumerate(expected) for c, t in enumerate(row)]
    dims = (len(expected), max(len(r) for r in expected))
    if actual is None:
        return score_cells(exp_cells, None, dims, presence_pool)
    act = ActualTable(
        [Cell(r, c, 1, 1, norm(t)) for r, row in enumerate(actual) for c, t in enumerate(row)],
        len(actual),
        max(len(r) for r in actual),
    )
    return score_cells(exp_cells, act, dims, presence_pool)


def match_rows(expected_sets: list[list[list[str]]], actuals: list[list[list[str]]]) -> list[list[list[str]] | None]:
    idx = _pair([_rows_texts(e) for e in expected_sets], [_rows_texts(a) for a in actuals])
    return [actuals[i] if i is not None else None for i in idx]


def all_row_texts(actuals: list[list[list[str]]]) -> set[str]:
    return set().union(*[_rows_texts(a) for a in actuals]) if actuals else set()
