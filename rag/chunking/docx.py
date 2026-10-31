"""DOCX 파서 — docling 문서(헤딩·표·그림) → Section, 스타일 없는 문서의 조문 헤딩 폴백; 채팅 첨부용 python-docx 평문 추출."""

import re
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

from rag.chunking import convert_process, docling_elements, sections
from rag.chunking.types import Section, TableBlock
from rag.identifiers import HeadingSplit, is_heading_line, split_heading_line

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument, TableItem

_DOCX_HEADING_RE = re.compile(r'^Heading (\d+)$')  # python-docx는 빌트인 헤딩을 영문명으로 준다


def docx_items(file_path: str | Path) -> Iterator[tuple[bool, int, str, bool]]:
    """DOCX body를 문서 순서대로 (is_heading, level, text, from_table)로 흘린다(python-docx). 채팅 첨부(docx_text) 전용.

    doc.paragraphs와 doc.tables를 따로 훑으면 표가 문서 끝으로 밀리므로 body 자식을 직접 순회한다.
    """
    doc = DocxDocument(str(file_path))
    for child in doc.element.body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            para = DocxParagraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            # 스타일 정의가 빠진 docx는 style이 None이다
            m = _DOCX_HEADING_RE.match(getattr(para.style, 'name', None) or '')
            yield (bool(m), int(m.group(1)) if m else 0, text, False)
        elif tag == 'tbl':
            for row in DocxTable(child, doc).rows:
                # 빈 셀도 자리를 지킨다 — 걸러내면 뒷 열이 앞으로 당겨져 헤더와 어긋난다. 완전 빈 행만 건너뛴다.
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    yield (False, 0, ' | '.join(cells), True)  # from_table — 폴백 헤딩 승격 대상에서 제외


def docx_text(file_path: str | Path) -> str:
    """DOCX 전체를 평문 하나로(채팅 첨부용). 헤딩 포함, 문서 순서 보존."""
    return '\n'.join(text for _, _, text, _ in docx_items(file_path))


_DOCX_TITLE_MAX = 60


def split_heading_head(text: str) -> tuple[str, str]:
    """조문·장 문단 → (헤딩 머리, 나머지 본문). 조는 첫 줄의 '제N조(제목)'까지(identifiers.split_heading_line),
    장은 첫 줄 전체.

    문단 전체를 헤딩으로 올리면 같은 문단에 이어진 조문 본문이 heading_path로 들어가고, 뒤에 본문 줄이 없으면
    섹션이 생기지 않아 색인에서 사라진다.
    """
    first, _, rest_lines = text.partition('\n')
    head_split = split_heading_line(first) or HeadingSplit(first.strip(), '')  # 장('제 N 장 …')은 첫 줄 전체가 머리
    return head_split.head, (head_split.rest + ('\n' + rest_lines if rest_lines else '')).strip()


def docx_fallback_headings(
    items: Sequence[tuple[bool, int, 'str | TableBlock', bool]],
) -> Sequence[tuple[bool, int, 'str | TableBlock', bool]]:
    """Heading 스타일이 하나도 없는 docx의 조문 패턴 승격. 입력에 헤딩이 하나라도 있으면 그대로 돌려준다.

    스타일 없는 문서는 전체가 섹션 하나가 되어 분할기가 아무 데서나 자른다. 층은 PDF(docling_elements.sections_from_docling_elements)와 같다:
      제목 = 첫 문단의 첫 줄(조문 패턴이 아니고 _DOCX_TITLE_MAX자 이하) → L1 — 휴리스틱이라 "목 차"도 제목이 된다
      '제 N 장' → L2 / '제 N 조' → L3(장이 아직 없으면 L2). 제목이 없으면 전부 한 단계 위.
    승격하지 않는 것(본문으로 남는다): 'N.'·'N.N' 번호 문단(목록 항목과 못 가른다), 표 행(from_table),
    인용 문단·괄호 없이 적은 조('제8조 목적') — 조 판정은 rag.identifiers.split_heading_line 한 곳이다.
    헤딩은 머리('제N조(제목)')까지만, 같은 문단에 이어진 본문은 본문 줄로 분리한다(split_heading_head).
    """
    if any(is_heading for is_heading, _, _, _ in items):
        return items
    out: list[tuple[bool, int, 'str | TableBlock', bool]] = []
    has_title = False
    seen_chapter = False
    for i, (is_heading, level, text, from_table) in enumerate(items):
        if from_table or not isinstance(text, str):  # 표·그림은 승격 대상이 아니다(표 블록은 언제나 from_table)
            out.append((is_heading, level, text, from_table))
            continue
        first_line = text.split('\n', 1)[0].strip()
        if (
            i == 0
            and len(first_line) <= _DOCX_TITLE_MAX
            and not sections.CHAPTER_RE.match(first_line)
            and not is_heading_line(first_line)
        ):
            has_title = True
            out.append((True, 1, first_line, False))
            rest = text[len(text.split('\n', 1)[0]) :].strip()
            if rest:
                out.append((False, 0, rest, False))
            continue
        # 인용 문단("제98조 및 제99조에 따라")을 제목으로 올리면 뒤에 곧바로 다른 제목이 올 때 그 문장이 색인에서 사라진다.
        if sections.CHAPTER_RE.match(text) or is_heading_line(first_line):
            if sections.CHAPTER_RE.match(text):
                seen_chapter = True
                level = 2
            else:
                level = 3 if seen_chapter else 2
            head, rest = split_heading_head(text)
            out.append((True, level, head, False))
            if rest:
                out.append((False, 0, rest, False))
        else:
            out.append((False, 0, text, False))
    if not has_title:
        out = [(h, (lv - 1 if h else lv), t, ft) for h, lv, t, ft in out]
    return out


class DocxItem(NamedTuple):
    """DOCX 항목 — docx_items·docx_docling_items와 같은 꼴이라 조문 폴백을 공유한다(튜플로 풀어 쓰기도 통한다)."""

    is_heading: bool
    level: int  # 헤딩 레벨(본문 줄은 0)
    text: 'str | TableBlock'
    from_table: bool  # 표·그림 — 폴백 헤딩 승격에서 제외하는 표지


def docx_table_rows(table: 'TableItem', doc: 'DoclingDocument') -> list[list[str]]:
    """DOCX·MD 표 → 행 목록(첫 행 헤더), 표 안의 표는 평탄화한다.

    표 안 요소는 따로 건너뛰므로(table_descendants) 안쪽 표는 여기서 바깥 표에 실어야 사라지지 않는다. 모양은 PDF
    TableFormer와 같다: 안쪽 표가 든 열은 안쪽 열 수만큼 넓어지고 그 행은 안쪽 행 수만큼 늘며, 같은 행의 다른 칸은 세로로,
    다른 행의 그 열 칸은 넓어진 열마다 반복한다(병합 반복 규칙과 같다). 안쪽 표가 둘 이상인 칸은 위아래로 이어 붙인다.
    """
    from docling_core.types.doc import RichTableCell, TableItem  # 지연 import

    grid = table.data.grid
    nested: dict[tuple[int, int], list[list[str]]] = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if not (
                isinstance(cell, RichTableCell) and cell.start_row_offset_idx == r and cell.start_col_offset_idx == c
            ):
                continue
            inner = [it for it, _ in doc.iterate_items(root=cell.ref.resolve(doc)) if isinstance(it, TableItem)]
            deeper = {d.self_ref for it in inner for d, _ in doc.iterate_items(root=it) if d is not it}
            inner = [it for it in inner if it.self_ref not in deeper]  # 3단 중첩은 재귀가 맡는다 — 중복 금지
            rows = [row2 for it in inner for row2 in docx_table_rows(it, doc)]
            if rows:
                # 칸에 안쪽 표와 함께 적힌 글자(라벨)는 첫 행으로 남긴다
                own = docling_elements.cell_text_norm(cell.text)
                nested[(r, c)] = ([[own] + [''] * (len(rows[0]) - 1)] if own else []) + rows
    base = [[docling_elements.grid_cell_text(cell, doc) for cell in row] for row in grid]
    if not nested:
        return [row for row in base if any(row)]
    width = {
        c: max(len(x) for (_, cc), rows in nested.items() if cc == c for x in rows)  # 열 c → 안쪽 열 수(최대)
        for c in {c for (_, c) in nested}
    }
    out: list[list[str]] = []
    for r, base_row in enumerate(base):
        inner_rows = max((len(rows) for (rr, _), rows in nested.items() if rr == r), default=0)
        for i in range(max(inner_rows, 1)):
            line: list[str] = []
            for c, text in enumerate(base_row):
                if c in width:
                    inner_rows_here = nested.get((r, c))
                    if inner_rows_here:
                        inner_cells = inner_rows_here[i] if i < len(inner_rows_here) else []
                        line.extend((inner_cells + [''] * width[c])[: width[c]])
                    else:
                        line.extend([text] * width[c])  # 넓어진 열에 걸친 칸 = 가로 병합 — 칸마다 반복
                else:
                    line.append(text)  # 안쪽 행마다 반복 — 세로 병합 반복 규칙(docling_grid_rows)과 같다
            out.append(line)
    return [row for row in out if any(row)]


def docx_docling_items(doc: 'DoclingDocument') -> Iterator[DocxItem]:
    """docling Word 문서 → DocxItem 흐름.

    - 헤딩 레벨은 docling section_header.level 그대로(Heading 스타일 레벨과 같다). 'title' 라벨은 본문 줄이다.
    - 표 칸 안의 요소는 건너뛴다 — docling은 서식 있는 칸의 내용을 표와 별도 요소로 한 번 더 내놓는다. 칸 내용·칸 안 그림·
      중첩 표는 표 행(docx_table_rows)에 이미 들어 있다.
    표는 TableBlock으로 내놓는다. 표와 그림은 from_table=True — 그림 캡션이 '제1장 …'로 시작해도 폴백 헤딩이 되지 않게.
    """
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem  # 지연 import

    inside_tables = docling_elements.table_descendants(doc)
    for item, _ in doc.iterate_items():
        if item.self_ref in inside_tables:
            continue
        if isinstance(item, SectionHeaderItem):
            if text := docling_elements.docling_norm(item.text):
                yield DocxItem(True, item.level, text, False)
        elif isinstance(item, TableItem):
            if rows := docx_table_rows(item, doc):
                yield DocxItem(False, 0, TableBlock(rows), True)
        elif text := docling_elements.docling_block_text(item, doc):
            yield DocxItem(False, 0, text, isinstance(item, PictureItem))


def docx_sections(file_path: str | Path) -> list[Section]:
    """DOCX → docling → 섹션. Heading 스타일이 경계이고, 스타일이 하나도 없으면 조문 패턴 폴백(docx_fallback_headings).

    헤딩은 본문에 넣지 않는다 — heading_path가 들고 있고 rag/index_text가 색인 입력 앞에 붙인다. 조립은 PDF와 같은
    sections.build_sections(본문 없는 헤딩 보존 포함). page는 없다(None).
    """
    items = list(docx_docling_items(convert_process.docling_convert(file_path)))
    return sections.build_sections(
        [(None, text, level if is_heading else None) for is_heading, level, text, _ in docx_fallback_headings(items)]
    )
