"""문서 파일 → 청크(ChunkData) 목록. 형식별 섹션 파서 → sections.pack_sections(묶기) → 크기 분할.

형식 분기는 chunk_file 한 곳뿐이다(호출부는 확장자를 보지 않는다).
- PDF·DOCX·MD: docling 변환(상주 자식 프로세스) → 헤딩 스택으로 섹션 조립(sections.build_sections).
  docling이 실패하면 색인도 실패한다 — 저품질 경로로 조용히 바꾸지 않는다.
- TXT: 평문 하나가 섹션 하나(heading_path 없음).
- XLSX: rag/xlsx_chunking에 위임(행 묶음 + 시트 완성본, 크기 초과면 업로드 거절).
채팅 첨부(extract_text)는 사용자 대기 경로라 docling을 안 태우고 pdfplumber·python-docx·openpyxl로 평문만 뽑는다.

표는 행 단위로 나누고 묶음마다 헤더를 반복한다. 묶음 둘 이상인 표의 청크 meta에는
{"is_table", "table_key", "table_text"(표 앞 설명 + 표 + 아래 주석)}를 싣는다 — 근거 준비(rag/retriever)가 걸린 묶음을
표 전체로 바꾼다. 표 하나가 TABLE_MAX_TOKENS를 넘으면 문서를 색인하지 않는다(TableTooLargeError).
LlamaIndex 의존은 이 모듈 안에 격리한다.
"""

import os
import re
from pathlib import Path
from typing import NamedTuple

from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from llama_index.core.node_parser import SentenceSplitter

from config import settings
from rag.chunking import convert_process, docling_elements, plain, sections, split
from rag.chunking.docling_elements import (
    count_picture_placeholders,
    docling_block_text,
    docling_grid_markdown,
    docling_grid_rows,
    docling_picture_text,
)
from rag.chunking.errors import CaptionFailedError, LlmReadFailedError, PictureLimitError, TableTooLargeError
from rag.chunking.plain import pdf_image_area_ratio
from rag.chunking.split import TABLE_NOTE
from rag.chunking.tables import (
    TABLE_BATCH_CHARS,
    TABLE_BATCH_TOKENS,
    TABLE_MAX_TOKENS,
    batch_table_rows,
    to_markdown_table,
)
from rag.chunking.types import ChunkData, Item, Section, TableBlock, blocks_of, join_blocks
from rag.identifiers import HeadingSplit, is_heading_line, split_heading_line

# 밖(ingestion·routers·xlsx_chunking·eval)이 쓰는 공개 표면 — 하위 모듈로 옮겨도 여기서 이름을 유지한다.
__all__ = [
    'TABLE_BATCH_CHARS',
    'TABLE_BATCH_TOKENS',
    'TABLE_MAX_TOKENS',
    'TABLE_NOTE',
    'CaptionFailedError',
    'ChunkData',
    'Item',
    'LlmReadFailedError',
    'PictureLimitError',
    'Section',
    'TableBlock',
    'TableTooLargeError',
    'abort_conversion',
    'batch_table_rows',
    'blocks_of',
    'chunk_file',
    'count_picture_placeholders',
    'docling_block_text',
    'docling_convert',
    'docling_grid_markdown',
    'docling_grid_rows',
    'docling_picture_text',
    'extract_text',
    'join_blocks',
    'pdf_image_area_ratio',
    'to_markdown_table',
]

_DOCX_HEADING_RE = re.compile(r'^Heading (\d+)$')  # python-docx는 빌트인 헤딩을 영문명으로 준다


def abort_conversion() -> None:
    """진행 중인 docling 변환을 끊는다(변환 프로세스 kill). 색인 잡이 취소될 때 rag/ingestion.py가 부른다."""
    convert_process.process.abort()


def docling_convert(file_path: str | Path, **kwargs):
    """docling 변환 한 번 → DoclingDocument(변환 프로세스에 맡긴다). PDF·DOCX·MD·XLSX가 공유한다.

    실패 분류(outbox.is_transient): TimeoutError·CaptionFailedError·LlmReadFailedError는 일시,
    PictureLimitError·불완전 변환 RuntimeError는 결정적.
    rag/xlsx_chunking이 이 이름을 import한다 — 테스트가 막으려면 이 모듈과 xlsx_chunking 둘 다 패치해야 한다.
    """
    return convert_process.process.convert(file_path, kwargs)


def _docling_sections(file_path: str | Path) -> list[Section]:
    """PDF → docling(레이아웃 ML + TableFormer, 그림 캡션, 표 이미지 OCR) → Section.

    인쇄 머리글·바닥글은 docling이 furniture로 분류해 iterate_items()에 나오지 않는다. 헤딩 층 복원은
    docling_elements.sections_from_docling_elements가 한다. 실패는 그대로 올린다 — 저품질 폴백 없음.
    """
    document = docling_convert(file_path, max_num_pages=settings.docling_max_num_pages)
    # 각주 사이드카는 변환이 성공한 뒤에만 연다 — 변환 실패 사유가 pdfplumber 오류에 가려지지 않게,
    # 그리고 가짜 변환기 테스트가 파일 없이도 변환 가드를 검증하게.
    footnote_markers = docling_elements.pdf_footnote_markers(file_path)
    return docling_elements.sections_from_docling_elements(
        docling_elements.docling_elements(document, footnote_markers)
    )


def _docx_items(file_path: str | Path):
    """DOCX body를 문서 순서대로 (is_heading, level, text, from_table)로 흘린다(python-docx). 채팅 첨부(_docx_text) 전용.

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


def _docx_text(file_path: str | Path) -> str:
    """DOCX 전체를 평문 하나로(채팅 첨부용). 헤딩 포함, 문서 순서 보존."""
    return '\n'.join(text for _, _, text, _ in _docx_items(file_path))


_DOCX_TITLE_MAX = 60


def _split_heading_head(text: str) -> tuple[str, str]:
    """조문·장 문단 → (헤딩 머리, 나머지 본문). 조는 첫 줄의 '제N조(제목)'까지(identifiers.split_heading_line),
    장은 첫 줄 전체.

    문단 전체를 헤딩으로 올리면 같은 문단에 이어진 조문 본문이 heading_path로 들어가고, 뒤에 본문 줄이 없으면
    섹션이 생기지 않아 색인에서 사라진다.
    """
    first, _, rest_lines = text.partition('\n')
    head_split = split_heading_line(first) or HeadingSplit(first.strip(), '')  # 장('제 N 장 …')은 첫 줄 전체가 머리
    return head_split.head, (head_split.rest + ('\n' + rest_lines if rest_lines else '')).strip()


def _docx_fallback_headings(items: list[tuple[bool, int, str, bool]]) -> list[tuple[bool, int, str, bool]]:
    """Heading 스타일이 하나도 없는 docx의 조문 패턴 승격. 입력에 헤딩이 하나라도 있으면 그대로 돌려준다.

    스타일 없는 문서는 전체가 섹션 하나가 되어 분할기가 아무 데서나 자른다. 층은 PDF(docling_elements.sections_from_docling_elements)와 같다:
      제목 = 첫 문단의 첫 줄(조문 패턴이 아니고 _DOCX_TITLE_MAX자 이하) → L1 — 휴리스틱이라 "목 차"도 제목이 된다
      '제 N 장' → L2 / '제 N 조' → L3(장이 아직 없으면 L2). 제목이 없으면 전부 한 단계 위.
    승격하지 않는 것(본문으로 남는다): 'N.'·'N.N' 번호 문단(목록 항목과 못 가른다), 표 행(from_table),
    인용 문단·괄호 없이 적은 조('제8조 목적') — 조 판정은 rag.identifiers.split_heading_line 한 곳이다.
    헤딩은 머리('제N조(제목)')까지만, 같은 문단에 이어진 본문은 본문 줄로 분리한다(_split_heading_head).
    """
    if any(is_heading for is_heading, _, _, _ in items):
        return items
    out: list[tuple[bool, int, str, bool]] = []
    has_title = False
    seen_chapter = False
    for i, (is_heading, level, text, from_table) in enumerate(items):
        if from_table:
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
            head, rest = _split_heading_head(text)
            out.append((True, level, head, False))
            if rest:
                out.append((False, 0, rest, False))
        else:
            out.append((False, 0, text, False))
    if not has_title:
        out = [(h, (lv - 1 if h else lv), t, ft) for h, lv, t, ft in out]
    return out


class _DocxItem(NamedTuple):
    """DOCX 항목 — _docx_items·_docx_docling_items와 같은 꼴이라 조문 폴백을 공유한다(튜플로 풀어 쓰기도 통한다)."""

    is_heading: bool
    level: int  # 헤딩 레벨(본문 줄은 0)
    text: str
    from_table: bool  # 표·그림 — 폴백 헤딩 승격에서 제외하는 표지


def _docx_table_rows(table, doc) -> list[list[str]]:
    """DOCX·MD 표 → 행 목록(첫 행 헤더), 표 안의 표는 평탄화한다.

    표 안 요소는 따로 건너뛰므로(_table_descendants) 안쪽 표는 여기서 바깥 표에 실어야 사라지지 않는다. 모양은 PDF
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
            rows = [row2 for it in inner for row2 in _docx_table_rows(it, doc)]
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
    for r, row in enumerate(base):
        inner_rows = max((len(rows) for (rr, _), rows in nested.items() if rr == r), default=0)
        for i in range(max(inner_rows, 1)):
            line: list[str] = []
            for c, text in enumerate(row):
                if c in width:
                    rows = nested.get((r, c))
                    if rows:
                        cells = rows[i] if i < len(rows) else []
                        line.extend((cells + [''] * width[c])[: width[c]])
                    else:
                        line.extend([text] * width[c])  # 넓어진 열에 걸친 칸 = 가로 병합 — 칸마다 반복
                else:
                    line.append(text)  # 안쪽 행마다 반복 — 세로 병합 반복 규칙(docling_grid_rows)과 같다
            out.append(line)
    return [row for row in out if any(row)]


def _table_descendants(doc) -> set[str]:
    """표 칸 안에 든 요소들의 self_ref. 표 자신은 빼고, 중첩 표는 바깥 표의 후손이라 들어간다(_docx_table_rows가 평탄화해 싣는다)."""
    return {item.self_ref for table in doc.tables for item, _ in doc.iterate_items(root=table) if item is not table}


def _docx_docling_items(doc):
    """docling Word 문서 → _DocxItem 흐름.

    - 헤딩 레벨은 docling section_header.level 그대로(Heading 스타일 레벨과 같다). 'title' 라벨은 본문 줄이다.
    - 표 칸 안의 요소는 건너뛴다 — docling은 서식 있는 칸의 내용을 표와 별도 요소로 한 번 더 내놓는다. 칸 내용·칸 안 그림·
      중첩 표는 표 행(_docx_table_rows)에 이미 들어 있다.
    표는 TableBlock으로 내놓는다. 표와 그림은 from_table=True — 그림 캡션이 '제1장 …'로 시작해도 폴백 헤딩이 되지 않게.
    """
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem  # 지연 import

    inside_tables = _table_descendants(doc)
    for item, _ in doc.iterate_items():
        if item.self_ref in inside_tables:
            continue
        if isinstance(item, SectionHeaderItem):
            if text := docling_elements.docling_norm(item.text):
                yield _DocxItem(True, item.level, text, False)
        elif isinstance(item, TableItem):
            if rows := _docx_table_rows(item, doc):
                yield _DocxItem(False, 0, TableBlock(rows), True)
        elif text := docling_elements.docling_block_text(item, doc):
            yield _DocxItem(False, 0, text, isinstance(item, PictureItem))


def _docx_sections(file_path: str | Path) -> list[Section]:
    """DOCX → docling → 섹션. Heading 스타일이 경계이고, 스타일이 하나도 없으면 조문 패턴 폴백(_docx_fallback_headings).

    헤딩은 본문에 넣지 않는다 — heading_path가 들고 있고 rag/index_text가 색인 입력 앞에 붙인다. 조립은 PDF와 같은
    sections.build_sections(본문 없는 헤딩 보존 포함). page는 없다(None).
    """
    items = list(_docx_docling_items(docling_convert(file_path)))
    return sections.build_sections(
        [(None, text, level if is_heading else None) for is_heading, level, text, _ in _docx_fallback_headings(items)]
    )


def extract_text(file_path: str | Path) -> str:
    """파일 전체를 텍스트 하나로 추출한다(청킹 없음). 채팅 첨부(컨텍스트 직접 주입)용.

    사용자 대기 경로라 docling을 안 태운다 — 표는 평문으로 흐르고, 이미지 속 글자(도표·스캔)는 읽지 못한다.
    """
    low = str(file_path).lower()
    if low.endswith('.pdf'):
        return '\n'.join(t for _, t in plain.pdf_pages(file_path)).strip()
    if low.endswith('.docx'):
        return _docx_text(file_path).strip()
    if low.endswith('.xlsx'):
        from rag.xlsx_chunking import (
            xlsx_plain_text,
        )  # 첨부는 openpyxl — docling 변환 프로세스를 색인과 나누지 않는다

        return xlsx_plain_text(str(file_path)).strip()  # 첨부는 행 묶음 없이 시트 전체
    return plain.read_text(file_path).strip()  # txt/md/기타


# MD 표 칸에서 docling이 망가뜨리는 두 표기 — 변환 전에 유니코드 사용자 영역 문자로 숨겼다가 결과에서 되돌린다.
# '\|'는 칸 구분으로 읽어 넘친 칸을 버리고, '<br>'은 그 행의 값을 표 밖 글로 내보낸다. '<br>' 변형은 '<br>' 하나로 되돌린다.
_MD_PIPE_MARK, _MD_BR_MARK = '\ue000', '\ue001'
_MD_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)


def _md_unmask(text: str) -> str:
    return text.replace(_MD_PIPE_MARK, '\\|').replace(_MD_BR_MARK, '<br>')


def _md_sections(file_path: str | Path) -> list[Section]:
    """마크다운 → docling(MD 백엔드, 같은 변환 프로세스) → 섹션. 표는 TableBlock으로, 헤딩은 문법대로(밑줄식 포함) 나온다.

    docling 출력을 그대로 쓰면 지는 곳 셋을 보정한다:
      ① 표 칸 안 '\\|'·'<br>' — 위 _MD_PIPE_MARK 주석
      ② 링크 — docling은 주소를 hyperlink 속성에만 둔다. '글(주소)'로 붙인다
      ③ 인라인 서식 — docling은 한 문장을 서식 단위 조각(같은 InlineGroup)으로 쪼갠다. 한 줄로 다시 잇고, 조각 사이
         띄어쓰기는 원문을 보고 정한다('**접수**로만' → '접수로만')
    레벨: '#'은 TitleItem(1단), '##'부터는 SectionHeaderItem(level 1부터) → +1. DOCX와 달리 MD의 '#'은 헤딩이다.
    빈 헤딩(YAML 앞머리)은 버리고, 코드 블록은 들여쓰기를 그대로 둔다. 글자 소실은 tests/test_chunking_md.py가 지킨다.

    알려진 한계(모양만 다르고 내용은 남는다):
      - ③ 원문에서 조각을 못 찾으면 띄운다 — 조사 앞에 빈칸이 생길 수 있다('확인 은')
      - 서식이 섞인 목록 항목은 번호가 따로 한 줄이 된다('1.' / '양식(…) 받기')
      - 그림('![대체 글](주소)')은 파일 안에 그림이 없어 캡션하지 않는다 — 대체 글과 자리표시가 남는다
      - 큰 표는 다른 형식과 같이 TABLE_MAX_TOKENS를 넘으면 색인하지 않는다
    """
    import tempfile

    from docling_core.types.doc import (
        CodeItem,
        InlineGroup,
        SectionHeaderItem,
        TableItem,
        TextItem,  # 지연 import
        TitleItem,
    )

    source = _MD_BR_RE.sub(_MD_BR_MARK, plain.read_text(file_path).replace('\\|', _MD_PIPE_MARK))
    with tempfile.NamedTemporaryFile('w', suffix='.md', encoding='utf-8', delete=False) as tmp:
        tmp.write(source)
    try:
        doc = docling_convert(tmp.name)
    finally:
        os.unlink(tmp.name)

    inside = _table_descendants(doc)
    items: list[Item] = []
    cursor = 0  # 원문에서 직전 글 조각이 끝난 자리 — ③ 띄어쓰기 판정
    group = None  # 직전 조각이 속한 InlineGroup
    for item, _ in doc.iterate_items():
        if item.self_ref in inside:
            continue
        parent = item.parent.resolve(doc) if item.parent else None
        inline = parent.self_ref if isinstance(parent, InlineGroup) else None
        raw = getattr(item, 'text', '') or ''
        found = source.find(raw, cursor) if raw else -1
        spaced = found < 0 or any(ch.isspace() for ch in source[cursor:found])
        if found >= 0:
            cursor = found + len(raw)
        if isinstance(item, (TitleItem, SectionHeaderItem)):
            if text := _md_unmask(docling_elements.docling_norm(item.text)):
                items.append((None, text, 1 if isinstance(item, TitleItem) else item.level + 1))
        elif isinstance(item, TableItem):
            if rows := _docx_table_rows(item, doc):
                items.append((None, TableBlock([[_md_unmask(c) for c in row] for row in rows]), None))
        elif isinstance(item, CodeItem):
            # 코드 블록은 들여쓰기가 구조다 — 공백을 접는 docling_elements.docling_norm을 태우지 않는다
            if text := _md_unmask('\n'.join(line.rstrip() for line in item.text.strip('\n').split('\n'))):
                items.append((None, text, None))
        else:
            text = docling_elements.docling_block_text(item, doc)
            if isinstance(item, TextItem) and item.hyperlink:
                text = f'{text}({item.hyperlink})'
            if text := _md_unmask(text):
                if inline and inline == group and items and items[-1][2] is None and isinstance(items[-1][1], str):
                    items[-1] = (None, items[-1][1] + (' ' if spaced else '') + text, None)
                else:
                    items.append((None, text, None))
        group = inline
    return sections.build_sections(items)


# 형식 → 섹션 파서. xlsx만 빠져 있다 — 행 묶음 + 시트 완성본·크기 초과 거절이라 계약이 다르다.
_SECTION_PARSERS = {
    '.pdf': _docling_sections,
    '.docx': _docx_sections,
    '.md': _md_sections,
    '.txt': plain.txt_sections,
}


def chunk_file(file_path: str | Path, *, description: str = '') -> list[ChunkData]:
    """파일 한 개를 청크 리스트로. 형식 분기는 여기 한 곳뿐이다 — 호출부는 확장자를 보지 않는다.

    xlsx는 chunk_xlsx로 위임한다(description은 xlsx 전용). 나머지는 섹션 추출 → sections.pack_sections → split.split_section.
    """
    suffix = Path(file_path).suffix.lower()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    if suffix == '.xlsx':
        from rag.xlsx_chunking import chunk_xlsx  # 지연 import (xlsx_chunking → chunking 순환 방지)

        return chunk_xlsx(str(file_path), description=description)

    parser = _SECTION_PARSERS.get(suffix)
    if parser is None:
        # 업로드는 SUPPORTED_SUFFIXES가 앞에서 거르므로 도달하지 않는다. 도달했다면
        # 지원 안 되는 파일이 파서를 잘못 타고 있다는 뜻이라 조용히 넘기지 않는다.
        raise ValueError(f'지원하지 않는 형식: {suffix or "확장자 없음"}')

    out: list[ChunkData] = []
    table_keys: dict[tuple[int, int], str] = {}  # (섹션 번호, 섹션 안 블록 번호) → 문서 안 표 순번
    for si, section in enumerate(sections.pack_sections(parser(file_path))):
        for text, offset, table in split.split_section(section, splitter):
            # 이 조각에 실린 헤딩만 — 색인이 식별자 사전으로, 근거 준비가 구간 경계로 쓴다.
            lines = {line.strip() for line in text.split('\n')}
            heads = [h for h in section.headings if h.strip() in lines]
            meta = {'headings': heads} if heads else {}
            if table is not None:
                key = table_keys.setdefault((si, table.block), f't{len(table_keys) + 1}')
                meta |= {'is_table': True, 'table_key': key, 'table_text': table.text}
            out.append(
                ChunkData(
                    text=text,
                    heading_path=list(section.heading_path),
                    page=sections.page_at(section, offset),
                    chunk_index=len(out),
                    meta=meta or None,
                )
            )
    return out
