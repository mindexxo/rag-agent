"""DoclingDocument → 요소 흐름 → Section — 격자·그림·목록 요소의 텍스트 정제, PDF 위첨자 각주 제거, PDF 헤딩 층 복원."""

import logging
import re
from collections.abc import Iterator
from pathlib import Path

import pdfplumber

from rag.chunking import sections
from rag.chunking.tables import to_markdown_table
from rag.chunking.types import Item, Section, TableBlock
from rag.identifiers import split_heading_line

logger = logging.getLogger(__name__)

# 캡션이 없는 그림의 자리표시 — 어떤 그림이 남는지는 count_picture_placeholders.
PICTURE_PLACEHOLDER = '<!-- image -->'


# 위첨자 각주 참조 번호. docling은 글자 크기를 버리고 'PM 2)'처럼 평문에 섞어 생성기를 헷갈리게 한다. 열거("1) …")와
# 글자 모양이 같아 정규식으로는 못 가르고, 유일한 물리적 차이인 글자 크기 비율을 pdfplumber 단어 크기로 판정한다.
_FOOTNOTE_TOKEN_RE = re.compile(r'^\d{1,2}\)$')
_FOOTNOTE_SUP_RATIO = 0.8  # 같은 줄 최대 글자 크기 대비 — 이보다 작은 'N)'만 각주 표시
_FOOTNOTE_LINE_TOL = 4.0  # 같은 줄로 묶는 top 차이(pt). 위첨자는 본문보다 조금 올라간다


def markers_from_words(words: list[tuple[str, float, float]]) -> set[str]:
    """(단어, 글자크기, top) 목록 → 위첨자 각주 표시 집합. 한 쪽 분량을 받는다.

    top이 _FOOTNOTE_LINE_TOL 안이면 같은 줄로 묶고, 줄 최대 크기의 _FOOTNOTE_SUP_RATIO 미만인 'N)' 토큰만
    각주 표시로 본다. 각주 정의 줄("2) PM 없는 경우 생략")은 줄 전체가 작은 글자라 걸리지 않는다(본문에 남긴다).
    """
    found: set[str] = set()
    line: list[tuple[str, float]] = []
    line_top: float | None = None

    def flush():
        if not line:
            return
        body = max(size for _, size in line)
        for text, size in line:
            if _FOOTNOTE_TOKEN_RE.match(text) and size < body * _FOOTNOTE_SUP_RATIO:
                found.add(text)

    for text, size, top in sorted(words, key=lambda w: (w[2], w[0])):
        if line_top is None or abs(top - line_top) > _FOOTNOTE_LINE_TOL:
            flush()
            line, line_top = [], top
        line.append((text, size))
    flush()
    return found


def pdf_footnote_markers(file_path: str | Path) -> dict[int, set[str]]:
    """쪽 → 위첨자 각주 표시 집합. pdfplumber로 docling 변환과 별개로 한 번 더 연다.

    best-effort: 실패하면 경고만 남기고 빈 dict — 표시가 본문에 남을 뿐이라, 보조 파서 오류로 문서를 failed 처리하는 쪽이 더 나쁘다.
    """
    out: dict[int, set[str]] = {}
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                words = [
                    (w['text'], float(w.get('size') or 0.0), float(w['top']))
                    for w in page.extract_words(extra_attrs=['size'])
                ]
                found = markers_from_words(words)
                if found:
                    out[page_no] = found
    except Exception as e:  # noqa: BLE001 — 보조 경로. 사유는 남기고 본 경로는 계속 간다
        logger.warning('각주 사이드카 실패(각주 표시 제거 없이 진행) %s: %s', file_path, e)
    return out


def strip_footnote_markers(text: str, markers: set[str]) -> str:
    """본문에서 각주 참조 'N)'를 뗀다 — 단어·닫는 괄호 **뒤에 붙은 것만**. 줄 머리의 정의 줄("2) PM …")과
    같은 번호가 아닌 열거, 더 큰 숫자의 일부("12)")는 그대로다. 같은 쪽에 같은 번호의 열거가 단어 뒤에 붙어 있으면 그것도 지워진다(감수)."""
    for marker in sorted(markers, key=len, reverse=True):
        # 앞의 공백은 같은 줄 안에서만([ \t]) — 줄바꿈을 허용하면 다음 줄 머리의 정의 줄 "2) PM …"까지 지운다.
        # (?<!\d): "서식 12)"의 "2)"는 각주가 아니라 숫자의 일부다.
        pattern = r'(?<=[\w\)\]）」])[ \t]?(?<!\d)' + re.escape(marker) + r'(?=$|[\s,.;:)>\]）」→])'
        text = re.sub(pattern, '', text)
    return text


def docling_norm(text: str) -> str:
    """docling 텍스트의 NBSP(\\xa0)를 보통 공백으로. 표 셀·헤딩에 섞여 들어와 검색 토큰을 깨뜨린다."""
    return re.sub(r'[  ]+', ' ', text).strip()


def docling_grid_markdown(table, doc=None) -> str | None:
    """docling 표 → markdown, 셀 원문(grid)에서 직접 그린다. docling 표 렌더의 정의점(첫 행 = 헤더, 열이 없으면 None).

    - 병합 칸은 범위의 모든 칸에 값을 둔다(가로·세로 모두) — 첫 칸에만 두면 나머지 칸이 빈칸으로 읽히거나, 청크 경계에서
      잘린 아래 행이 분류어를 잃는다. 대가는 긴 병합 셀의 중복이다.
    - export_to_dataframe을 쓰지 않는다 — column_header 표시가 없는 표에 열 이름 0,1,2…를 붙이고, Word 서식 칸을
      '**항목**'처럼 마크다운 굵게로 내보낸다(셀 원문에는 '**'가 없다).
    - 값이 하나도 없는 행은 빼고, 빈 셀은 자리를 지킨다(to_markdown_table).
    - 칸 안의 그림은 그 칸 글자 뒤에 캡션(없으면 자리표시)을 한 줄로 붙인다(doc가 있을 때) — 캡처 설명이 제 행과 붙어 있게.
      여러 줄 캡션의 줄바꿈은 사라진다(마크다운 표는 행 = 한 줄).
    """
    rows = docling_grid_rows(table, doc)
    return to_markdown_table(rows[0], rows[1:]) if rows else None


def docling_grid_rows(table, doc=None) -> list[list[str]]:
    """docling 표 → 행 목록(첫 행 = 헤더). docling_grid_markdown의 행 단계(규칙은 그 docstring) — 행 묶음은 글로
    바꾸기 전에 나눠야 해서 따로 둔다(batch_table_rows)."""
    rows = [[grid_cell_text(cell, doc) for cell in row] for row in table.data.grid]
    return [row for row in rows if any(row)]


_XLSX_MIDNIGHT_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}) 00:00:00$')


def cell_text_norm(text: str) -> str:
    """칸 글자 정리: 줄바꿈은 공백으로(markdown 표는 행 = 한 줄), 엑셀 날짜 칸의 ` 00:00:00` 꼬리는 뗀다
    (docling이 datetime을 str로 바꾸며 붙인다). 자정이 아닌 실제 시각은 그대로 둔다."""
    text = docling_norm(text.replace('\n', ' '))
    return _XLSX_MIDNIGHT_RE.sub(r'\1', text)


def grid_cell_text(cell, doc) -> str:
    """표 칸 하나의 글자 — 칸 원문(cell_text_norm) + 칸 안 그림의 설명(docling_grid_markdown 참조)."""
    from docling_core.types.doc import PictureItem, RichTableCell  # 지연 import — docling은 톱레벨에 안 들인다

    parts = [cell_text_norm(cell.text)]
    if doc is not None and isinstance(cell, RichTableCell):
        parts += [
            re.sub(r'\s+', ' ', docling_picture_text(item))
            for item, _ in doc.iterate_items(root=cell.ref.resolve(doc))
            if isinstance(item, PictureItem)
        ]
    return ' '.join(p for p in parts if p)


def docling_picture_text(element) -> str:
    """그림 요소의 본문 한 줄 — 캡션, 없으면 자리표시(캡션 대상이 아닌 그림)."""
    return docling_norm(docling_picture_caption(element)) or PICTURE_PLACEHOLDER


def docling_block_text(item, doc) -> str:
    """docling 요소 하나 → 본문 텍스트(없으면 ''). DOCX·MD·XLSX가 공유하는 요소 종류 분기의 정의점.

    목록 항목은 docling이 text에서 뗀 번호(marker)를 되붙인다. 헤딩 판정은 형식마다 달라 호출부가 한다.
    """
    from docling_core.types.doc import ListItem, PictureItem, TableItem, TextItem  # 지연 import

    match item:
        case TableItem():
            return docling_grid_markdown(item, doc) or ''
        case PictureItem():
            return docling_picture_text(item)
        case ListItem(marker=marker) if marker:
            return docling_norm(f'{marker} {item.text}')
        case TextItem():
            return docling_norm(item.text)
    return ''


def count_picture_placeholders(chunks) -> int:
    """청크들에 남은 그림 자리표시 수 — 캡션 대상에서 빠진 그림(면적 하한 미만·MD 그림·엑셀 차트 객체)의 수.

    캡션이 붙은 그림은 본문과 구분되지 않아 셀 수 없다. 지표 기록은 호출부(rag/ingestion.py)가 한다 — 이 모듈은
    0층이라 rag/metrics.py를 import하지 않는다.
    """
    return sum(c.text.count(PICTURE_PLACEHOLDER) for c in chunks)


def docling_picture_caption(element) -> str:
    """그림 요소의 VLM 캡션(없으면 ''). docling은 캡션을 element.text가 아니라 annotations에 담는다 — 첫 비어 있지 않은 것."""
    for ann in getattr(element, 'annotations', None) or []:
        text = (getattr(ann, 'text', '') or '').strip()
        if text:
            return text
    return ''


def docling_elements(
    doc, footnote_markers: dict[int, set[str]] | None = None
) -> Iterator[tuple[str, str, int | None, list[list[str]] | None]]:
    """docling PDF 문서 → (label, text, page, table_rows) 흐름(읽기 순서). table_rows는 표일 때만 행 목록(docling_grid_rows).

    sections_from_docling_elements와 나눈 것은 매핑 규칙을 docling 없이 튜플만으로 단위 테스트하기 위해서다.
    footnote_markers(쪽 → 각주 표시)는 텍스트 요소에서만 뗀다 — 표·그림은 대상이 아니다.
    """
    footnote_markers = footnote_markers or {}
    for element, _depth in doc.iterate_items():
        label = element.label.value if hasattr(element.label, 'value') else str(element.label)
        page = element.prov[0].page_no if getattr(element, 'prov', None) else None
        if label == 'table':
            yield label, '', page, docling_grid_rows(element, doc) or None
        elif label == 'picture':
            # 캡션이 없는 그림은 빈 문자열이 나가고, 소비자가 자리표시로 바꾼다.
            yield label, docling_picture_caption(element), page, None
        else:
            text = getattr(element, 'text', '') or ''
            # docling은 목록 번호·글머리('2.1', '-', '(1)')를 text에서 떼어 marker에 둔다 — export_to_markdown처럼 다시 붙인다.
            marker = getattr(element, 'marker', '') or ''
            if label == 'list_item' and marker:
                text = f'{marker} {text}'
            if page in footnote_markers:
                text = strip_footnote_markers(text, footnote_markers[page])
            yield label, text, page, None


def _join_split_article_titles(elements):
    """닫는 괄호만 다음 요소로 떨어진 조 제목을 잇는다 — '제 7 조 ( 징계종류' + ')' → '제 7 조 ( 징계종류 )'."""
    elements = list(elements)
    i = 0
    while i < len(elements):
        label, text, page, table = elements[i]
        nxt = elements[i + 1] if i + 1 < len(elements) else None
        t = docling_norm(text or '')
        if (
            label in ('section_header', 'text')
            and nxt is not None
            and docling_norm(nxt[1] or '') in (')', '）')
            and (head := split_heading_line(t))
            and head.head == t
            and not any(c in t for c in ')）')
        ):
            yield label, f'{t} {docling_norm(nxt[1])}', page, table
            i += 2
            continue
        yield label, text, page, table
        i += 1


def sections_from_docling_elements(elements) -> list[Section]:
    """(label, text, page, table) 흐름 → Section. PDF 헤딩 층 복원 규칙의 정의점. table은 행 목록(TableBlock으로
    운반)이거나 markdown 문자열(그대로 글 블록 — 튜플만 먹이는 테스트용).

    docling은 장·조를 구분하지 않으므로 여기서 층을 만든다:
      첫 헤딩 = 문서 제목(L1) / '제 N 장' = L2 / 그 외 헤딩 = L3 (장이 아직 없으면 L2).
    그래야 sections.pack_sections가 같은 장 아래 조항끼리 묶는다. 그림은 캡션(없으면 자리표시), 나머지 텍스트 요소는 본문 줄.

    조 제목 보정 — docling 레이아웃 모델은 본문과 같은 글꼴의 조 제목을 들쭉날쭉 section_header/text로 판정한다. 두 꼴만 고친다:
      ① 문서 제목 뒤의 text 요소가 조 제목 꼴(rag/identifiers.split_heading_line)로 시작하면 제목으로 올리고, 같은 요소에
         이어진 글은 그 조의 본문으로 둔다. 한계: 조 제목 꼴로 시작하는 본문 문단(쪽 경계에서 끊긴 문장·목차 줄)도 올라간다(글은 남는다).
      ② 닫는 괄호가 다음 요소로 떨어진 조 제목은 하나로 잇는다(_join_split_article_titles) — 안 이으면 ')'가 경로에 들어간다.
    docling이 문서 제목까지 본문으로 낸 문서에서는 보정이 걸리지 않는다. 조문 체계 밖의 제목('1. 목적')은 글자로 가를 단서가 없어 보정하지 않는다.
    """
    items: list[Item] = []
    seen_title = False
    seen_chapter = False
    for label, text, page, table_markdown in _join_split_article_titles(elements):
        if label == 'text' and seen_title and (head := split_heading_line(docling_norm(text))):
            # ① 조 제목 꼴 본문 요소 → 제목(+ 이어진 글은 본문)
            items.append((page, head.head, 3 if seen_chapter else 2))
            if head.rest:
                items.append((page, head.rest, None))
            continue
        if label == 'section_header':
            text = docling_norm(text)
            if not text:
                continue
            if not seen_title:
                level, seen_title = 1, True
            elif sections.CHAPTER_RE.match(text):
                level, seen_chapter = 2, True
            else:
                level = 3 if seen_chapter else 2
            items.append((page, text, level))
        elif label == 'table':
            if not table_markdown:
                continue
            if isinstance(table_markdown, str):  # markdown 문자열 — 튜플만 먹이는 단위 테스트 호환
                items.append((page, table_markdown, None))
                continue
            block = TableBlock(table_markdown, [page] * len(table_markdown))
            # 쪽을 넘긴 표 잇기 — docling은 쪽마다 표를 따로 낸다. 바로 앞 항목이 표이고 헤더가 같을 때만 뒤 표의 헤더를
            # 떼고 행을 잇는다. 행별 쪽 번호는 각자 것을 지킨다(row_pages).
            if items and isinstance(items[-1][1], TableBlock) and items[-1][1].rows[0] == block.rows[0]:
                prev_page, prev_block, _ = items[-1]
                items[-1] = (
                    prev_page,
                    TableBlock(prev_block.rows + block.rows[1:], prev_block.row_pages + block.row_pages[1:]),
                    None,
                )
                continue
            items.append((page, block, None))
        elif label == 'picture':
            caption = docling_norm(text)
            items.append((page, caption or PICTURE_PLACEHOLDER, None))
        else:
            text = docling_norm(text)
            if text:
                items.append((page, text, None))
    return sections.build_sections(items)
