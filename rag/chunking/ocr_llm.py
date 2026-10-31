"""상용 OCR(OCR_PROVIDER=llm) — PDF의 사진 표·스캔 쪽을 LLM이 읽는다: 쪽 판정 → 같은 종류 쪽 묶음별 변환 → 사진 표 채우기 → 합치기.

OCR_PROVIDER=llm일 때만 탄다 — triton(기본)이면 이 모듈의 어느 것도 불리지 않는다.
PDF를 쪽마다 판정해 연속된 같은 종류끼리 묶고 묶음마다 따로 변환한 뒤 합친다(DoclingDocument.concatenate —
page_range로 변환해도 쪽 번호는 원래 번호다). 묶음 경계에서 표·헤딩이 끊길 수 있다(감수).
- 디지털 쪽: 기본 경로에서 OCR만 끈 것. 사진 표는 '글자 칸이 하나도 없는 표'로 나오므로 그 사진을 잘라 LLM이 칸을
  채운다(_fill_photo_tables). 화면 캡처·흐름도는 docling이 그림으로 분류해 캡션으로 간다.
- 스캔 쪽: VlmPipeline — 쪽 통째로 LLM이 마크다운을 쓰고 docling이 문서로 만든다.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import pdfplumber

from config import settings
from rag.chunking import document_converter
from rag.chunking.errors import LlmReadFailedError

if TYPE_CHECKING:
    from docling_core.types.doc import DoclingDocument

# 스캔 쪽 판정 — 둘 다 만족해야 스캔이다. 글자 하한이 0이 아닌 것은 스캔본에도 쪽 번호·도장 몇 글자가 글자층으로 박혀서다.
# 사진 조건은 글자 없는 표지·빈 쪽을 거른다. 보이지 않는 글자층을 깐 스캔본은 디지털로 판정돼 그 글자층이 색인된다.
SCAN_MAX_CHARS = 20
SCAN_MIN_IMAGE_AREA = 0.5
_OCR_LLM_SCALE = 2.0  # 쪽·사진을 LLM에 보낼 때의 해상도 배수(72dpi 기준, docling ApiVlmOptions 기본)
# docling 표 영역은 사진의 표 부분뿐이라 사진 안 제목 줄이 빠진다 → 표를 담은 사진 전체를 잘라 보낸다. 단 사진이 표보다
# 이 비율 넘게 크면(쪽 배경·레터헤드) 표 영역만 — 사진 밖 디지털 글이 두 번 색인되지 않게. 한계: CropBox가 MediaBox와 다른
# 쪽은 pdfplumber 좌표(종이 기준)와 docling 좌표(CropBox 기준)가 어긋나 표 영역만 잘린다(제목만 빠진다).
_PHOTO_MAX_AREA_RATIO = 3.0
_PHOTO_BBOX_TOL = 2.0  # pt — 표 영역이 사진 안에 드는지 볼 때의 여유(레이아웃 상자가 사진 경계를 살짝 넘는다)

# 쪽 통째 프롬프트. 절 제목에 '## '를 강제한다 — 평문 번호 제목은 docling 마크다운 해석기가 앞 번호 목록에 합쳐 번호를
# 다시 매긴다(글자가 바뀐다). 문서 제목 '#'는 맨 앞 하나 — 쪽 단위 호출이라 쪽마다 '#'가 생기지 않게.
_SCAN_PAGE_PROMPT = (
    "이 쪽을 마크다운으로 옮겨 적어라. 규칙:\n"
    "1. 문서 제목(문서 맨 앞의 큰 제목)만 '# '로 쓴다. 쪽 중간이나 둘째 쪽 이후의 제목은 문서 제목이 아니다.\n"
    "2. 절·항 제목은 번호가 있든 없든 반드시 '## '로 시작한다. 예: '## 1. 접수 원칙', '## 2.1 사진 증빙'. 번호는 원문 그대로 쓴다.\n"
    "3. 목록은 원문 표시 그대로 — 글머리표는 '- ', 번호 목록은 원문 번호 그대로.\n"
    "4. 표는 마크다운 표(| 칸 | 칸 |)로. 병합된 칸은 그 값을 해당 칸마다 반복해 적고, 빈 칸은 비워 둔다.\n"
    "5. 보이는 글자를 그대로 쓰고 없는 내용을 지어내지 마라. 설명·주석 없이 마크다운만 출력."
)
# 사진 표 프롬프트. 머리글을 지어내지 말라는 규칙은 머리글 없는 표에 '구분 | 내용' 같은 머리글을 만들어 넣는 것을 막는다.
_PHOTO_TABLE_PROMPT = (
    "이 이미지의 표를 마크다운 표(| 칸 | 칸 |)로 옮겨 적어라. 규칙:\n"
    "1. 표 밖(위·아래)에 제목·설명 같은 글 줄이 있으면 표 앞에 그 줄을 그대로 쓴다. 없으면 쓰지 않는다.\n"
    "2. 원본에 머리글 행이 있으면 첫 행에 쓰고, 없으면 머리글을 만들어 넣지 말고 첫 데이터 행을 첫 행에 쓴다.\n"
    "3. 병합된 칸은 그 값을 해당 칸마다 반복해 적고, 빈 칸은 비워 둔다.\n"
    "4. 보이는 글자를 띄어쓰기까지 그대로 쓰고 지어내지 마라. 설명 없이 출력."
)

_converter = None


class PdfPage(NamedTuple):
    width: float
    height: float
    chars: int  # 글자층 글자 수
    images: tuple[
        tuple[float, float, float, float], ...
    ]  # 박힌 사진 (x0, top, x1, bottom) — 왼쪽 위 원점, pt, 쪽 안으로 자름


def scan_profile(file_path: str | Path, max_pages: int | None) -> list[PdfPage] | None:
    """쪽마다 글자 수·사진 위치(pdfplumber). 쪽 수가 max_pages를 넘으면 None — 거절은 docling이 기본 경로와 같은 실패로 한다."""
    pages = []
    with pdfplumber.open(str(file_path)) as pdf:
        if max_pages is not None and len(pdf.pages) > max_pages:
            return None
        for p in pdf.pages:
            w, h = float(p.width), float(p.height)
            boxes = (
                (
                    max(0.0, float(im['x0'])),
                    max(0.0, float(im['top'])),
                    min(w, float(im['x1'])),
                    min(h, float(im['bottom'])),
                )
                for im in p.images
            )
            pages.append(PdfPage(w, h, len(p.chars), tuple(b for b in boxes if b[2] > b[0] and b[3] > b[1])))
    return pages


def is_scan_page(page: PdfPage) -> bool:
    area = page.width * page.height
    biggest = max(((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in page.images), default=0.0)
    return page.chars < SCAN_MAX_CHARS and area > 0 and biggest / area >= SCAN_MIN_IMAGE_AREA


def page_runs(scan_flags: list[bool]) -> list[tuple[bool, int, int]]:
    """쪽별 스캔 여부 → 연속된 같은 종류 묶음 [(스캔인가, 첫 쪽, 끝 쪽)] — 쪽 번호는 1부터, 끝 쪽 포함(docling page_range와 같다)."""
    runs: list[tuple[bool, int, int]] = []
    for no, flag in enumerate(scan_flags, start=1):
        if runs and runs[-1][0] == flag:
            runs[-1] = (flag, runs[-1][1], no)
        else:
            runs.append((flag, no, no))
    return runs


def _vlm_converter():
    """프로세스당 1회: 스캔 쪽용 VlmPipeline 변환기. 연결은 캡션 설정을 공유한다(config.py ocr_provider 주석)."""
    global _converter
    with document_converter.DOCLING_LOCK:
        if _converter is None:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import VlmPipelineOptions
            from docling.datamodel.pipeline_options_vlm_model import ApiVlmOptions, ResponseFormat
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.pipeline.vlm_pipeline import VlmPipeline

            vlm_opts = ApiVlmOptions(
                url=settings.vlm_caption_url,
                headers=document_converter.caption_headers(),
                prompt=_SCAN_PAGE_PROMPT,
                response_format=ResponseFormat.MARKDOWN,
                scale=_OCR_LLM_SCALE,
                timeout=settings.ocr_llm_timeout_seconds,
                concurrency=settings.ocr_llm_concurrency,
                # 온도는 옵션 필드로 — docling ApiVlmModel이 params 위에 vlm_options.temperature를 덮어쓴다
                temperature=settings.vlm_caption_temperature,
                params={'model': settings.vlm_caption_model, 'max_completion_tokens': settings.ocr_llm_max_tokens},
            )
            _converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_cls=VlmPipeline,
                        pipeline_options=VlmPipelineOptions(enable_remote_services=True, vlm_options=vlm_opts),
                    )
                }
            )
    return _converter


def convert_pdf(file_path: str | Path, **kwargs) -> 'DoclingDocument':
    """상용 OCR의 PDF 변환 — 쪽 묶음마다 경로를 골라 변환하고 합친다(모듈 docstring). 실패 분류는 document_converter.convert_with·LlmReadFailedError."""
    pages = scan_profile(file_path, kwargs.get('max_num_pages'))
    if not pages:  # 쪽 상한 초과(None)·빈 PDF — docling이 기본 경로와 같은 방식으로 거절·처리한다
        return document_converter.convert_with(document_converter.docling_converter(), file_path, **kwargs)
    parts = []
    for scan, first, last in page_runs([is_scan_page(p) for p in pages]):
        span = {} if (first, last) == (1, len(pages)) else {'page_range': (first, last)}
        if scan:
            doc = document_converter.convert_with(_vlm_converter(), file_path, **kwargs, **span)
            require_pages(doc, first, last)
            _titles_to_section_headers(doc)
        else:
            doc = document_converter.convert_with(document_converter.docling_converter(), file_path, **kwargs, **span)
            _fill_photo_tables(doc, file_path, pages)
        parts.append(doc)
    if len(parts) == 1:
        return parts[0]
    from docling_core.types.doc import DoclingDocument  # 지연 import — docling은 톱레벨에 안 들인다

    return DoclingDocument.concatenate(parts)


def require_pages(doc, first: int, last: int) -> None:
    """스캔 쪽마다 글 요소가 하나는 있어야 한다 — LLM 빈 응답(docling이 오류를 빈 글자로 삼킨다)은 그 쪽이 통째로 빈다."""
    seen = {prov.page_no for item, _ in doc.iterate_items() for prov in (getattr(item, 'prov', None) or [])}
    missing = [n for n in range(first, last + 1) if n not in seen]
    if missing:
        raise LlmReadFailedError(f'스캔 쪽 {missing} LLM 응답 없음(오류·한도 초과) — 나중에 다시 시도한다')


def _titles_to_section_headers(doc) -> None:
    """title 라벨을 section_header로 바꾼다 — LLM이 '#'로 쓴 줄은 title이 되는데 PDF 섹션 조립
    (docling_elements.sections_from_docling_elements)은 section_header만 헤딩으로 봐서, 그대로면 문서 제목이 본문 줄로 떨어진다."""
    from docling_core.types.doc import DocItemLabel  # 지연 import — docling은 톱레벨에 안 들인다

    for item, _ in doc.iterate_items():
        if item.label == DocItemLabel.TITLE:
            item.label = DocItemLabel.SECTION_HEADER


def _fill_photo_tables(doc, file_path: str | Path, pages: list[PdfPage]) -> None:
    """글자 칸이 하나도 없는 표(= OCR을 끈 경로의 사진 표)를 LLM으로 채운다. 표 밖 줄(제목)은 표 바로 앞 글로 넣는다."""
    from concurrent.futures import ThreadPoolExecutor

    from docling_core.types.doc import DocItemLabel, TableItem  # 지연 import — docling은 톱레벨에 안 들인다

    targets = [
        item
        for item, _ in doc.iterate_items()
        if isinstance(item, TableItem) and item.prov and not any((c.text or '').strip() for c in item.data.table_cells)
    ]
    if not targets:
        return
    crops = _photo_crops(doc, file_path, pages, targets)
    with ThreadPoolExecutor(max_workers=max(1, settings.ocr_llm_concurrency)) as pool:
        replies = list(pool.map(_read_photo_table, crops))
    for item, reply in zip(targets, replies):
        outside, rows = split_table_reply(reply)
        if rows:
            _fill_table_cells(item, rows)
        if outside:
            doc.insert_text(sibling=item, label=DocItemLabel.TEXT, text=outside, prov=item.prov[0], after=False)


def photo_box(
    table: tuple[float, float, float, float], images: tuple[tuple[float, float, float, float], ...]
) -> tuple[float, float, float, float]:
    """잘라 보낼 영역 — 표를 담은 사진 중 가장 작은 것(면적 상한 안), 없으면 표 영역 그대로. 좌표는 왼쪽 위 원점 pt."""
    l, t, r, b = table
    limit = _PHOTO_MAX_AREA_RATIO * (r - l) * (b - t)
    tol = _PHOTO_BBOX_TOL
    holders = [
        box
        for box in images
        if box[0] <= l + tol
        and box[1] <= t + tol
        and box[2] >= r - tol
        and box[3] >= b - tol
        and (box[2] - box[0]) * (box[3] - box[1]) <= limit
    ]
    return min(holders, key=lambda x: (x[2] - x[0]) * (x[3] - x[1])) if holders else table


def _photo_crops(doc, file_path: str | Path, pages: list[PdfPage], targets) -> list:
    """표마다 잘라 보낼 이미지(PIL). pypdfium2로 쪽 하나씩 렌더해 자르고 버린다 — docling generate_page_images는
    쪽 이미지를 문서 내내 들고 있어 큰 문서에서 수백 MB가 된다."""
    import pypdfium2  # 지연 import — docling 의존성이고 상용 OCR에서만 쓴다

    crops, rendered_no, rendered = [], None, None
    pdf = pypdfium2.PdfDocument(str(file_path))
    try:
        for item in targets:
            prov = item.prov[0]
            page = pages[prov.page_no - 1]
            bb = prov.bbox.to_top_left_origin(doc.pages[prov.page_no].size.height)
            box = photo_box((bb.l, bb.t, bb.r, bb.b), page.images)
            if rendered_no != prov.page_no:
                rendered_no, rendered = prov.page_no, pdf[prov.page_no - 1].render(scale=_OCR_LLM_SCALE).to_pil()
            crops.append(rendered.crop(tuple(round(v * _OCR_LLM_SCALE) for v in box)))
    finally:
        pdf.close()
    return crops


def _read_photo_table(image) -> str:
    """사진 표 하나를 LLM에 보낸다(docling api_image_request). 빈 응답이면 일시 실패(LlmReadFailedError).

    출력이 상한에서 잘렸거나(length) 벤더가 걸렀으면 결정적 실패(RuntimeError) — 잘린 표가 행이 빠진 채 색인되지 않게,
    그리고 같은 이미지는 다시 보내도 같다. 스캔 쪽의 같은 두 경우는 document_converter.convert_with가 PARTIAL_SUCCESS로 잡아 같은 분류다.
    """
    from docling.datamodel.base_models import VlmStopReason  # 지연 import — docling은 톱레벨에 안 들인다
    from docling.utils.api_image_request import api_image_request

    reply = api_image_request(
        image=image,
        prompt=_PHOTO_TABLE_PROMPT,
        url=settings.vlm_caption_url,
        timeout=settings.ocr_llm_timeout_seconds,
        headers=document_converter.caption_headers(),
        model=settings.vlm_caption_model,
        temperature=settings.vlm_caption_temperature,
        max_completion_tokens=settings.ocr_llm_max_tokens,
    )
    text = (reply.text or '').strip()
    if not text:
        raise LlmReadFailedError('사진 표 LLM 응답 없음(오류·한도 초과) — 나중에 다시 시도한다')
    if reply.stop_reason == VlmStopReason.LENGTH:
        raise RuntimeError(
            f'사진 표 LLM 출력이 상한({settings.ocr_llm_max_tokens}토큰)에서 잘렸다 — OCR_LLM_MAX_TOKENS를 올려 다시 올린다'
        )
    if reply.stop_reason == VlmStopReason.CONTENT_FILTERED:
        raise RuntimeError('사진 표 LLM 응답이 벤더 콘텐츠 필터에 걸렸다')
    return text


_MD_SEPARATOR_CELL = re.compile(r'^:?-+:?$')


def split_table_reply(text: str) -> tuple[str, list[list[str]]]:
    """LLM 답 → (표 밖 줄을 이은 글, 표 행 목록). '|'로 시작하는 줄이 표다 — 구분선(|---|)은 버리고 칸 수는 가장 긴 행에 맞춰 빈 칸으로
    채운다. 코드 울타리(```)는 버리고, 표 밖 줄의 마크다운 머리 표시('#', '**')는 뗀다. 표 아래 줄(출처 등)도 표 앞 글에 합친다."""
    rows, outside = [], []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('```'):
            continue
        if line.startswith('|'):
            cells = [c.strip() for c in line.strip('|').split('|')]
            if not all(_MD_SEPARATOR_CELL.match(c) for c in cells):
                rows.append(cells)
        else:
            outside.append(re.sub(r'^#+\s*', '', line).strip('*').strip())
    width = max((len(r) for r in rows), default=0)
    return ' '.join(o for o in outside if o), [r + [''] * (width - len(r)) for r in rows]


def _fill_table_cells(item, rows: list[list[str]]) -> None:
    """표 요소의 칸을 행 목록으로 바꾼다 — 첫 행이 머리글(docling_grid_rows가 첫 행을 헤더로 쓴다)."""
    from docling_core.types.doc import TableCell, TableData  # 지연 import — docling은 톱레벨에 안 들인다

    cells = [
        TableCell(
            text=text,
            start_row_offset_idx=i,
            end_row_offset_idx=i + 1,
            start_col_offset_idx=j,
            end_col_offset_idx=j + 1,
            column_header=(i == 0),
        )
        for i, row in enumerate(rows)
        for j, text in enumerate(row)
    ]
    item.data = TableData(num_rows=len(rows), num_cols=len(rows[0]), table_cells=cells)
