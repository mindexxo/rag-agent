"""평문 읽기 — txt 인코딩, PDF 글자층(채팅 첨부용), PDF 이미지 면적 비율."""

from pathlib import Path

import pdfplumber

from rag.chunking.types import Section


def read_text(file_path) -> str:
    """텍스트 파일 읽기 — utf-8 → cp949(국내 txt에 흔함) → utf-8 replace 순으로 시도한다."""
    raw = Path(file_path).read_bytes()
    for enc in ('utf-8', 'cp949'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def pdf_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """PDF → [(쪽 번호, 글자층 텍스트)] (pdfplumber, 표는 공백 구분 평문). 채팅 첨부(extract_text) 전용."""
    with pdfplumber.open(str(file_path)) as pdf:
        return [(i, page.extract_text() or '') for i, page in enumerate(pdf.pages, start=1)]


def pdf_image_area_ratio(file_path: str | Path) -> float:
    """PDF 쪽 면적 대비 삽입 이미지 면적 비율(%). PDF가 아니면 0.

    이미지 속 내용은 캡션·OCR로 읽어도 틀릴 수 있고 텍스트 커버리지로는 드러나지 않는다 — 면적이 유일한 탐지 신호라
    호출부(rag/ingestion.py)가 이 값으로 문서에 경고를 남긴다.
    """
    if not str(file_path).lower().endswith('.pdf'):
        return 0.0
    image_area = page_area = 0.0
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            page_area += (page.width or 0) * (page.height or 0)
            for image in page.images:
                image_area += abs(image['x1'] - image['x0']) * abs(image['bottom'] - image['top'])
    return image_area / page_area * 100 if page_area else 0.0


def txt_sections(file_path: str | Path) -> list[Section]:
    """평문 txt — 파일 전체가 섹션 하나. 빈 파일은 섹션 0개(split_text('')가 ['']를 내 빈 청크가 새지 않게)."""
    text = read_text(file_path)
    return [Section([], None, text)] if text.strip() else []
