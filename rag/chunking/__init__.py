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

from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter

from config import settings
from rag.chunking import convert_process, docling_elements, docx, markdown, plain, sections, split
from rag.chunking.convert_process import abort_conversion, docling_convert
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


def _docling_sections(file_path: str | Path) -> list[Section]:
    """PDF → docling(레이아웃 ML + TableFormer, 그림 캡션, 표 이미지 OCR) → Section.

    인쇄 머리글·바닥글은 docling이 furniture로 분류해 iterate_items()에 나오지 않는다. 헤딩 층 복원은
    docling_elements.sections_from_docling_elements가 한다. 실패는 그대로 올린다 — 저품질 폴백 없음.
    """
    document = convert_process.docling_convert(file_path, max_num_pages=settings.docling_max_num_pages)
    # 각주 사이드카는 변환이 성공한 뒤에만 연다 — 변환 실패 사유가 pdfplumber 오류에 가려지지 않게,
    # 그리고 가짜 변환기 테스트가 파일 없이도 변환 가드를 검증하게.
    footnote_markers = docling_elements.pdf_footnote_markers(file_path)
    return docling_elements.sections_from_docling_elements(
        docling_elements.docling_elements(document, footnote_markers)
    )


def extract_text(file_path: str | Path) -> str:
    """파일 전체를 텍스트 하나로 추출한다(청킹 없음). 채팅 첨부(컨텍스트 직접 주입)용.

    사용자 대기 경로라 docling을 안 태운다 — 표는 평문으로 흐르고, 이미지 속 글자(도표·스캔)는 읽지 못한다.
    """
    low = str(file_path).lower()
    if low.endswith('.pdf'):
        return '\n'.join(t for _, t in plain.pdf_pages(file_path)).strip()
    if low.endswith('.docx'):
        return docx.docx_text(file_path).strip()
    if low.endswith('.xlsx'):
        from rag.xlsx_chunking import (
            xlsx_plain_text,
        )  # 첨부는 openpyxl — docling 변환 프로세스를 색인과 나누지 않는다

        return xlsx_plain_text(str(file_path)).strip()  # 첨부는 행 묶음 없이 시트 전체
    return plain.read_text(file_path).strip()  # txt/md/기타


# 형식 → 섹션 파서. xlsx만 빠져 있다 — 행 묶음 + 시트 완성본·크기 초과 거절이라 계약이 다르다.
_SECTION_PARSERS = {
    '.pdf': _docling_sections,
    '.docx': docx.docx_sections,
    '.md': markdown.md_sections,
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
