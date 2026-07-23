"""문서 파싱 + 청킹 모듈 (비ML 추출).

 형식별 파서 → SentenceSplitter로 크기 분할 → ChunkData 리스트.
 ingestion(워커)이 이 결과를 받아 임베딩 + DB 적재.

 ================================================================
   형식별 파싱
 ================================================================
 - PDF : pdfplumber 페이지별 텍스트 추출. page 번호 보존(인용용), heading_path 없음(레이아웃 미인식).
 - DOCX: python-docx 문단 + 표 셀 텍스트를 평문으로. heading_path 없음.
 - MD  : LlamaIndex MarkdownNodeParser로 # 헤딩 구조 보존 → heading_path 채움(유일하게 heading 있음).
 - TXT : 평문을 인코딩 감지(utf-8→cp949)해 읽어 문장 단위 분할. heading_path 없음.
 - XLSX: rag/xlsx_chunking (openpyxl, 시트/표 단위).

 공통: SentenceSplitter(chunk_size=512, chunk_overlap=50)로 크기 cap.
   - 큰 섹션이 비대해져 검색 정확도 떨어지는 것 방지 + LLM 컨텍스트 부담 완화
   - overlap 50으로 경계 의미 단절 완화

 결과 ChunkData: text / heading_path / page / chunk_index / meta.
 (LlamaIndex 의존성은 이 모듈 안에 격리 — ingestion은 ChunkData 도메인 타입만 본다.)
 """

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument
from llama_index.core import Document as LiDocument
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')

@dataclass
class ChunkData:
    """청킹 단계 중간 결과 (임베딩 / DB insert 전)."""
    text: str                   # 청크 본문
    heading_path: list[str]     # 예: ["3. 배송지연 보상", "3.2 지급 기준"]
    page: int | None            # PDF는 페이지 번호, DOCX는 None
    chunk_index: int            # 문서 내 순서 (0부터)
    meta: dict | None = None    # F1a: 청크 metadata (xlsx는 {"is_table": True, "sheet": 시트명})

def _read_text(file_path) -> str:
    """텍스트 파일을 인코딩 감지해 읽는다 (P2 CP949).
    utf-8 우선 → 실패 시 cp949(국내 txt 빈발) 폴백 → 그래도 실패면 replace 최후.
    (기존 utf-8+errors='replace'는 CP949를 조용히 �로 깨뜨렸음.)
    """
    raw = Path(file_path).read_bytes()
    for enc in ('utf-8', 'cp949'):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode('utf-8', errors='replace')


def _extract_heading_path(metadata: dict, text: str) -> list[str]:
    """조상 헤딩(metadata.header_path) + 자기 헤딩(text 첫 줄)을 합쳐 반환.

    MarkdownNodeParser는 'header_path'에 조상 경로만 '/' 구분자로 저장하고,
    자기 자신의 헤딩은 chunk text 첫 줄에 그대로 둠.

    예: metadata={'header_path': '/3. 표준 상담 스크립트/'}, text='### 3.1 ...\\n...'
        -> ['3. 표준 상담 스크립트', '3.1 ...']
    """
    path_str = metadata.get('header_path', '/')
    ancestors = [seg for seg in path_str.split('/') if seg]

    # 본문 첫 줄에서 현재 헤딩 추출 (있으면)
    first_line = text.lstrip().split('\n', 1)[0]
    m = _HEADING_RE.match(first_line)
    if m:
        ancestors.append(m.group(2).strip())

    return ancestors

def _pdf_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """PDF를 페이지별 텍스트로 (page 번호 보존 — 인용용). 텍스트 레이어 없으면 빈 문자열.
    비ML 추출(pdfplumber) — 표는 행/열이 공백 구분 텍스트로 나오며 구조 서식은 없다.
    """
    with pdfplumber.open(str(file_path)) as pdf:
        return [(i, page.extract_text() or '') for i, page in enumerate(pdf.pages, start=1)]


def _docx_text(file_path: str | Path) -> str:
    """DOCX 본문 문단 + 표 셀 텍스트를 평문으로 (python-docx, 비ML)."""
    doc = DocxDocument(str(file_path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:                       # docx 표는 구조화 포맷 → 셀 값 안정 추출
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(' | '.join(cells))
    return '\n'.join(parts)


def extract_text(file_path: str | Path) -> str:
    """파일 전체를 텍스트 하나로 추출한다 (청킹 없음). 채팅 첨부(컨텍스트 직접 주입)용.

    비ML 추출 — PDF 내 이미지·표·레이아웃 서식은 복원하지 않고 텍스트만 (채널톡 전략과 동일).
    """
    low = str(file_path).lower()
    if low.endswith('.pdf'):
        return '\n'.join(t for _, t in _pdf_pages(file_path)).strip()
    if low.endswith('.docx'):
        return _docx_text(file_path).strip()
    if low.endswith('.xlsx'):
        from rag.xlsx_chunking import chunk_xlsx     # 첨부 xlsx 지원 유지
        return '\n'.join(c.text for c in chunk_xlsx(str(file_path))).strip()
    return _read_text(file_path).strip()  # txt/md/기타


def chunk_txt(file_path: str | Path) -> list[ChunkData]:
    """평문 txt 청킹 — 인코딩 감지해 읽어 문장 단위로 분할.
    heading 구조가 없으므로 heading_path는 비운다.
    """
    text = _read_text(file_path)
    if not text.strip():
        return []                 # 빈 파일 — split_text('')가 ['']를 반환해 빈 청크가 새는 것 방지 (PDF 가드와 동일)
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    return [
        ChunkData(text=chunk, heading_path=[], page=None, chunk_index=i)
        for i, chunk in enumerate(splitter.split_text(text))
    ]


def chunk_file(file_path: str | Path) -> list[ChunkData]:
    """파일 한 개를 청크 리스트로 변환 (pdf/docx/md). 비ML 파싱.

    - PDF : pdfplumber 페이지별 텍스트 → 크기 분할. heading_path 없음(레이아웃 미인식), page 보존.
    - DOCX: python-docx 문단·표 → 크기 분할. heading_path 없음.
    - MD  : 마크다운 헤딩(#) 구조 보존(MarkdownNodeParser) → heading_path 채움.
    """
    low = str(file_path).lower()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    if low.endswith('.pdf'):
        out: list[ChunkData] = []
        for page_no, text in _pdf_pages(file_path):
            if not text.strip():
                continue          # 빈/스캔 페이지 — split_text('')가 ['']를 반환해 빈 청크가 새는 것 방지
            for chunk in splitter.split_text(text):
                out.append(ChunkData(text=chunk, heading_path=[], page=page_no, chunk_index=len(out)))
        return out

    if low.endswith('.docx'):
        text = _docx_text(file_path)
        return [
            ChunkData(text=chunk, heading_path=[], page=None, chunk_index=i)
            for i, chunk in enumerate(splitter.split_text(text))
        ]

    # .md — 마크다운 헤딩 구조 보존 (md는 # 헤딩이 네이티브)
    text = _read_text(file_path)
    nodes = splitter(MarkdownNodeParser()([LiDocument(text=text)]))
    return [
        ChunkData(
            text=node.get_content(),
            heading_path=_extract_heading_path(node.metadata, node.get_content()),
            page=None,
            chunk_index=i,
        )
        for i, node in enumerate(nodes)
    ]
