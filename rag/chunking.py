"""문서 파싱 + 청킹 모듈 (비ML 추출).

 형식별 섹션 파서 → _pack_sections(묶기) → SentenceSplitter(크기 cap) → ChunkData.
 워커(rag/documents.py)가 이 결과를 받아 임베딩 + DB 적재.

 **형식 분기는 chunk_file 한 곳뿐이다** — 두 곳에 있던 게 CLI로 xlsx를 넣으면 ZIP
 바이너리가 색인되던 버그의 원인이었다(#42).

 ================================================================
   형식별 파싱 — heading_path는 xlsx 포함 네 형식 모두 채운다 (txt만 빈다)
 ================================================================
 채움률은 문서 조판에 달렸다. 모의 코퍼스에서 100%였으나(2026-08-15) 실문서 PDF에서는
 93%였다 — 절 제목이 본문과 같은 크기인 문서는 헤딩이 안 잡힌다 (#137).

 - PDF : **docling**(레이아웃 ML + TableFormer, OCR 끔)이 기본 — #143. 인쇄 부산물·제목 분할·표 빈 셀을
         모델이 구조적으로 처리하고, 헤딩 층만 한국 규정 번호 체계('제 N 장'/'제 N 조')로 복원한다.
         `docling_enabled=False`면 pdfplumber 글자 크기 휴리스틱(비상 경로). page는 **청크마다 실제 쪽**.
         채팅 첨부(extract_text)는 항상 pdfplumber — 사용자 대기 경로라 콜드 스타트를 안 태운다.
 - DOCX: python-docx로 body를 문서 순서대로 읽어 Heading 스타일 기준 섹션 분할.
 - MD  : '#' 헤딩 정규식 + 코드 펜스 가드.
 - TXT : 평문(인코딩 감지 utf-8→cp949). 헤딩 구조가 없어 heading_path는 빈다.
 - XLSX: rag/xlsx_chunking (openpyxl). 시트=1청크라 이 파이프라인 밖 — 분할하지 않고
         150행 초과면 업로드를 거절한다.

 공통: SentenceSplitter(chunk_size=512, chunk_overlap=50)로 크기 cap.
   - 큰 섹션이 비대해져 검색 정확도 떨어지는 것 방지 + LLM 컨텍스트 부담 완화
   - overlap 50으로 경계 의미 단절 완화

 결과 ChunkData: text / heading_path / page / chunk_index / meta.
 (PDF 표 청크에 meta['is_table']을 붙이지 마라 — retriever의 '한 시트만' 필터가
  엑셀 기준이라 PDF 표 여러 개가 1개로 잘린다. rag/retriever.py:_keep_single_table 참조.)
 (LlamaIndex 의존성은 이 모듈 안에 격리 — 호출부는 ChunkData 도메인 타입만 본다.)
 """

import re
import threading
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from llama_index.core.node_parser import SentenceSplitter

from config import settings

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')
_DOCX_HEADING_RE = re.compile(r'^Heading (\d+)$')     # python-docx는 빌트인 헤딩을 영문명으로 준다
_MD_FENCE_RE = re.compile(r'^(`{3,}|~{3,})')          # 코드 펜스 — 안쪽 '#'은 헤딩이 아니다
# 섹션 묶음 예산(문자). 섹션 하나하나는 보통 100~150자라, 섹션=청크로 두면 top_k=5가
# 실어 나르는 컨텍스트가 옛 방식의 1/3로 줄어든다(실측 2211자→707자).
# 주의: 이 값이 상한으로 작동하는 일은 드물다 — _pack_sections의 경계 규칙(같은 대분류
# 아래 형제끼리만)이 먼저 걸려 실측 청크 중앙값은 200자대다. 실제 토큰 상한은 SentenceSplitter.
_PACK_CHARS = 500

# 청크 꼬리에 번호만 남는 것을 되돌릴 때 쓴다 (#137 결함 3). 정의점은 _strip_orphan_marker /
# _restore_leading_marker. 번호 뒤 공백은 물론 개행 하나까지 허용한다 — 원문에서 번호와 항목이
# 같은 줄일 때(공백)와 번호가 독립 줄일 때(개행)가 둘 다 나온다.
_ORPHAN_MARKER_RE = re.compile(r'(?:^|\n)[ \t]*(\d{1,2}\.|\(\d{1,2}\))[ \t]*\n?$')

# 한국 규정 문서의 장(章) 제목 — '제 3 장 복 무', '제3장 총칙'. docling은 장과 조를 같은 레벨로
# 내놓으므로(#143 실측: 26쪽 141개 헤딩 전부 level 1) 이 패턴으로 층을 복원한다. 마크다운의 '#'처럼
# 문서가 스스로 선언하는 층이라 폰트 크기 휴리스틱과 성격이 다르다 — 코퍼스가 아니라 도메인 관례다.
_CHAPTER_RE = re.compile(r'^제\s*\d+\s*장\b')

# docling 그림 요소의 자리표시. 텍스트 PDF 안에 렌더된 도표는 파서로는 못 읽는다(#137 결함 4).
# 지우지 않고 남겨 "여기 도표가 있었다"를 알린다.
# **캡션이 붙으면 이 자리를 캡션이 대체한다**(#162) — 자리표시가 남아 있다는 것은 VLM 호출이
# 실패했다는 뜻이다(count_picture_placeholders).
_PICTURE_PLACEHOLDER = '<!-- image -->'

# 그림 캡션 프롬프트(#162). **도형 어휘(박스·화살표·불릿)를 쓰지 않는다** — 하나의 프롬프트가
# 모든 문서의 모든 그림에 적용되므로 흐름도를 전제한 문구는 표·사진에 해롭다. 실측(2026-09-14):
# 흐름도를 전제한 문구는 불릿 귀속을 한 칸 밀거나 통째로 생략했고, 아래 타입 무관 문구가 흐름도·
# 수식·표·차트 4유형에서 가장 정확했다.
_PICTURE_CAPTION_PROMPT = (
    '이미지의 내용을 한국어로 옮겨라.\n'
    '- 이미지 안의 글자를 하나도 빠뜨리지 말고 원문 그대로 옮긴다. 고쳐 쓰거나 요약하지 않는다.\n'
    '- 요소들 사이에 관계가 있으면 그 관계를 함께 적는다.\n'
    '- 관계가 분명하지 않으면 보이는 위치대로 나열만 한다.\n'
    '- 이미지에 없는 내용은 쓰지 않는다.'
)

@dataclass
class ChunkData:
    """청킹 단계 중간 결과 (임베딩 / DB insert 전)."""
    text: str                   # 청크 본문
    heading_path: list[str]     # 예: ["3. 배송지연 보상", "3.2 지급 기준"]
    page: int | None            # PDF는 페이지 번호, DOCX는 None
    chunk_index: int            # 문서 내 순서 (0부터)
    meta: dict | None = None    # F1a: 청크 metadata (xlsx는 {"is_table": True, "sheet": 시트명})


@dataclass
class _Section:
    """형식별 파서 → _pack_sections 사이의 내부 계약. 이 모듈 밖으로 나가지 않는다.

    파서 넷(_pdf/_docx/_md/_txt_sections)이 전부 이 형태를 내놓기 때문에 그 뒤 단계
    (묶기 → 크기 분할 → ChunkData)를 형식과 무관하게 한 번만 쓴다.
    """
    heading_path: list[str]
    page: int | None            # PDF만 채운다 — 나머지는 파일에 페이지 개념이 없다
    body: str
    # body를 '\n'으로 나눈 줄과 1:1로 대응하는 쪽 번호. PDF만 채운다.
    # page 하나로는 여러 쪽에 걸친 섹션을 표현할 수 없어 뒤쪽 청크가 앞 쪽 번호를
    # 물려받았다 (#137 결함 6 — 실측 193청크 중 95개가 틀린 쪽을 가리켰다).
    line_pages: list[int | None] | None = None


# (page, text, level) — 두 PDF 경로(docling / pdfplumber)가 _build_sections에 흘리는 공통 단위.
# level이 None이면 본문 줄, 정수면 그 레벨의 헤딩. text는 여러 줄일 수 있다(표 markdown).
_Item = tuple[int | None, str, int | None]


def to_markdown_table(header: list[str], body: list[list[str]]) -> str:
    """헤더 + 데이터행 → markdown 표. 헤더가 청크에 포함돼 컬럼 의미가 보존된다.

    PDF·XLSX 표 직렬화의 단일 정의점 (rag/xlsx_chunking이 이 이름을 import한다).
    빈 셀은 자리를 지킨다 — 지우면 뒷 열이 앞으로 당겨져 값이 다른 열로 읽힌다
    (#137 결함 5: 개인부담 O가 회사지원 O로 읽히는 형태. 유실이 0이라 커버리지에 안 잡히고 답만 틀린다).
    """
    lines = ['| ' + ' | '.join(header) + ' |',
             '| ' + ' | '.join('---' for _ in header) + ' |']
    for row in body:
        # 행 길이를 헤더에 맞춤 (짧으면 빈칸 채움)
        cells = (row + [''] * len(header))[:len(header)]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


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


def _pdf_pages(file_path: str | Path) -> list[tuple[int, str]]:
    """PDF를 페이지별 텍스트로 (page 번호 보존 — 인용용). 텍스트 레이어 없으면 빈 문자열.
    비ML 추출(pdfplumber) — 표는 행/열이 공백 구분 텍스트로 나오며 구조 서식은 없다.
    채팅 첨부(extract_text) 전용 — 사용자 대기 경로라 docling 콜드 스타트(4.4초)를 태우지 않는다.
    """
    with pdfplumber.open(str(file_path)) as pdf:
        return [(i, page.extract_text() or '') for i, page in enumerate(pdf.pages, start=1)]


# ── 섹션 조립 — 두 PDF 경로의 공통 후단 ──────────────────────────────────────────

def _build_sections(items: list[_Item]) -> list[_Section]:
    """(page, text, level) 흐름을 헤딩 스택으로 _Section 목록에 접는다.

    docx·md 파서와 같은 스택 규약(`del stack[level-1:]` + append)이다. 그 위에 #137에서 확인된
    두 가지를 더한다:
      - **줄별 쪽 번호(line_pages)** — 섹션이 여러 쪽에 걸치면 뒤쪽 청크가 앞 쪽을 물려받았다
        (결함 6). 표 markdown처럼 text가 여러 줄이면 그 줄 수만큼 같은 쪽을 채워 1:1을 지킨다.
      - **본문 없는 헤딩 보존(orphans)** — 같은 레벨 헤딩이 연달아 오면 `del stack[level-1:]`이
        직전 것을 지워 흔적 없이 소멸했다(결함 1·2의 소멸 경로). 본문을 한 번도 못 만든 헤딩은
        붙잡아 다음 섹션 본문에 되살린다. (부모→자식처럼 레벨이 깊어지는 경우는 스택 뒤에 붙으므로
        원래 안전하다 — _docx_sections docstring의 "정보 손실은 없다"는 그 경우에만 맞다.)
    """
    sections: list[_Section] = []
    stack: list[str] = []
    buf: list[str] = []
    buf_pages: list[int | None] = []
    orphans: list[str] = []
    start_page: int | None = items[0][0] if items else None
    last_page: int | None = start_page

    def add(text: str, page: int | None) -> None:
        buf.append(text)
        buf_pages.extend([page] * (text.count('\n') + 1))

    def flush() -> bool:
        """섹션을 만들었으면 True. 본문 없는 헤딩은 섹션이 되지 않는다 (docx·md도 같은 가드)."""
        if not buf:
            return False
        sections.append(_Section(list(stack), start_page, '\n'.join(buf), list(buf_pages)))
        buf.clear()
        buf_pages.clear()
        return True

    for page, text, level in items:
        if page is not None:
            last_page = page
        if level is not None:
            produced = flush()
            removed = stack[level - 1:]
            if not produced and removed:
                orphans.extend(removed)
            del stack[level - 1:]
            stack.append(text)
        else:
            # 섹션의 page는 '첫 본문 줄'의 페이지다 — 헤딩이 페이지 끝에 걸리면
            # 본문은 다음 쪽에서 시작하고, 인용은 본문이 있는 쪽을 가리켜야 맞다.
            if not buf:
                start_page = page
                for orphan in orphans:
                    add(orphan, page)
                orphans.clear()
            add(text, page)
    if orphans and not buf:            # 문서 끝에 남은 것도 버리지 않는다
        start_page = last_page
        for orphan in orphans:
            add(orphan, last_page)
    flush()
    return sections


# ── PDF 경로 ① docling — 기본 ────────────────────────────────────────────────────

_DOCLING_LOCK = threading.Lock()
_docling_converter = None
_docling_semaphore: threading.Semaphore | None = None


def _docling_runtime():
    """프로세스당 1회: 변환기(모델 로딩)와 변환 동시성 세마포어.

    모델은 한 번만 올린다 — 이후 쪽당 0.3초(#143 실측). **리눅스 컨테이너 실측(2026-09-13)**:
    변환 피크 1.8GB·상주 1.4GB이고, 같은 프로세스로 12회 반복하면 4~6회차에서 평평해진다(누수 없음).
    다만 작업 전으로 돌아오지도 않는다 — 모델이 상주하고 glibc가 해제분을 OS에 바로 반납하지 않는다.
    컨테이너 메모리 상한(docker-compose.yml)은 이 피크 위에 잡는다. 세마포어는 arq `max_jobs`(10)와 별개로
    **변환만** 직렬화한다 — chunk_file은 to_thread로 돌아 잡이 겹치면 변환도 겹쳐 메모리가
    배수로 난다. 임베딩 I/O는 max_jobs대로 계속 겹치게 둔다.
    """
    global _docling_converter, _docling_semaphore
    with _DOCLING_LOCK:
        if _docling_converter is None:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (PdfPipelineOptions,
                                                            PictureDescriptionApiOptions,
                                                            TableFormerMode)
            from docling.document_converter import DocumentConverter, PdfFormatOption

            opts = PdfPipelineOptions(artifacts_path=settings.docling_artifacts_path)
            # 메모리 노브 — config.py 주석 참조. docling_settings(perf.page_batch_size)는 쓰지 않는다:
            # 2.126의 기본 파이프라인이 그 값을 읽지 않는다(옛 Legacy 경로 전용).
            opts.layout_batch_size = settings.docling_layout_batch_size
            opts.queue_max_size = settings.docling_queue_max_size
            opts.do_ocr = settings.docling_do_ocr
            opts.do_table_structure = True
            opts.table_structure_options.mode = (TableFormerMode.FAST if settings.docling_table_mode == 'fast'
                                                 else TableFormerMode.ACCURATE)
            opts.table_structure_options.do_cell_matching = settings.docling_do_cell_matching
            opts.document_timeout = settings.docling_document_timeout_seconds
            opts.accelerator_options = AcceleratorOptions(
                num_threads=settings.docling_num_threads,
                device=AcceleratorDevice(settings.docling_device))
            # 그림 캡션(#162). enable_remote_services를 안 켜면 do_picture_description이 설정
            # 단계에서 OperationNotAllowed로 죽는다 — VLM 다운과 달리 docling이 삼켜주지 않는
            # 경로라 항상 같이 켠다. VLM에 못 닿으면 캡션이 안 붙고 자리표시가 남을 뿐이다.
            opts.enable_remote_services = True
            opts.do_picture_description = True
            opts.picture_description_options = PictureDescriptionApiOptions(
                url=settings.vlm_caption_url,
                params={'model': settings.vlm_caption_model},
                prompt=_PICTURE_CAPTION_PROMPT,
                scale=settings.vlm_caption_scale,
                timeout=settings.vlm_caption_timeout_seconds)
            _docling_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
            _docling_semaphore = threading.Semaphore(settings.docling_max_concurrency)
    return _docling_converter, _docling_semaphore


def _docling_norm(text: str) -> str:
    """docling 텍스트의 NBSP(\\xa0)를 보통 공백으로. 표 셀·헤딩에 섞여 들어와 검색 토큰을 깨뜨린다."""
    return re.sub(r'[  ]+', ' ', text).strip()


def _docling_table_markdown(item, doc) -> str | None:
    """docling 표 → markdown. 헤더는 dataframe 컬럼, 빈 셀은 ''로 자리를 지킨다(xlsx와 같은 규약)."""
    df = item.export_to_dataframe(doc)
    if df.shape[1] == 0:
        return None
    header = [_docling_norm(str(c)) for c in df.columns]
    rows = [[_docling_norm('' if v is None else str(v)) for v in row] for row in df.itertuples(index=False)]
    return to_markdown_table(header, rows)


def count_picture_placeholders(chunks) -> int:
    """청크들에 남은 그림 자리표시 수(#162). 캡션이 안 붙은 그림의 개수다.

    캡션이 켜져 있는데 이 값이 0이 아니면 VLM 호출이 실패했다는 뜻이다 — docling이 VLM 실패를
    삼키므로(예외 없음) 이것이 유일한 관측 창이다. 지표 기록은 호출부(rag/documents.py)가 한다:
    이 모듈은 0층 leaf라 rag/metrics.py를 import하지 않는다(AGENTS.md 계층 규칙).

    **캡션이 붙은 그림은 셀 수 없다** — 캡션은 일반 본문과 구분되지 않게 들어가기 때문이다(설계 결정).
    그래서 이 지표는 성공률이 아니라 실패 신호다.
    """
    return sum(c.text.count(_PICTURE_PLACEHOLDER) for c in chunks)


def _docling_picture_caption(element) -> str:
    """그림 요소의 VLM 캡션(#162). 없으면 빈 문자열.

    docling은 캡션을 `annotations`에 담는다(2.126 기준). 여러 개가 달릴 수 있어 첫 비어 있지 않은
    것을 쓴다 — 우리 설정은 캡션 엔진 하나뿐이라 실질적으로 0개 또는 1개다.
    """
    for ann in getattr(element, 'annotations', None) or []:
        text = (getattr(ann, 'text', '') or '').strip()
        if text:
            return text
    return ''


def _docling_elements(doc):
    """docling 문서 → (label, text, page, table_markdown) 튜플 흐름. 읽기 순서.

    _sections_from_docling_elements가 소비한다. 둘로 나눈 이유: 매핑 규칙(헤딩 층·표·그림)을
    docling 없이 튜플만으로 단위 테스트하기 위해서다 — 모델 다운로드가 필요한 테스트와 분리.
    """
    for element, _depth in doc.iterate_items():
        label = element.label.value if hasattr(element.label, 'value') else str(element.label)
        page = element.prov[0].page_no if getattr(element, 'prov', None) else None
        if label == 'table':
            yield label, '', page, _docling_table_markdown(element, doc)
        elif label == 'picture':
            # 캡션(#162)은 element.text가 아니라 annotations에 붙는다 — text만 보면 조용히 버려진다.
            # VLM이 죽거나 꺼져 있으면 annotations가 비어 빈 문자열이 나가고, 소비자가 자리표시로 되돌린다.
            yield label, _docling_picture_caption(element), page, None
        else:
            text = getattr(element, 'text', '') or ''
            # 목록 항목의 번호·글머리('2.1', '-', '(1)')는 text에서 떼어 marker에 둔다. 빠뜨리면
            # '2.1 얼리버드 할인율 상향'이 '얼리버드 할인율 상향'으로 색인된다(모의 코퍼스 33건 대조에서
            # 발견). docling의 export_to_markdown과 같은 방식으로 다시 붙인다.
            marker = getattr(element, 'marker', '') or ''
            if label == 'list_item' and marker:
                text = f'{marker} {text}'
            yield label, text, page, None


def _sections_from_docling_elements(elements) -> list[_Section]:
    """(label, text, page, table_markdown) 흐름 → _Section. 헤딩 층 복원 규칙의 정의점.

    docling은 장·조를 구분하지 않으므로(_CHAPTER_RE 참조) 여기서 층을 만든다:
      첫 헤딩 = 문서 제목(L1) / '제 N 장' = L2 / 그 외 헤딩 = L3 (장이 아직 없으면 L2).
    그러면 _pack_sections의 joinable(공통 조상 깊이 ≥2)이 현행과 같이 동작해 같은 장 아래
    조항끼리 묶인다. 이 규칙 없이 두면 26쪽 규정이 73→170청크로 갈렸다(#143 실측).
    표는 markdown 그대로, 그림은 캡션(없으면 _PICTURE_PLACEHOLDER), 나머지 텍스트 요소는 본문 줄.
    """
    items: list[_Item] = []
    seen_title = False
    seen_chapter = False
    for label, text, page, table_markdown in elements:
        if label == 'section_header':
            text = _docling_norm(text)
            if not text:
                continue
            if not seen_title:
                level, seen_title = 1, True
            elif _CHAPTER_RE.match(text):
                level, seen_chapter = 2, True
            else:
                level = 3 if seen_chapter else 2
            items.append((page, text, level))
        elif label == 'table':
            if table_markdown:
                items.append((page, table_markdown, None))
        elif label == 'picture':
            # 캡션이 있으면 그것이 본문이다(#162). 없으면 자리표시 — 둘 다 한 줄 항목으로 들어간다.
            caption = _docling_norm(text)
            items.append((page, caption or _PICTURE_PLACEHOLDER, None))
        else:
            text = _docling_norm(text)
            if text:
                items.append((page, text, None))
    return _build_sections(items)


def _docling_sections(file_path: str | Path) -> list[_Section]:
    """PDF → docling(레이아웃 ML + TableFormer) → _Section. PDF 인제스션의 기본 경로 (#143).

    pdfplumber 휴리스틱(#141)이 손으로 막던 것을 레이아웃 모델이 구조적으로 대신한다:
      - 인쇄 헤더·URL 푸터는 page_header/page_footer로 분류돼 iterate_items()에 나오지 않는다 (결함 1)
      - 제목이 폰트 혼용으로 쪼개지지 않는다 (결함 2)
      - 표는 셀 구조로 나와 빈 셀·병합 셀이 보존된다 (결함 5)
    임계 상수가 0개다. 헤딩 층 복원은 _sections_from_docling_elements가 한다.

    실패(예외·타임아웃)는 그대로 올린다 — pdfplumber 폴백 없음. 저품질 색인을 조용히 만들지 않는다
    (strict-grounded). 호출부(index_pending_document)가 failed로 기록한다.
    이미지 도표는 VLM 캡션으로 읽는다 (#162) — VLM에 못 닿아 캡션이 없으면
    _PICTURE_PLACEHOLDER 자리표시가 남는다. 격자 표 이미지는 docling이 TableItem으로 분류해
    이 경로를 타지 않는다(별도 이슈).
    """
    converter, semaphore = _docling_runtime()
    with semaphore:
        result = converter.convert(str(file_path), max_num_pages=settings.docling_max_num_pages,
                                   raises_on_error=True)
    return _sections_from_docling_elements(_docling_elements(result.document))


# ── PDF 경로 ② pdfplumber — 비상 스위치 (docling_enabled=False) ─────────────────

@dataclass
class _Line:
    """pdfplumber 줄 하나 — 헤딩 판정에 크기가 필요해 함께 나른다."""
    page: int
    text: str
    size: float                 # 줄의 최대 글자 크기


def _pdf_lines(file_path: str | Path) -> list[_Line]:
    """PDF를 (page, 줄, 최대 글자크기)로. extract_text()는 크기를 버리므로 extract_text_lines()."""
    out: list[_Line] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for line in page.extract_text_lines():
                text = (line.get('text') or '').strip()
                if not text:
                    continue
                sizes = [c['size'] for c in line.get('chars', []) if c.get('size')]
                out.append(_Line(page_no, text, round(max(sizes), 1) if sizes else 0.0))
    return out


def _pdfplumber_sections(file_path: str | Path) -> list[_Section]:
    """PDF → pdfplumber 글자 크기 휴리스틱 → _Section. **비상 경로** — `docling_enabled=False`일 때만.

    본문 크기 = 글자 수 기준 최빈 크기, 그보다 큰 줄이 헤딩. 크기 차이가 없으면 헤딩 0개로 폴백.
    #137에서 확정된 결함 1(인쇄 부산물이 투표를 전복)·2(폰트 혼용으로 제목 분할)·5(표 빈 셀 소실)를
    **이 경로는 막지 않는다** — #141의 손 수정은 docling 도입(#143)으로 제거했다. 표는 공백 구분
    평문으로 들어가고 인쇄 헤더·URL 푸터가 본문에 섞인다. docling을 못 쓰는 상황의 임시 수단이다.
    결함 3·6·③(번호 복원·줄별 쪽·고아 헤딩)은 _build_sections·chunk_file 단계라 여기서도 유지된다.
    """
    lines = _pdf_lines(file_path)
    if not lines:
        return []
    weight: dict[float, int] = {}
    for line in lines:
        weight[line.size] = weight.get(line.size, 0) + len(line.text)
    body_size = max(weight, key=lambda size: weight[size])
    level_of = {size: level for level, size in
                enumerate(sorted((s for s in weight if s > body_size), reverse=True), start=1)}
    return _build_sections([(line.page, line.text, level_of.get(line.size)) for line in lines])


def _pdf_sections(file_path: str | Path) -> list[_Section]:
    """PDF 파서 디스패처 — 기본 docling, `docling_enabled=False`면 pdfplumber 비상 경로 (#143)."""
    if settings.docling_enabled:
        return _docling_sections(file_path)
    return _pdfplumber_sections(file_path)


def _docx_items(file_path: str | Path):
    """DOCX body를 **문서 원래 순서대로** (is_heading, level, text)로 흘린다.

    doc.paragraphs와 doc.tables를 따로 훑으면 표가 전부 문서 끝으로 밀린다
    (예: '2.1 기준표' 밑의 표가 자기 섹션에서 떨어져 나와 마지막 청크로 감).
    body 자식을 직접 순회해 문단·표의 상대 순서를 보존한다.
    """
    doc = DocxDocument(str(file_path))
    for child in doc.element.body.iterchildren():
        tag = child.tag.split('}')[-1]
        if tag == 'p':
            para = DocxParagraph(child, doc)
            text = para.text.strip()
            if not text:
                continue
            # style이 None인 문단이 실제로 있다 (스타일 정의가 빠진 docx) — getattr로 방어
            m = _DOCX_HEADING_RE.match(getattr(para.style, 'name', None) or '')
            yield (bool(m), int(m.group(1)) if m else 0, text)
        elif tag == 'tbl':                         # docx 표는 구조화 포맷 → 셀 값 안정 추출
            for row in DocxTable(child, doc).rows:
                # 빈 셀도 자리를 지킨다 — 걸러내면 뒷 컬럼이 앞으로 당겨져 헤더와 어긋난다.
                # ['제주','','5000원','도서산간 별도'] → '제주 | 5000원 | 도서산간 별도'가 되어
                # LLM이 '조건=5000원'으로 읽는다. 완전 빈 행만 스킵 (xlsx 파서와 같은 원리).
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    yield (False, 0, ' | '.join(cells))


def _docx_text(file_path: str | Path) -> str:
    """DOCX 전체를 평문 하나로 (채팅 첨부 — 컨텍스트 직접 주입용). 헤딩 포함, 문서 순서 보존."""
    return '\n'.join(text for _, _, text in _docx_items(file_path))


def _docx_sections(file_path: str | Path) -> list[_Section]:
    """DOCX를 (heading_path, 본문) 섹션 목록으로 — Heading 스타일이 경계.

    헤딩 텍스트는 본문에 넣지 않는다. heading_path가 들고 있고, 인덱스 입력엔
    rag/index_text가 앞에 붙이므로 본문에까지 넣으면 같은 문구가 두 번 들어간다.

    본문이 없는 헤딩(상위 목차처럼 바로 하위 헤딩이 오는 경우)은 섹션을 만들지 않는다 —
    '헤딩만 있고 내용 없는 청크'가 인덱스에 새는 것을 막는다(md 경로에서 실제로 관측된 노이즈).
    그 헤딩은 하위 섹션의 heading_path에 조상으로 남으므로 정보 손실은 없다.
    """
    sections: list[_Section] = []
    stack: list[str] = []      # 현재 헤딩 경로 — 인덱스가 곧 레벨-1
    lines: list[str] = []

    def flush() -> None:
        if lines:
            sections.append(_Section(list(stack), None, '\n'.join(lines)))   # page는 docx에 없음
            lines.clear()

    for is_heading, level, text in _docx_items(file_path):
        if is_heading:
            flush()                    # 이전 섹션 마감 후 경로 갱신
            del stack[level - 1:]      # 같은 레벨·하위 레벨 걷어내고
            stack.append(text)
        else:
            lines.append(text)
    flush()
    return sections


def _common_prefix(paths: list[list[str]]) -> list[str]:
    """heading_path들의 공통 조상 경로."""
    common = list(paths[0])
    for path in paths[1:]:
        while common != path[:len(common)]:
            common.pop()
    return common


def _pack_sections(
        sections: list[_Section],
        max_chars: int = _PACK_CHARS,
) -> list[_Section]:
    """연속 섹션을 예산까지 묶는다 — 섹션 하나가 청크 하나면 너무 잘아지므로.
    묶음의 page는 첫 섹션 것을 쓴다.

    묶음의 heading_path는 참여 섹션들의 **공통 조상**이고, 그보다 깊은 헤딩은 본문에
    줄로 남긴다. 정보 손실 없이 청크 크기만 옛 방식 수준으로 되돌린다.
      예: '2.1 신청 채널'과 '2.2 최소 후원 금액'을 묶으면
          heading_path=[문서명, '2. 정기후원 신청'], 본문 앞에 각 소제목이 한 줄씩.

    **묶기를 실제로 막는 건 예산이 아니라 경계다.** `joinable`이 공통 조상 깊이 2를
    요구하는데 heading_path[0]이 문서명이므로, 이는 "같은 대분류(H2) 아래 형제끼리만
    묶는다"는 뜻이다. 상위가 다르면 크기가 남아도 끊는다 — 무작정 묶으면 공통 조상이
    문서명 하나로 떨어져 heading_path가 무의미해지기 때문이다.
    그래서 실측 청크 중앙값은 max_chars(500)보다 훨씬 작은 200자대다 (2026-08-15).

    `size`는 body 길이만 센다 — flush가 덧붙이는 헤딩 줄은 빠지므로 max_chars는
    엄밀한 상한이 아니라 소프트 예산이다. 실제 상한은 뒤이어 SentenceSplitter가 건다.
    """
    packed: list[_Section] = []
    buf: list[_Section] = []
    size = 0

    def joinable(path: list[str]) -> bool:
        """buf에 이 섹션을 더해도 공통 조상이 문서명 아래로 유지되는가."""
        return len(_common_prefix([s.heading_path for s in buf] + [path])) >= 2

    def flush() -> None:
        nonlocal size
        if not buf:
            return
        common = _common_prefix([s.heading_path for s in buf])
        lines: list[str] = []
        pages: list[int | None] = []
        for s in buf:
            extra = s.heading_path[len(common):]   # 공통 조상보다 깊은 헤딩은 본문에 보존
            lines.extend(extra)
            pages.extend([s.page] * len(extra))
            lines.append(s.body)
            # 줄별 page를 그대로 이어붙인다. 없으면(docx·md·txt) 섹션 page로 채운다 —
            # 묶음 page를 buf[0].page 하나로만 두면 뒤 섹션 본문이 앞 쪽 번호를
            # 물려받는다 (#137 결함 6의 두 근원 중 하나).
            body_lines = s.body.split('\n')
            if s.line_pages and len(s.line_pages) == len(body_lines):
                pages.extend(s.line_pages)
            else:
                pages.extend([s.page] * len(body_lines))
        packed.append(_Section(common, buf[0].page, '\n'.join(lines), pages))
        buf.clear()
        size = 0

    for section in sections:
        if buf and (size + len(section.body) > max_chars or not joinable(section.heading_path)):
            flush()
        buf.append(section)
        size += len(section.body)
    flush()
    return packed


def pdf_image_area_ratio(file_path: str | Path) -> float:
    """PDF 쪽 면적 대비 삽입 이미지 면적 비율(%). 도표가 이미지로 렌더된 문서 감지용.

    pdfplumber는 이미지 안 글자를 한 글자도 뽑지 못한다. 쪽에 다른 텍스트가 있으면
    '청크 0개 → failed' 검사와 '빈 쪽 수' 검사를 **둘 다 통과**해, 도표만 조용히
    사라진 채 문서가 ready가 된다 (#137 결함 4 — 실측 3건, 프로세스 흐름도 2개가
    통째로 소멸해 섹션 둘이 근거 0이 된 문서 포함).

    텍스트 커버리지로는 절대 잡히지 않는다 — 탐지 신호는 면적뿐이다.
    실측 분포(실문서 21건): 도표가 이미지인 문서 21.1% / 4.4% / 0.7%, 나머지 18건 0%.
    OCR은 도입하지 않는다(로컬 ML 무게 — [[docling_removal]]). 감지해서 알린다.
    """
    if not str(file_path).lower().endswith('.pdf'):
        return 0.0
    image_area = page_area = 0.0
    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            page_area += (page.width or 0) * (page.height or 0)
            for image in page.images:
                image_area += (abs(image['x1'] - image['x0'])
                               * abs(image['bottom'] - image['top']))
    return image_area / page_area * 100 if page_area else 0.0


def _page_at(section: _Section, offset: int) -> int | None:
    """섹션 본문의 문자 오프셋이 실제로 몇 쪽에서 왔는지 (#137 결함 6).

    line_pages가 없으면(docx·md·txt) 섹션 page를 그대로 쓴다 — 그 형식들은 page가
    애초에 None이라 동작이 바뀌지 않는다.
    """
    if not section.line_pages:
        return section.page
    position = 0
    for index, line in enumerate(section.body.split('\n')):
        position += len(line) + 1                      # '\n' 포함
        if offset < position:
            return (section.line_pages[index] if index < len(section.line_pages)
                    else section.page)
    return section.line_pages[-1]


def _strip_orphan_marker(chunk: str) -> str:
    """청크 꼬리에 목록 번호만 남았으면 떼어낸다 (#137 결함 3).

    SentenceSplitter의 기본 secondary_chunking_regex가 마침표를 문맥 없이 문장 끝으로
    봐서 '3. 항목' → ['3.', ' 항목']으로 끊는다. 청크 경계가 여기 걸리면 번호는 앞
    청크 꼬리에 고아로 남는다(실측 266청크 중 50개, 조항형 규정에 집중). 그 자리에선
    아무 의미가 없으므로 뗀다 — 항목 쪽에 붙이는 것은 _restore_leading_marker가 한다.
    """
    match = _ORPHAN_MARKER_RE.search(chunk)
    return chunk[:match.start()].rstrip() if match and match.end() == len(chunk) else chunk


def _restore_leading_marker(body: str, offset: int, chunk: str) -> str:
    """청크가 목록 번호 **바로 뒤**에서 시작하면 그 번호를 머리에 되살린다 (#137 결함 3).

    판정은 오프셋으로 한다 — 원문에서 이 청크 앞에 붙어 있던 것이 번호뿐일 때만
    주입한다. 다음 청크가 overlap으로 번호를 이미 갖고 있으면 offset이 번호보다 앞이라
    조건이 성립하지 않는다.

    **추측으로 하면 안 된다.** 초기 구현이 '다음 청크가 그 번호로 시작하지 않으면 주입'
    으로 판정했다가, overlap 안쪽에 번호가 이미 있는 경우를 놓쳐 **엉뚱한 자리에 번호를
    중복 주입**했다(실측: txt 코퍼스 6건에서 `1. [상황 2] 배송 지연 항의\n\n1. "…"`).
    overlap 길이는 토큰 기준이라 문자 수로 역산할 수 없어, 오프셋만이 정확한 근거다.
    """
    prefix = body[:offset]
    match = _ORPHAN_MARKER_RE.search(prefix)
    if match and match.end() == len(prefix):
        return f'{match.group(1)} {chunk.lstrip()}'
    return chunk


def extract_text(file_path: str | Path) -> str:
    """파일 전체를 텍스트 하나로 추출한다 (청킹 없음). 채팅 첨부(컨텍스트 직접 주입)용.

    비ML 추출. PDF 표는 chunk_file 경로에서만 구조를 살린다(_pdf_tables) — 이 함수는
    첨부를 컨텍스트에 통째로 넣는 용도라 표를 평문으로 흘린다.
    **이미지로 렌더된 도표는 어느 경로에서도 못 읽는다** — 정책이 아니라 한계다.
    채널톡도 같은 선을 긋고 "향후 업데이트에서 지원 예정"으로 남겨 두었다
    ("PDF 내 이미지, 표, 레이아웃 등은 인식되지 않고, 텍스트 콘텐츠만 ALF가 참조할 수
    있습니다", docs.channel.io 지식 ALF v2). 감지는 pdf_image_area_ratio가 한다.
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


def _txt_sections(file_path: str | Path) -> list[_Section]:
    """평문 txt — 헤딩 개념이 없으므로 파일 전체가 섹션 하나다.

    빈 파일은 섹션 0개로 떨어뜨린다. split_text('')가 ['']를 반환해 빈 청크가
    새는 것을 막는다 (다른 파서의 `if buf:` 가드와 같은 목적).
    """
    text = _read_text(file_path)
    return [_Section([], None, text)] if text.strip() else []


def _md_sections(file_path: str | Path) -> list[_Section]:
    """마크다운을 (heading_path, page=None, 본문) 섹션으로 — '#' 헤딩이 경계.

    docx·pdf와 같은 스택 관리(`del stack[level-1:]`)를 쓴다. 세 파서가 같은 계약을
    내놓아야 `_pack_sections` 이후를 공유할 수 있다.

    **헤딩은 본문에 넣지 않는다** — `_docx_sections`와 같은 이유로, heading_path가
    들고 있고 인덱스 입력엔 `rag/index_text`가 앞에 붙이므로 본문에까지 넣으면 같은
    문구가 두 번 들어간다. 옛 MarkdownNodeParser 경로는 자기 헤딩을 본문 첫 줄에
    남겨서 md만 이 원칙에서 벗어나 있었다.

    **본문 없는 헤딩은 섹션이 되지 않는다** — 헤딩을 버퍼에 안 넣으므로 `flush()`의
    가드에 자동으로 걸린다. 옛 경로는 이걸 안 해서 md 청크의 19%가 `'## 4. 결제·정산'`
    같은 헤딩 한 줄뿐이었다(실측 78개 중 15개). `_docx_sections` docstring이 이 노이즈를
    *"md 경로에서 실제로 관측된"* 이라고 적어두고도 docx에만 가드를 넣었던 것을 맞춘다.

    **코드 펜스 안의 '#'은 헤딩이 아니다.** 백틱과 틸드 둘 다 본다 — 옛 파서는 백틱만
    처리했다. 들여쓰기 코드블록은 `_HEADING_RE`가 열 0을 요구해 공짜로 걸러진다.
    """
    sections: list[_Section] = []
    stack: list[str] = []
    buf: list[str] = []
    fence: str | None = None       # 열려 있는 펜스 마커('```'/'~~~~' 등), 없으면 None

    def flush() -> None:
        # 빈 줄만 남은 버퍼는 섹션이 아니다 — 헤딩 사이의 빈 줄이 빈 섹션을 만들지 않게.
        # (docx·pdf는 파서가 빈 줄을 애초에 안 흘려서 `if buf:`로 충분하다.)
        if any(line.strip() for line in buf):
            sections.append(_Section(list(stack), None, '\n'.join(buf).strip('\n')))
        buf.clear()

    for line in _read_text(file_path).splitlines():
        marker = _MD_FENCE_RE.match(line.strip())
        if marker:
            token = marker.group(1)
            if fence is None:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = None       # 같은 문자·같은 길이 이상으로만 닫힌다 (CommonMark)
            buf.append(line)
            continue
        m = None if fence else _HEADING_RE.match(line)
        if m:
            flush()
            del stack[len(m.group(1)) - 1:]
            stack.append(m.group(2).strip())
        else:
            buf.append(line)       # 빈 줄도 보존 — md는 빈 줄이 문단 구분이다
    flush()
    return sections


# 형식 → 섹션 파서. 이 넷은 _Section으로 수렴하므로 뒤 단계(묶기·분할)를 공유한다.
# xlsx만 빠져 있다 — 시트=1청크·분할 없음·행상한 초과 시 거절이라 계약이 근본적으로 다르다.
_SECTION_PARSERS = {
    '.pdf': _pdf_sections,      # 글자 크기 휴리스틱
    '.docx': _docx_sections,    # Heading 스타일
    '.md': _md_sections,        # '#' 정규식 + 펜스 가드
    '.txt': _txt_sections,      # 헤딩 없음 — 통째로 섹션 1개
}


def chunk_file(file_path: str | Path, *, description: str = '') -> list[ChunkData]:
    """파일 한 개를 청크 리스트로. **형식 분기는 여기 한 곳뿐이다.**

    분기가 두 곳(여기 + 호출부)에 있던 게 CLI로 xlsx를 넣으면 ZIP 바이너리가 색인되던
    버그의 원인이었다(#42). 호출부는 이제 확장자를 보지 않는다.

    xlsx는 `chunk_xlsx`로 위임한다 — 시트=1청크, 분할 없음, 150행 초과 시 업로드 거절,
    `description` 병합. 나머지는 섹션 추출 → `_pack_sections` → 크기 분할을 공유한다.
    `description`은 xlsx 전용(다른 형식은 무시).
    """
    suffix = Path(file_path).suffix.lower()
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)

    if suffix == '.xlsx':
        from rag.xlsx_chunking import chunk_xlsx   # 지연 import (xlsx_chunking → chunking 순환 방지)
        return chunk_xlsx(str(file_path), description=description)

    parser = _SECTION_PARSERS.get(suffix)
    if parser is None:
        # 업로드는 SUPPORTED_SUFFIXES가 앞에서 거르므로 도달하지 않는다. 도달했다면
        # 지원 안 되는 파일이 파서를 잘못 타고 있다는 뜻이라 조용히 넘기지 않는다.
        raise ValueError(f'지원하지 않는 형식: {suffix or "확장자 없음"}')

    out: list[ChunkData] = []
    for section in _pack_sections(parser(file_path)):
        cursor = 0
        for chunk in splitter.split_text(section.body):
            # 청크의 실제 시작 위치로 쪽을 고른다. 섹션 page 하나를 모든 청크에
            # 물려주면 여러 쪽에 걸친 섹션의 뒤쪽 청크가 앞 쪽을 가리킨다 (#137 결함 6).
            # 오프셋은 목록 번호 복원(결함 3)의 판정 근거이기도 하므로 손대기 전에 구한다.
            found = section.body.find(chunk, cursor)
            offset = found if found >= 0 else cursor
            cursor = max(cursor, offset)
            text = _strip_orphan_marker(_restore_leading_marker(section.body, offset, chunk))
            if not text.strip():
                continue
            out.append(ChunkData(text=text, heading_path=list(section.heading_path),
                                 page=_page_at(section, offset), chunk_index=len(out)))
    return out
