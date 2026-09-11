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

 - PDF : pdfplumber로 줄별 글자 크기를 보고 헤딩 판정. page도 보존 — 유일.
         투표 전에 인쇄 부산물을 빼고(_drop_furniture), 표는 markdown으로 조립하며
         (_pdf_tables), 기호만으로 된 줄은 헤딩 후보에서 뺀다. 셋 다 #137 실측 결함이다.
         page는 **청크마다 그 청크가 실제로 있는 쪽**이다 (섹션 시작 쪽이 아니다 — #137 결함 6).
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
from dataclasses import dataclass
from pathlib import Path

import pdfplumber
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from llama_index.core.node_parser import SentenceSplitter

_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$')
_DOCX_HEADING_RE = re.compile(r'^Heading (\d+)$')     # python-docx는 빌트인 헤딩을 영문명으로 준다
_MD_FENCE_RE = re.compile(r'^(`{3,}|~{3,})')          # 코드 펜스 — 안쪽 '#'은 헤딩이 아니다
# 섹션 묶음 예산(문자). 섹션 하나하나는 보통 100~150자라, 섹션=청크로 두면 top_k=5가
# 실어 나르는 컨텍스트가 옛 방식의 1/3로 줄어든다(실측 2211자→707자).
# 주의: 이 값이 상한으로 작동하는 일은 드물다 — _pack_sections의 경계 규칙(같은 대분류
# 아래 형제끼리만)이 먼저 걸려 실측 청크 중앙값은 200자대다. 실제 토큰 상한은 SentenceSplitter.
_PACK_CHARS = 500

# ── PDF 인쇄 부산물(page furniture) 판별 ──────────────────────────────────────
# 브라우저 "인쇄 → PDF"로 만든 인트라넷 문서는 상·하단 여백에 인쇄 시각과 원본 URL을
# 찍는다. 이 줄이 본문 크기 투표(_pdf_sections)에 참여하면 짧은 문서에서 최빈값을
# 빼앗아 본문 전체가 헤딩으로 승격되고, flush()의 '본문 없으면 섹션 없음' 가드와
# 겹쳐 내용이 소멸한다 (#137 결함 1 — 실측 최소 격차 13글자로 뒤집혔다).
# 위치와 형태를 **함께** 본다: 위치만 보면 여백에 걸친 정상 본문을 지우고,
# 형태만 보면 본문에 인용된 URL까지 지운다.
_FURNITURE_RE = re.compile(
    r'https?://'                                            # 원본 URL 푸터
    r'|^\d{1,4}\.\s*\d{1,2}\.\s*\d{1,2}\.\s*(?:오전|오후)'   # 인쇄 시각 헤더
    r'|^\d+\s*/\s*\d+$'                                    # 쪽 표기 '3/5'
)
# 실측(#137, 인쇄 PDF 9건): 인쇄 헤더 top 2.0% / URL 푸터 bottom 98.2%
#                           본문 최상단 9.0% / 본문 최하단 95.8% → 6%·96%로 가른다.
_FURNITURE_TOP_PCT = 6.0
_FURNITURE_BOTTOM_PCT = 96.0

# 기호·구두점만으로 이뤄진 줄은 헤딩 후보에서 뺀다. 제목이 '[대분류 | 소분류] 문서명'
# 꼴이면 한글과 대괄호가 서로 다른 폰트로 조판돼 베이스라인이 어긋나고(실측 3.0pt),
# pdfplumber가 '[ | ]' 조각을 별도 줄로 끊는다. 그 조각이 제목과 같은 크기라 같은 레벨
# 헤딩이 되고, del stack[level-1:]이 진짜 제목을 덮어썼다 (#137 결함 2, 인쇄 PDF 9건).
_SYMBOLS_ONLY_RE = re.compile(r'^[\W_]*$')

# 표 판별 — 실측 근거와 깨질 조건은 _pdf_tables docstring. 애매하면 평문으로 보낸다.
_TABLE_MIN_FILL_RATIO = 0.20       # 격자에서 채워진 셀의 비율

# 청크 꼬리에 번호만 남는 것을 되돌릴 때 쓴다 (#137 결함 3). 정의점은 _merge_orphan_markers.
# 꼬리/머리 판정 공용. 번호 뒤 공백은 물론 개행 하나까지 허용한다 — 원문에서 번호와
# 항목이 같은 줄일 때(공백)와 번호가 독립 줄일 때(개행)가 둘 다 나온다.
_ORPHAN_MARKER_RE = re.compile(r'(?:^|\n)[ \t]*(\d{1,2}\.|\(\d{1,2}\))[ \t]*\n?$')

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


@dataclass
class _Line:
    """PDF 줄 하나. 헤딩 판정에 크기가, 부산물 제거에 위치가 필요해 함께 나른다.

    좌표를 버리면 하류 어느 단계도 복구할 수 없다 — 인쇄 부산물 판별(위치)과
    표 영역 판별(bbox)이 둘 다 이 단계에만 있는 정보를 쓴다.
    """
    page: int
    text: str
    size: float                 # 줄의 최대 글자 크기 (헤딩 판정용)
    top_pct: float              # 페이지 높이 대비 % — 부산물 판별
    bottom_pct: float
    is_table: bool = False      # 표에서 조립된 줄 — 크기 투표·헤딩 판정에서 제외


def to_markdown_table(header: list[str], body: list[list[str]]) -> str:
    """헤더 + 데이터행 → markdown 표. 헤더가 청크에 포함돼 컬럼 의미가 보존된다.

    PDF·XLSX 표 직렬화의 단일 정의점 (rag/xlsx_chunking이 이 이름을 import한다).
    빈 셀은 자리를 지킨다 — 지우면 뒷 열이 앞으로 당겨져 값이 다른 열로 읽힌다.
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
    """
    with pdfplumber.open(str(file_path)) as pdf:
        return [(i, page.extract_text() or '') for i, page in enumerate(pdf.pages, start=1)]


def _pdf_tables(page) -> list[tuple[float, float, str]]:
    """페이지의 표를 (top, bottom, markdown) 목록으로. 표가 아닌 격자는 걸러낸다.

    extract_tables()는 빈 셀을 ''로, 병합 셀을 None으로 **정확히** 준다 — 평탄화
    텍스트(extract_text_lines)가 버리는 열 위치 정보가 여기 살아 있다. 빈 셀이
    사라지면 뒷 열이 앞으로 당겨져 'O' 표시가 다른 열의 값으로 읽힌다
    (#137 결함 5 — 개인부담 O가 회사지원 O로 읽히는 형태. 유실이 0이라
    커버리지 지표에는 안 잡히고 답만 틀린다). docx·xlsx 파서가 같은 이유로 자리를
    지키고 있었고 PDF만 이 규약에서 빠져 있었다.

    **확실한 표만 통과시킨다.** find_tables()의 기본 전략은 그려진 선을 격자로 보는데,
    조항형 규정의 본문 테두리·들여쓰기 선이 격자로 오인돼 **산문이 표로 둔갑한다**
    (실측: 노사협의회 규정의 조항 19줄이 `| 문장 |  |  |` 꼴로 감싸이고 그 과정에서
    조항 번호 ③이 셀 경계로 잘려 나갔다). 표로 못 알아보는 손실보다 산문을 표로
    왜곡하는 손실이 크므로, 애매하면 평문 경로로 보낸다.

    ── 판별 기준 두 개 (실문서 21건의 격자 54개를 원본 육안 판정과 대조해 잡았다.
       정답: 진짜 표 30개 / 가짜 격자 24개. 두 조건으로 54/54 분리된다.)

    1. **값이 2개 이상 채워진 행이 하나 이상.** 임계가 아니라 정의다 — 표의 행은
       여러 필드를 가진 레코드이고, 그런 행이 하나도 없으면 테두리 쳐진 산문이다.
       이 조건 하나가 가짜 24개 중 20개를 걸러낸다.
    2. **채워진 셀 비율 >= 20%.** 남은 가짜 4개는 pdfplumber가 14~28열을 지어낸
       경우로, 수백 개 셀 중 수십 개만 채워진다(밀도 8~11%). 진짜 표의 최저 밀도는
       25%라 간극이 2.3배다. "셀 다섯 중 하나는 채워야 표"라는 뜻.

    ── 깨질 조건: **빈 셀이 아주 많은 진짜 표**(대각선 체크표 등)는 거부된다.
       그때는 밀도를 낮추거나 열 수 대비 지표로 바꿀 것.

    ── 기각한 대안 (같은 실측에서):
       - `lines_strict` 전략(rect 변을 경계로 안 쓰는 것): 오검출 원인을 정확히
         겨냥하지만 이 문서들은 진짜 표도 rect로 그려서 **표 0개 검출**이 된다.
       - 표 영역 원문 대비 셀 텍스트 보존율: 판별력이 없다 — 가짜도 99~100%다.
       - docling(TableFormer) 재도입: 품질·라이선스(MIT)·CPU 동작은 맞지만
         **상주 메모리 약 6.2GB**라 워커 장비에 과하다(2026-09-10 판단).
         이 기준은 그래서 임시 해법이고, 파서 교체는 별도 이슈로 남긴다.
    """
    out: list[tuple[float, float, str]] = []
    for table in page.find_tables():
        rows = [[('' if cell is None else str(cell)).replace('\n', ' ').strip()
                 for cell in row]
                for row in table.extract()]
        rows = [row for row in rows if any(row)]     # 완전 빈 행만 스킵 (docx·xlsx와 동일)
        if len(rows) < 2 or len(rows[0]) < 2:
            continue
        if not any(sum(1 for cell in row if cell) >= 2 for row in rows):
            continue                                 # 레코드가 없다 → 산문이다
        cells = len(rows) * len(rows[0])
        filled = sum(1 for row in rows for cell in row if cell)
        if filled / cells < _TABLE_MIN_FILL_RATIO:
            continue                                 # 격자를 거의 안 채웠다 → 오검출이다
        out.append((table.bbox[1], table.bbox[3], to_markdown_table(rows[0], rows[1:])))
    return out


def _pdf_lines(file_path: str | Path) -> list[_Line]:
    """PDF를 _Line 목록으로 — 헤딩 판정(크기)·부산물 제거(위치)·표 보존에 필요한 것만.

    extract_text()는 크기·좌표를 버리므로 extract_text_lines()로 줄별 char를 본다.
    표 영역은 markdown으로 따로 조립하고(_pdf_tables), 그 영역에 걸친 평문 줄은
    빼서 같은 내용이 두 번 들어가지 않게 한다.
    """
    out: list[_Line] = []
    with pdfplumber.open(str(file_path)) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            height = page.height or 1
            tables = _pdf_tables(page)
            for line in page.extract_text_lines():
                text = (line.get('text') or '').strip()
                if not text:
                    continue
                middle = (line['top'] + line['bottom']) / 2
                if any(top <= middle <= bottom for top, bottom, _ in tables):
                    continue                     # 표 안 줄은 markdown 쪽이 담당한다
                sizes = [c['size'] for c in line.get('chars', []) if c.get('size')]
                out.append(_Line(page_no, text, round(max(sizes), 1) if sizes else 0.0,
                                 line['top'] / height * 100,
                                 line['bottom'] / height * 100))
            for top, bottom, markdown in tables:
                out.append(_Line(page_no, markdown, 0.0, top / height * 100,
                                 bottom / height * 100, is_table=True))
    out.sort(key=lambda line: (line.page, line.top_pct))
    return out


def _drop_furniture(lines: list[_Line]) -> list[_Line]:
    """상·하단 여백의 인쇄 부산물(인쇄 시각·원본 URL·쪽 표기)을 뺀다 (#137 결함 1).

    여러 쪽에서 같은 자리에 반복되는 줄도 부산물로 본다 — 1쪽짜리 문서는 반복이
    성립하지 않으므로 형태(_FURNITURE_RE)가 그 경우를 받친다.
    """
    pages_of: dict[str, set[int]] = {}
    for line in lines:
        pages_of.setdefault(line.text, set()).add(line.page)
    repeated = {text for text, pages in pages_of.items() if len(pages) >= 2}

    def is_furniture(line: _Line) -> bool:
        if line.is_table:
            return False
        in_margin = (line.top_pct < _FURNITURE_TOP_PCT
                     or line.bottom_pct > _FURNITURE_BOTTOM_PCT)
        return in_margin and (bool(_FURNITURE_RE.search(line.text))
                              or line.text in repeated)

    # 기호·구두점만 남은 줄은 헤딩 후보에서 빼는 것으로 끝내면 본문으로 떨어져
    # 임베딩·BM25·프롬프트에 잡음으로 들어간다. 의미가 없으므로 여기서 버린다.
    return [line for line in lines
            if not is_furniture(line)
            and (line.is_table or not _SYMBOLS_ONLY_RE.match(line.text))]


def _pdf_sections(file_path: str | Path) -> list[_Section]:
    """PDF를 (heading_path, 시작 page, 본문, 줄별 page) 섹션 목록으로 — 글자 크기로 헤딩 판정.

    DOCX와 달리 PDF엔 구조 태그가 없어 조판에 기대는 휴리스틱이다:
      본문 크기 = 가장 많이 쓰인 글자 크기, 그보다 **큰** 줄을 헤딩으로 본다.
      크기 종류를 큰 순으로 정렬해 1,2,3… 레벨로 매긴다.
    크기 차이가 없는 문서(전부 같은 크기, 스캔본 등)는 헤딩 0개 → heading_path=[]로
    폴백한다 (조판이 달라도 손해는 없게).

    투표 전에 인쇄 부산물을 빼고(_drop_furniture), 표는 본문으로 고정하며(is_table),
    기호만으로 된 줄은 헤딩 후보에서 뺀다 — 셋 다 #137에서 실문서로 확인된 결함이다.
    """
    lines = _drop_furniture(_pdf_lines(file_path))
    if not lines:
        return []

    # 본문 크기 = 글자 수 기준 최빈 크기 (줄 수가 아니라 분량 기준이라 제목에 안 휘둘린다).
    # 표 줄은 세지 않는다 — 표는 정의상 본문이고, 표 안의 큰 글자가 투표에 끼면
    # 산문 본문 크기를 밀어낸다.
    weight: dict[float, int] = {}
    for line in lines:
        if not line.is_table:
            weight[line.size] = weight.get(line.size, 0) + len(line.text)
    if not weight:                       # 표만 있는 문서 — 헤딩 없이 본문으로 둔다
        body = '\n'.join(line.text for line in lines)
        pages = [line.page for line in lines for _ in line.text.split('\n')]
        return [_Section([], lines[0].page, body, pages)]
    body_size = max(weight, key=lambda size: weight[size])

    # 본문보다 큰 크기들 → 큰 순으로 레벨 부여
    level_of = {size: level for level, size in
                enumerate(sorted((s for s in weight if s > body_size), reverse=True), start=1)}

    sections: list[_Section] = []
    stack: list[str] = []
    buf: list[str] = []
    buf_pages: list[int | None] = []
    orphans: list[str] = []      # 본문 없이 교체된 헤딩 — 다음 섹션 본문에 되살린다
    start_page = lines[0].page

    def add(text: str, page: int) -> None:
        """본문 줄 추가. 표 markdown은 여러 줄이라 page도 그 줄 수만큼 채운다
        (line_pages와 body.split('\n')의 1:1 대응이 깨지면 결함 6 수정이 무의미해진다)."""
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

    def is_heading(line: _Line) -> bool:
        return (not line.is_table
                and line.size in level_of
                and not _SYMBOLS_ONLY_RE.match(line.text))

    for line in lines:
        if is_heading(line):
            level = level_of[line.size]
            produced = flush()
            removed = stack[level - 1:]
            # 같은 레벨 헤딩을 덮어쓰기 전에, 본문을 한 번도 못 만든 것은 붙잡아 둔다.
            # del stack[level-1:]이 직전 동일 레벨 헤딩의 자리를 지우기 때문에
            # '본문 없는 헤딩 + 같은 레벨 헤딩' 연속열은 흔적 없이 소멸했다 (#137).
            # (부모→자식처럼 레벨이 깊어지는 경우는 스택 뒤에 붙으므로 원래 안전하다 —
            #  _docx_sections docstring의 "정보 손실은 없다"는 그 경우에만 맞다.)
            if not produced and removed:
                orphans.extend(removed)
            del stack[level - 1:]
            stack.append(line.text)
        else:
            # 섹션의 page는 '첫 본문 줄'의 페이지다 — 헤딩이 페이지 끝에 걸리면
            # 본문은 다음 쪽에서 시작하고, 인용은 본문이 있는 쪽을 가리켜야 맞다.
            if not buf:
                start_page = line.page
                for orphan in orphans:
                    add(orphan, line.page)
                orphans.clear()
            add(line.text, line.page)
    if orphans and not buf:            # 문서 끝에 남은 것도 버리지 않는다
        for orphan in orphans:
            add(orphan, lines[-1].page)
    flush()
    return sections


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
