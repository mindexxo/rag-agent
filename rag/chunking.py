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

 - PDF : **docling**(레이아웃 ML + TableFormer + 원격 한국어 OCR #168)이 기본 — #143. 인쇄 부산물·제목 분할·표 빈 셀을
         모델이 구조적으로 처리하고, 헤딩 층만 한국 규정 번호 체계('제 N 장'/'제 N 조')로 복원한다.
         docling이 실패하면 색인은 failed로 끝난다 — 저품질 비상 경로로 조용히 바꾸지 않는다(#272).
         page는 **청크마다 실제 쪽**. 채팅 첨부(extract_text)는 pdfplumber — 사용자 대기 경로라 콜드 스타트를 안 태운다.
 - DOCX: **docling**(SimplePipeline — 레이아웃·OCR 없음)이 기본(#220). Heading 스타일 레벨 그대로 섹션 분할,
         스타일이 없으면 조문 패턴 폴백(#205). 그림은 VLM 캡션, 표 셀 안 중첩 표·워드 자동 번호까지 읽는다.
         채팅 첨부만 python-docx(그림·중첩 표·자동 번호를 잃는다 — 첨부는 통째로 넣는 평문이라 감수).
 - MD  : **docling**(MD 백엔드, #267) — 헤딩(밑줄식 포함)·표를 문법대로. 표 칸 '\\|'·'<br>', 링크 주소, 인라인 서식
         조각을 보정한다(_md_sections). 채팅 첨부는 원문 그대로.
 - TXT : 평문(인코딩 감지 utf-8→cp949). 헤딩 구조가 없어 heading_path는 빈다.
 - XLSX: rag/xlsx_chunking. 행 묶음 청크 + 시트 완성본(#222)이라 이 파이프라인 밖 — 크기 상한(내용의 양) 초과면 업로드를 거절한다.
         읽기는 docling(시트 위 그림 캡션·병합 셀), 행 세기는 openpyxl(#220).

 공통: SentenceSplitter(chunk_size=512, chunk_overlap=50)로 크기 cap.
   - 큰 섹션이 비대해져 검색 정확도 떨어지는 것 방지 + LLM 컨텍스트 부담 완화
   - overlap 50으로 경계 의미 단절 완화

 결과 ChunkData: text / heading_path / page / chunk_index / meta.
 묶음 2개 이상으로 나뉜 표(PDF·DOCX)의 묶음은 meta에 {"is_table": True, "table_key": 문서 안 표 순번, "table_text": 표 전체}를
 싣는다(표 전체 = 앞 설명 + 표 + 아래 주석, #268) — 근거 준비(rag/retriever.prepare_evidence)가 걸린 묶음을 '걸린 묶음 + 표 전체'로
 바꾼다(#257·#268 결정 4, 엑셀 #222와 같은 계약).
 표 하나가 TABLE_MAX_TOKENS를 넘으면 문서를 색인하지 않는다(TableTooLargeError).
 (LlamaIndex 의존성은 이 모듈 안에 격리 — 호출부는 ChunkData 도메인 타입만 본다.)
 """

import re
import logging
import os
import pickle
import socket
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import pdfplumber
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from llama_index.core.node_parser import SentenceSplitter

from config import settings
from rag.identifiers import is_heading_line, split_heading_line
from rag.tokens import CONSERVATIVE_CHARS_PER_TOKEN, estimate_tokens_conservative

logger = logging.getLogger(__name__)

_DOCX_HEADING_RE = re.compile(r'^Heading (\d+)$')     # python-docx는 빌트인 헤딩을 영문명으로 준다
# 섹션 묶음 예산(문자). 섹션 하나하나는 보통 100~150자라, 섹션=청크로 두면 top_k=5가
# 실어 나르는 컨텍스트가 옛 방식의 1/3로 줄어든다(실측 2211자→707자).
# 주의: 이 값이 상한으로 작동하는 일은 드물다 — _pack_sections의 경계 규칙(같은 대분류
# 아래 형제끼리만)이 먼저 걸려 실측 청크 중앙값은 200자대다. 실제 토큰 상한은 SentenceSplitter.
_PACK_CHARS = 500

# 청크 꼬리에 번호만 남는 것을 되돌릴 때 쓴다 (#137 결함 3). 정의점은 _strip_orphan_marker /
# _restore_leading_marker. 번호 뒤 공백은 물론 개행 하나까지 허용한다 — 원문에서 번호와 항목이
# 같은 줄일 때(공백)와 번호가 독립 줄일 때(개행)가 둘 다 나온다.
_ORPHAN_MARKER_RE = re.compile(r'(?:^|\n)[ \t]*(\d{1,2}\.|\(\d{1,2}\))[ \t]*\n?$')

# OCR 상수 (#168) — 서버(serving/triton-ocr/model_repository/ocr/1/model.py)와 짝이다. 설정으로 빼지 않는다:
# 값이 바뀌면 서버 모델도 같이 바뀌어야 하고, 둘 다 실측으로 고정한 것이라 환경별로 다를 이유가 없다.
_OCR_LANG = 'korean'   # docling 기본은 ['chinese'] — 중국어 모델로 한글 표를 읽으면 '3년'→'34' 같은 틀린 값이
                       # 채워진다(#144의 'OCR 기각'은 이 상태의 측정이었다 — #168에서 정정)
_OCR_SCALE = 3.0       # 크롭 렌더 배수(72dpi 기준). 인프로세스 RapidOcrOptions 기본값과 같다. **KserveV2OcrOptions의
                       # 기본값은 2.0**이라 명시하지 않으면 인프로세스 실측과 결과가 달라진다

# 한국 규정 문서의 장(章) 제목 — '제 3 장 복 무', '제3장 총칙'. docling은 장과 조를 같은 레벨로
# 내놓으므로(#143 실측: 26쪽 141개 헤딩 전부 level 1) 이 패턴으로 층을 복원한다. 마크다운의 '#'처럼
# 문서가 스스로 선언하는 층이라 폰트 크기 휴리스틱과 성격이 다르다 — 코퍼스가 아니라 도메인 관례다.
_CHAPTER_RE = re.compile(r'^제\s*\d+\s*장\b')

# docling 그림 요소의 자리표시. #137 결함 4 당시 'docling 표 모드·OCR·VLM 셋 다 실측 실패'로 적었는데,
# 그 OCR은 중국어 모델이었다 — 한국어 RapidOCR은 표 이미지의 셀을 읽는다(#168). 그림은 VLM 캡션(#162)이
# 읽고, 캡션이 없을 때만 이 자리표시가 남는다. 지우지 않고 남겨 "여기 도표가 있었다"를 알리고,
# **캡션이 붙으면 이 자리를 캡션이 대체한다**(#162). #223부터 VLM 실패는 자리표시가 아니라 문서 재시도다
# (_PictureGuard) — 그래서 남은 자리표시는 **캡션 대상이 아니었던 그림**(면적 하한 picture_area_threshold
# 미만의 작은 그림, XLSX의 #249)뿐이다(count_picture_placeholders).
_PICTURE_PLACEHOLDER = '<!-- image -->'

# 그림 캡션 프롬프트(#162·#204·#247). 하나의 프롬프트가 모든 문서의 모든 그림에 적용된다.
# 첫 여섯 줄은 #162·#204의 형태 중립 문구 그대로다 — 흐름도를 전제한 문구는 표·사진에 해롭고 불릿 귀속을 한 칸 밀었다
# (2026-09-14). 그래서 #247(2026-10-18)은 유형별 지시를 **'흐름도라면 / 조직도라면 / 차트라면' 조건부로만** 더하고, 박스 아래
# 설명을 박스에 붙이라는 지시는 넣지 않는다 — 넣어 보니(V8) 첫 박스에 설명이 없는 채용 흐름도에서 설명을 한 칸씩 밀어 붙여
# '모집공고: 면접 대상자 선정' 같은 확신 있는 오답을 냈다(#162와 같은 현상). 7B는 박스-설명 공간 귀속을 안정적으로 못 한다.
# #247 실측(eval/doc_fidelity — prompt_ab·holdout, 결과는 저장소 밖):
#   - 현행 여섯 줄에서 '한국어로'·'위치대로 나열'을 빼면 효과가 없거나 나빠졌다(제거 실험 V1·V2·V4) — 그대로 둔다
#   - 합성 그림 16장(정답 자동 채점): 항목-값 짝 25→32/33, 흐름도 화살표 0→14/15, 빈 칸 환각 1→0, 글자 77→86/91
#   - 튜닝에 안 쓴 홀드아웃 12장: 꺾은선 월 라벨 1→7/7, 작은 표 이미지 칸 8→12/19, 조직도·분류 트리 연결 0→11/11, 나빠짐 없음
#   - 사람 블라인드 비교(실문서 16 + 합성 15): 개선 19 · 악화 2 · 비슷 10
# 남는 한계(기록): 계열 중간의 빈 칸에 이웃 값을 지어내는 경우(홀드아웃 K04), 작은 숫자 오독(-6.6→-5.6), 빽빽한 화면 캡처의 요약·누락.
_PICTURE_CAPTION_PROMPT = (
    '이미지의 내용을 한국어로 옮겨라.\n'
    '- 이미지 안의 글자를 하나도 빠뜨리지 말고 원문 그대로 옮긴다. 고쳐 쓰거나 요약하지 않는다.\n'
    '- 요소들 사이에 관계가 있으면 그 관계를 함께 적는다.\n'
    '- 관계가 분명하지 않으면 보이는 위치대로 나열만 한다. 관계가 없다는 말은 쓰지 않는다.\n'
    '- 흐름도·절차도라면 화살표 하나를 한 줄로 "<출발 박스 이름> → <도착 박스 이름>" 형식으로 적고, 화살표 위 글자(예/아니오 등)가 '
    '있으면 줄 끝에 괄호로 붙인다.\n'
    '- 조직도·계층도라면 위에서 아래로, 연결선 하나를 한 줄로 "<상위 이름> → <하위 이름>" 형식으로 적는다.\n'
    '- 차트라면 제목·축 이름·단위·범례를 먼저 적고, 값은 "가로축 항목 (계열): 값" 형식으로 항목마다 한 줄씩 적는다.\n'
    '- 이미지에 없는 내용은 쓰지 않는다.\n'
    '- 값이 비어 있는 칸·막대·점은 지어내지 말고 "값 없음"으로 적는다.\n'
    '- 머리말·맺음말·질문·제안 없이, 첫 줄부터 이미지 안의 글자로 시작한다.'
)
# 셋째 불릿의 뒷문장과 마지막 불릿은 #204 — 7B VLM이 캡션 앞뒤에 "이미지의 내용은 다음과 같습니다:" /
# "…정확히 기록했습니다. 수정이 필요하시면 알려주세요" / "A 섹션은 B 섹션과 관련이 없습니다"를 붙여 본문으로
# 색인됐다(타임레포트 12캡션 중 머리말 6·맺음말 1·관계부재 1, 2026-10-08). 실측: 금지 문구를 넣자 맺음말·관계부재는
# 0/12이 됐고 머리말은 5/12 남았다 — 금지할 문구를 프롬프트에 **인용하면 오히려 유도**하는 것으로 보여 인용 없이
# 적는다. 문구 목록 후처리는 두지 않는다(모델 말버릇에 맞춘 예외 목록이 자라는 구조 — 리뷰에서 기각). 머리말이
# 남더라도 캡션당 한 줄 15자라 검색·생성에 무해하다.

@dataclass
class ChunkData:
    """청킹 단계 중간 결과 (임베딩 / DB insert 전)."""
    text: str                   # 청크 본문
    heading_path: list[str]     # 예: ["3. 배송지연 보상", "3.2 지급 기준"]
    page: int | None            # PDF는 페이지 번호, DOCX는 None
    chunk_index: int            # 문서 내 순서 (0부터)
    meta: dict | None = None    # 청크 metadata — 표 묶음은 {"is_table", "table_key", "table_text"}(#257), 묶음 헤딩은 {"headings"}(#238)


@dataclass
class _Section:
    """형식별 파서 → _pack_sections 사이의 내부 계약. 이 모듈 밖으로 나가지 않는다.

    파서 넷(_pdf/_docx/_md/_txt_sections)이 전부 이 형태를 내놓기 때문에 그 뒤 단계
    (묶기 → 크기 분할 → ChunkData)를 형식과 무관하게 한 번만 쓴다. 표는 blocks에 _TableBlock으로 운반된다(#252).
    """
    heading_path: list[str]
    page: int | None            # PDF만 채운다 — 나머지는 파일에 페이지 개념이 없다
    body: str
    # body를 '\n'으로 나눈 줄과 1:1로 대응하는 쪽 번호. PDF만 채운다.
    # page 하나로는 여러 쪽에 걸친 섹션을 표현할 수 없어 뒤쪽 청크가 앞 쪽 번호를
    # 물려받았다 (#137 결함 6 — 실측 193청크 중 95개가 틀린 쪽을 가리켰다).
    line_pages: list[int | None] | None = None
    # 묶음(_pack_sections)이 본문 줄로 내려보낸 하위 헤딩 — 원문 그대로. 어느 줄이 헤딩이었는지를
    # 여기 기록해 두어야 색인이 식별자('11조')를 **추정 없이** 뽑고(rag/identifiers, #238 결정 15),
    # 분할이 헤딩을 본문과 같은 조각에 둘 수 있다(_split_section). 묶지 않은 섹션은 비어 있다.
    headings: list[str] = field(default_factory=list)
    # 본문을 이루는 블록 목록(#252 결정 1 = b′): 글 블록(str) 또는 표 블록(_TableBlock). body·line_pages는 이 목록에서
    # 파생된 값이다(파서가 둘 다 채운다 — body 문자열을 없애는 일은 #230). None이면(txt·md) 본문 전체가
    # 글 블록 하나다. 크기 분할(chunk_file)은 이 목록을 보고 표만 행 단위로 나눈다 — 표를 글줄에서 추측하지 않는다.
    blocks: list['str | _TableBlock'] | None = None


class _TableBlock(NamedTuple):
    """표 하나 — 첫 행이 헤더인 행 목록. 파서(PDF·DOCX)가 표를 글로 바꾸지 않고 이대로 chunk_file까지 운반한다(#252).

    글로 바꾸는 시점은 둘뿐이다: 섹션 body(검색·쪽 매핑·테스트가 보는 파생 문자열)와 행 묶음 청크(batch_table_rows).
    둘 다 to_markdown_table을 쓰므로 같은 표는 어디서든 같은 글이다.
    row_pages — 행마다의 쪽 번호(헤더 포함, PDF만). 쪽을 넘긴 표를 이으면(#252 E) 뒷 쪽 행이 앞 쪽 번호를 물려받는 #137
    결함 6이 되살아나므로 행 단위로 들고 다닌다(리뷰 지적). None이면 블록 전체가 한 쪽.
    """
    rows: list[list[str]]
    row_pages: list[int | None] | None = None

    def render(self) -> str:
        return to_markdown_table(self.rows[0], self.rows[1:])

    def line_pages(self, page: int | None) -> list[int | None]:
        """렌더한 줄(헤더·구분선·행)마다의 쪽 번호 — 구분선은 헤더 쪽을 따른다."""
        rp = self.row_pages or [page] * len(self.rows)
        return [rp[0], rp[0]] + list(rp[1:])


def _blocks_of(section: '_Section') -> list['str | _TableBlock']:
    return section.blocks if section.blocks is not None else [section.body]


def _join_blocks(blocks: list['str | _TableBlock']) -> list['str | _TableBlock']:
    """이웃한 글 블록을 '\n'으로 합친다 — 글 블록 경계는 의미가 없고(분할기가 다시 나눈다) 표 블록 경계만 의미가 있다."""
    out: list[str | _TableBlock] = []
    for b in blocks:
        if isinstance(b, str) and out and isinstance(out[-1], str):
            out[-1] = f'{out[-1]}\n{b}'
        else:
            out.append(b)
    return out


# (page, text, level) — PDF·DOCX·MD 경로가 _build_sections에 흘리는 공통 단위(#233).
# level이 None이면 본문 줄, 정수면 그 레벨의 헤딩. text는 여러 줄일 수 있고, 표는 _TableBlock으로 온다(#252 — 행 목록 그대로 운반).
_Item = tuple[int | None, 'str | _TableBlock', int | None]


# 표 행 묶음 하나의 크기(#222 XLSX_BATCH_TOKENS가 원본 — 측정 근거 없음, 기획 메모 "20~30행씩 + 헤더 반복" 규모). #252부터
# PDF·DOCX 표도 같은 값으로 나눈다. 글자 수 환산은 보수적 계수(rag/tokens).
TABLE_BATCH_TOKENS = 1000
TABLE_BATCH_CHARS = int(TABLE_BATCH_TOKENS * CONSERVATIVE_CHARS_PER_TOKEN)

# 표 하나당 생성 모델 토큰 상한 — **형식 공통**(#257 결정 2: 엑셀·PDF·DOCX가 같은 값). 근거 예산은 표를 통째로 넣거나
# 통째로 빼므로(rag/retriever.prepare_evidence) 상한을 넘는 표는 일부만 보여 줄 길이 없다 — 일부만 보여 주면 모델이 나머지
# 행을 지어낸다. 그래서 받지 않는다: 엑셀은 업로드 거절(rag/xlsx_chunking), PDF·DOCX는 색인 실패(TableTooLargeError).
# 값의 근거는 컨텍스트가 아니라 **답의 정확도**다(#222 실측 — rag/xlsx_chunking 모듈 docstring): 300행 표(5,954토큰)에서
# 단가를 옮겨 적다 틀렸다. 컨텍스트가 정하는 물리 상한(근거 예산 EVIDENCE_BUDGET_TOKENS 22,000)보다 작아야 한다 —
# 생성 모델을 바꾸면 이 값을 다시 잰다.
TABLE_MAX_TOKENS = 6000


def batch_table_rows(header: list[str], body: list[list[str]], budget_chars: int = TABLE_BATCH_CHARS) -> list[str]:
    """표(헤더 + 행 목록) → 행 묶음 markdown들. 묶음마다 헤더 전체와 구분선을 반복한다. **형식 공통 정의점**(#252 — #222의
    xlsx 행 묶음을 옮겨 옴; rag/xlsx_chunking이 이 이름을 import한다).

    **글로 바꾸기 전에 행 목록에서 나눈다**(#222 리뷰) — 렌더한 글을 줄 단위로 다시 해석하던 첫 구현은 칸 안 줄바꿈에서
    헤더가 깨지고(헤더 칸 "구분⏎(원)") 행이 중간에서 잘렸다(데이터 칸 "첫줄⏎| 둘째줄", 8묶음 중 6개). PDF·DOCX는 #252 전까지
    표가 글줄로 SentenceSplitter를 타서 45행 표의 경계 행이 `30,` / `148원 | 5 |`로 쪼개졌다(#246 실측). 행 목록 단계에서
    나누면 경계가 항상 행과 행 사이다. 묶음 크기는 렌더한 길이로 잰다(행 하나를 따로 렌더해 더한다 — 행마다 독립적으로
    렌더되므로 정확하다). 행 하나가 예산보다 길면 그 행 혼자 한 묶음이다(자르지 않는다). 데이터 행이 없으면 헤더만.
    """
    head_len = len(to_markdown_table(header, []))
    batches, cur, size = [], [], head_len
    for row in body:
        row_len = len(to_markdown_table(header, [row])) - head_len      # '\n' + 그 행 한 줄(들)
        if cur and size + row_len > budget_chars:
            batches.append(to_markdown_table(header, cur))
            cur, size = [], head_len
        cur.append(row)
        size += row_len
    batches.append(to_markdown_table(header, cur))
    return batches


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


# ── 섹션 조립 — PDF·DOCX·MD 공통 후단 ──────────────────────────────────────────────

def _build_sections(items: list[_Item]) -> list[_Section]:
    """(page, text, level) 흐름을 헤딩 스택으로 _Section 목록에 접는다. PDF·DOCX·MD가 이 함수 하나로 조립한다(#233).

    스택 규약은 `del stack[level-1:]` + append다. 그 위에 #137·#233에서 확인된 것을 더한다:
      - **줄별 쪽 번호(line_pages)** — 섹션이 여러 쪽에 걸치면 뒤쪽 청크가 앞 쪽을 물려받았다
        (#137 결함 6). 표 markdown처럼 text가 여러 줄이면 그 줄 수만큼 같은 쪽을 채워 1:1을 지킨다.
      - **본문 없는 헤딩 보존(고아)** — 같은 레벨 헤딩이 연달아 오면 `del stack[level-1:]`이 직전 것을 흔적 없이 지웠다
        (#137 결함 1·2). 본문을 한 번도 못 만든 헤딩은 붙잡아 다음 섹션 본문 앞에 되살린다(문서 순서). 실제로 생기는 고아는
        거의 전부 docling PDF가 장 없는 문서에서 '2. 반품 규정'과 '2.1 신청 기한'을 같은 레벨로 잡은 상위 제목이고
        (#233 실측: 모의 123건 중 고아 82건, 81건이 이 꼴, 삭제 조 0건), 그때는 다음 본문 앞이 맞는 자리다(상위 맥락 유지).
        알려진 한계: 본문 없는 '제3조(삭제)'도 다음 조 본문 앞에 붙어 경로는 제4조가 된다 — "제3조" 질문의 구간 오리기
        (rag/identifiers.cut_identifier_span)가 제4조 본문까지 넘긴다. 배치를 바꾸면 위 상위 제목 81건이 함께 흔들려
        고치지 않았다(#233 설계 9-재).
      - **이미 경로로 나간 헤딩은 고아가 아니다**(#233) — 예전엔 '제2조(삭제)' 뒤에 다음 장이 오면 앞 조의 경로로 이미 나간
        '제1장'까지 고아로 잡아 다음 장 첫 본문에 다시 붙였다. 스택은 섹션을 만들 때 통째로 경로가 되므로 "나간 것"은 항상
        스택의 앞부분이다 — 그 길이(emitted) 하나로 가른다.
      - **문서 끝 헤딩 보존**(#233 설계 1) — 문서가 본문 없는 헤딩으로 끝나면('제7조(삭제)') 그 헤딩을 본문으로 남긴다. 경로는
        이미 나간 앞부분이라 같은 글이 경로와 본문에 두 번 들어가지 않는다.
    (부모→자식처럼 레벨이 깊어지는 경우는 스택 뒤에 붙으므로 원래 안전하다.)
    본문이 하나도 없는 문서(헤딩만 있는 빈 양식)는 섹션 0개 — 업로드가 "추출된 텍스트 없음"으로 실패한다(rag/documents, 결정 3).
    """
    sections: list[_Section] = []
    stack: list[str] = []
    stack_pos: list[int] = []         # stack 원소마다 items 속 위치 — 고아를 되살릴 때 문서 순서로 놓으려고
    orphans: list[tuple[int, str]] = []   # (items 속 위치, 헤딩)
    emitted = 0                        # stack 앞에서 몇 개가 이미 섹션 경로로 나갔나 — 그 헤딩들은 고아가 될 수 없다
    buf: list[str] = []
    buf_pages: list[int | None] = []
    buf_blocks: list[str | _TableBlock] = []
    start_page: int | None = items[0][0] if items else None
    last_page: int | None = start_page

    def add(text: 'str | _TableBlock', page: int | None) -> None:
        if isinstance(text, _TableBlock):
            buf.append(text.render())
            buf_pages.extend(text.line_pages(page))
        else:
            buf.append(text)
            buf_pages.extend([page] * (text.count('\n') + 1))
        buf_blocks.append(text)

    def flush(path: list[str]) -> None:
        """본문이 있으면 섹션을 만든다. 본문 없는 헤딩은 섹션이 되지 않는다."""
        nonlocal emitted
        if not buf:
            return
        sections.append(_Section(list(path), start_page, '\n'.join(buf), list(buf_pages), blocks=_join_blocks(buf_blocks)))
        emitted = len(path)
        buf.clear()
        buf_pages.clear()
        buf_blocks.clear()

    for pos, (page, text, level) in enumerate(items):
        if page is not None:
            last_page = page
        if level is not None:
            flush(stack)
            cut = max(emitted, level - 1)                       # 걷어낼 헤딩 중 경로로 한 번도 안 나간 것
            orphans.extend(zip(stack_pos[cut:], stack[cut:]))
            emitted = min(emitted, level - 1)
            del stack[level - 1:], stack_pos[level - 1:]
            stack.append(text)
            stack_pos.append(pos)
        else:
            # 섹션의 page는 '첫 본문 줄'의 페이지다 — 헤딩이 페이지 끝에 걸리면
            # 본문은 다음 쪽에서 시작하고, 인용은 본문이 있는 쪽을 가리켜야 맞다.
            if not buf:
                start_page = page
                for _, orphan in sorted(orphans):   # 빠진 순서가 아니라 문서 순서 — 상위가 하위보다 늦게 빠질 수 있다
                    add(orphan, page)
                orphans.clear()
            add(text, page)
    if buf:
        flush(stack)
    elif sections:                     # 문서가 본문 없는 헤딩으로 끝났다 — 버리지 않고 본문으로(경로는 이미 나간 앞부분)
        start_page = last_page
        for _, heading in sorted([*orphans, *zip(stack_pos[emitted:], stack[emitted:])]):
            add(heading, last_page)
        flush(stack[:emitted])
    return sections


# ── PDF 경로 ① docling — 기본 ────────────────────────────────────────────────────

_DOCLING_LOCK = threading.Lock()
_docling_converter = None


class TableTooLargeError(ValueError):
    """PDF·DOCX·MD 표 하나가 TABLE_MAX_TOKENS를 넘는다 — **결정적** 실패(#257 결정 2·3, MD는 #267부터).

    ValueError라 outbox.is_transient가 결정적으로 분류해 문서를 바로 failed로 굳히고, 메시지가 status_reason으로 화면에
    나간다(PictureLimitError와 같은 길). 문구 규칙은 엑셀 거절(XlsxTableTooLarge)과 같다 — 토큰을 말하지 않고 허용 크기
    대비 %와 행×열로 말한다. 엑셀처럼 업로드 순간 거절하지 못하는 것은 표 크기를 docling 변환 뒤에야 알 수 있어서다.
    """

    def __init__(self, page: int | None, header: list[str], percent: int, rows: int, cols: int):
        self.page, self.percent, self.rows, self.cols = page, percent, rows, cols
        name = ' | '.join(h for h in header if h.strip())[:30]
        where = f"{page}쪽 표" if page else "표"
        super().__init__(f"{where}('{name}')가 허용 크기의 {percent}%입니다(현재 {rows}행 × {cols}열). "
                         "상담에 필요 없는 열이나 긴 설명을 줄여 다시 올려 주세요.")


class PictureLimitError(ValueError):
    """캡션 대상 그림이 상한(settings.docling_max_pictures)을 넘는다 — **결정적** 실패(#223 결정 4).

    ValueError라 outbox.is_transient가 결정적으로 분류한다 — 다시 돌려도 그림 수는 같다.
    """


class CaptionFailedError(ConnectionError):
    """그림 캡션 요청이 실패했다 — **일시** 실패(#223 결정 3).

    docling은 VLM 오류(타임아웃·5xx·연결 거부)를 로그로만 남기고 **빈 캡션**을 돌려준다
    (api_image_request). 그대로 두면 그림 내용이 빠진 문서가 ready가 된다 — 사용자도 운영자도 모른다.
    그래서 빈 캡션을 실패로 올린다. ConnectionError 하위라 outbox.is_transient가 일시로 분류해
    백오프 재시도한다 — VLM이 살아나면 저절로 ready가 된다. 알려진 오분류: VLM이 특정 그림에만
    4xx를 내는 결정적 경우도 일시로 잡혀 최대 MAX_ATTEMPTS까지 돈다 — 관측된 적 없어 감수한다.
    """


class _PictureGuard:
    """docling 캡션 모델을 감싸 그림 장수 상한과 캡션 실패를 건다(#223 결정 3·4).

    docling에는 둘 다 옵션이 없다 — 장수 상한도, 캡션 실패를 오류로 올리는 스위치도. 그래서 파이프라인의
    enrichment_pipe에 든 캡션 모델 인스턴스를 이 대리 객체로 바꿔 끼운다(_guard_picture_models).
    배치 크기·prepare_element 등 나머지는 __getattr__로 원본에 넘긴다.

    - 장수: 문서의 **첫 배치**에서 캡션 대상 수를 한 번 센다 — 문서가 다 만들어진 뒤라 정확하고, 넘으면
      캡션을 하나도 붙이기 전에 끝난다(넘는 문서에 VLM 시간을 쓰지 않는다). 대상 = docling이 실제로
      VLM에 보낼 그림, 즉 면적 하한(picture_area_threshold)을 넘는 것 — 로고·아이콘은 세지 않는다.
    - 실패: 캡션이 붙은 그림마다 결과를 보고 빈 캡션이면 멈춘다. 동시 1(settings.vlm_caption_concurrency)이면 docling이
      그림을 하나씩 요청하므로 첫 실패 뒤 요청이 없다 — VLM이 멈췄을 때의 비용은 타임아웃(45초) 한 번이다.
      동시 N이면 그 배치 안에서 이미 보낸 요청은 끝까지 기다린다(ThreadPoolExecutor 종료 대기).
      (5xx는 docling의 재시도 세션이 그림당 몇 차례 다시 부른다 — 읽기 타임아웃만 재시도가 없다.)
    문서 경계는 _convert_local이 arm()으로 알린다 — id(doc)로 가르면 해제된 문서의 id가 재사용될 수 있다.
    """

    def __init__(self, inner):
        self._inner = inner
        self._counted = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def arm(self) -> None:
        self._counted = False

    def __call__(self, doc, element_batch):
        if not self._counted:
            self._counted = True
            n = _captionable_count(doc, self._inner.options.picture_area_threshold)
            if n > settings.docling_max_pictures:
                raise PictureLimitError(f'그림 {n}장 — 문서당 {settings.docling_max_pictures}장을 넘는 문서는 받지 않는다')
        for item in self._inner(doc, element_batch):
            desc = item.meta.description if item.meta is not None else None
            if desc is not None and not (desc.text or '').strip():
                raise CaptionFailedError('그림 캡션 실패(VLM 응답 없음·오류) — 나중에 다시 시도한다')
            yield item


def _captionable_count(doc, area_threshold: float) -> int:
    """docling이 캡션을 요청할 그림 수 — PictureDescriptionBaseModel.__call__의 면적 판정을 그대로 따른다.

    순회도 docling과 같은 doc.iterate_items()다(본문 층만 — doc.pictures는 머리글 등 furniture도 담아 더 크다).
    분류 허용·거부 목록(classification_allow/deny)은 쓰지 않으므로 그 판정은 옮기지 않았다.
    """
    from docling_core.types.doc import PictureItem   # 지연 import — docling은 톱레벨에 안 들인다
    n = 0
    for item, _ in doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        if item.prov:
            prov = item.prov[0]
            page = doc.pages.get(prov.page_no)
            if page is not None:
                page_area = page.size.width * page.size.height
                if page_area > 0 and prov.bbox.area() / page_area < area_threshold:
                    continue
        n += 1
    return n


_XLSX_PICTURE_AREA_THRESHOLD = 0.0   # 엑셀 그림 캡션 면적 하한(#249 결정 1) — 근거는 _docling_runtime의 xlsx_opts 주석

_picture_guards: list[_PictureGuard] = []


def _guard_picture_models(converter) -> None:
    """세 형식의 파이프라인을 미리 띄우고 캡션 모델을 _PictureGuard로 바꿔 끼운다.

    initialized_pipelines는 DocumentConverter가 공개 속성으로 둔 파이프라인 캐시다(docling 2.126 docstring).
    #249부터 XLSX는 캡션 면적 하한이 달라(0) DOCX와 파이프라인을 공유하지 않는다 — 형식마다 따로 감싼다.
    docling 업그레이드로 이 구조가 바뀌면 캡션 실패·장수 테스트가 먼저 깨진다.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.models.picture_description_base_model import PictureDescriptionBaseModel
    for fmt in (InputFormat.PDF, InputFormat.DOCX, InputFormat.XLSX):
        converter.initialize_pipeline(fmt)
    for pipeline in converter.initialized_pipelines.values():
        for i, model in enumerate(pipeline.enrichment_pipe):
            if isinstance(model, PictureDescriptionBaseModel):
                guard = _PictureGuard(model)
                pipeline.enrichment_pipe[i] = guard
                _picture_guards.append(guard)


def _docling_runtime():
    """프로세스당 1회: 변환기(모델 로딩). **변환 프로세스(_ConvertProcess) 안에서** 불린다(#223).

    모델은 한 번만 올린다 — 이후 쪽당 0.3초(#143 실측). **리눅스 컨테이너 실측(2026-09-13)**:
    변환 피크 1.8GB·상주 1.4GB이고, 같은 프로세스로 12회 반복하면 4~6회차에서 평평해진다(누수 없음).
    OCR을 켜도 이 수준이다 — 엔진이 원격(Triton, #168)이라 워커엔 HTTP 클라이언트뿐: 26쪽 실문서 원격 OCR
    피크 1,699MB(리눅스 컨테이너 2026-09-17). 인프로세스로 켰을 때는 2,990MB였다. 다만 **정상상태는 더 높다**:
    125건을 한 프로세스로 돌리면 RSS가 2,3xx MB에서 평탄해지고 OCR 유무로 30MB밖에 안 다르다(2026-09-18,
    docker-compose.yml mem_limit 주석). 단일 문서 피크로 상한을 정하면 안 된다.
    다만 작업 전으로 돌아오지도 않는다 — 모델이 상주하고 glibc가 해제분을 OS에 바로 반납하지 않는다.
    컨테이너 메모리 상한(docker-compose.yml)은 이 피크 위에 잡는다 — 이 메모리는 이제 워커 본체가 아니라
    변환 프로세스에 있다(합계는 같다). 변환 동시성은 변환 프로세스가 하나라 1이다(옛 세마포어의 역할).
    """
    global _docling_converter
    with _DOCLING_LOCK:
        if _docling_converter is None:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (ConvertPipelineOptions, KserveV2OcrOptions,
                                                            PdfPipelineOptions,
                                                            PictureDescriptionApiOptions,
                                                            TableFormerMode)
            from docling.document_converter import (DocumentConverter, ExcelFormatOption, PdfFormatOption,
                                                    WordFormatOption)

            opts = PdfPipelineOptions(artifacts_path=settings.docling_artifacts_path)
            # 메모리 노브 — config.py 주석 참조. docling_settings(perf.page_batch_size)는 쓰지 않는다:
            # 2.126의 기본 파이프라인이 그 값을 읽지 않는다(옛 Legacy 경로 전용).
            opts.layout_batch_size = settings.docling_layout_batch_size
            opts.queue_max_size = settings.docling_queue_max_size
            opts.do_ocr = settings.docling_do_ocr
            if settings.docling_do_ocr:
                # 엔진은 워커가 아니라 원격 Triton(#168, config.py ocr_kserve_* 주석). 기본 OcrMode는 비트맵
                # 영역만 OCR하고 텍스트 레이어는 건드리지 않는다 — OCR on/off의 본문이 동일함을 실측했다.
                # 각도 분류기(use_cls=False)는 서버 쪽 상수다. 요청 실패는 docling이 삼키고 errors에만 남기므로
                # _docling_sections가 errors를 보고 실패로 올린다.
                opts.ocr_options = KserveV2OcrOptions(
                    url=settings.ocr_kserve_url, model_name=settings.ocr_kserve_model_name,
                    transport='http', lang=[_OCR_LANG], scale=_OCR_SCALE,
                    timeout=settings.ocr_kserve_timeout_seconds)
            opts.do_table_structure = True
            opts.table_structure_options.mode = (TableFormerMode.FAST if settings.docling_table_mode == 'fast'
                                                 else TableFormerMode.ACCURATE)
            opts.table_structure_options.do_cell_matching = settings.docling_do_cell_matching
            # opts.document_timeout은 걸지 않는다(#223) — 문서 마감은 _ConvertProcess가 프로세스째 끊는다.
            # docling의 것은 PDF 쪽 루프에서만 검사하고, 걸리면 PARTIAL_SUCCESS(→ 아래 가드가 결정적 실패)로
            # 끝나 "마감 초과 = 일시 실패"(결정 5)와 어긋난다. 같은 900초 둘이 경쟁하게 두지 않는다.
            opts.accelerator_options = AcceleratorOptions(
                num_threads=settings.docling_num_threads,
                device=AcceleratorDevice(settings.docling_device))
            # 그림 캡션(#162). enable_remote_services를 안 켜면 do_picture_description이 설정
            # 단계에서 OperationNotAllowed로 죽는다 — 항상 같이 켠다. VLM에 못 닿으면 docling은 빈 캡션을
            # 주는데, _PictureGuard가 그것을 CaptionFailedError(일시)로 올린다(#223 — 조용한 자리표시 색인 폐지).
            caption_opts = PictureDescriptionApiOptions(
                url=settings.vlm_caption_url,
                params={'model': settings.vlm_caption_model,
                        'temperature': settings.vlm_caption_temperature,    # 결정적 캡션(#209) — params는 요청 body에 그대로 들어간다
                        'max_tokens': settings.vlm_caption_max_tokens},     # 반복 고리 폭주 상한(#247 결정 16, config 주석)
                prompt=_PICTURE_CAPTION_PROMPT,
                picture_area_threshold=settings.vlm_caption_area_threshold,      # 기본 0.05 → 0.01(#247 결정 17, config 주석)
                scale=settings.vlm_caption_scale,
                timeout=settings.vlm_caption_timeout_seconds,
                concurrency=settings.vlm_caption_concurrency)
            opts.enable_remote_services = True
            opts.do_picture_description = True
            opts.picture_description_options = caption_opts
            # DOCX·XLSX(#220) — 본문이 원래 텍스트라 레이아웃·OCR 없는 SimplePipeline이다. 캡션만 PDF와 같은
            # 설정을 공유한다(같은 VLM·프롬프트·온도). 시간 상한은 PDF와 같은 _ConvertProcess 마감이다(#223).
            office_opts = ConvertPipelineOptions(enable_remote_services=True, do_picture_description=True,
                                                 picture_description_options=caption_opts)
            # XLSX만 면적 하한 0(#249 결정 1) — 엑셀은 쪽이 없어 docling이 시트를 칸 단위로 재고(그림 앵커까지 포함한
            # 최대 행·열) OneCellAnchor 그림을 크기와 무관하게 1×1칸으로 잡는다. 비율이 그림 크기가 아니라 '시트에 데이터가
            # 얼마나 많은가'를 재게 되어, 칸이 100개만 넘어도 큰 차트가 하한 1%에 걸려 캡션 없이 빠졌다(homeplus_17: 6×26칸
            # → 0.64%, 하한 0으로 재현하니 두 차트 모두 정확한 캡션). 그래서 엑셀 그림은 전부 캡션한다 — 로고 캡션 한 줄은
            # 무해하고, 그림 장수 상한(_PictureGuard, docling_max_pictures)은 이 하한을 그대로 읽어 엑셀 그림 전부를 센다.
            # 엑셀 차트 객체(데이터 칸에 연결된 네이티브 차트)도 그림으로 세지지만 이미지가 없어 캡션되지 않는다 — 값은 시트 칸에
            # 있어 표로 색인되고 자리표시만 남는다(실측). 차트+그림 100개 넘는 엑셀만 과대 집계로 거절될 수 있어 두었다(리뷰, 결정 2 = a).
            xlsx_opts = ConvertPipelineOptions(
                enable_remote_services=True, do_picture_description=True,
                picture_description_options=caption_opts.model_copy(
                    update={'picture_area_threshold': _XLSX_PICTURE_AREA_THRESHOLD}))
            _docling_converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts),
                                InputFormat.DOCX: WordFormatOption(pipeline_options=office_opts),
                                InputFormat.XLSX: ExcelFormatOption(pipeline_options=xlsx_opts)})
            _guard_picture_models(_docling_converter)
    return _docling_converter


# ── docling 텍스트 정제 (#204) — 각주 표시 인라인·VLM 캡션 잡담 ──────────────────

# 위첨자 각주 참조 번호. docling은 글자 크기를 버리고 'PM 2)'처럼 평문에 섞는다 — 생성기가 그 토큰에
# 걸려 승인자에서 PM을 빼먹었다(절제 실측: 원본 1/3 → '2)' 제거 3/3, #203 b2_sf004). 열거("1) 직원이 …
# 2) 회사의 …")와 글자 모양이 같아 정규식 일괄 제거는 출장여비 지침 5청크를 깬다(코퍼스 스캔) — 둘을 가르는
# 유일한 물리적 차이는 **글자 크기 비율**(각주 0.67~0.73, 열거 1.0)이라 pdfplumber 단어 크기로 판정한다.
# docling의 정의 줄("2) PM 없는 경우 생략")은 9쪽 중 2쪽에서만 나와(3·6·7쪽 누락) 정의 줄 기반 판정은 기각했다.
_FOOTNOTE_TOKEN_RE = re.compile(r'^\d{1,2}\)$')
_FOOTNOTE_SUP_RATIO = 0.8        # 같은 줄 최대 글자 크기 대비. 실측 각주 ≤0.73, 정의 줄·열거 ≥0.81
_FOOTNOTE_LINE_TOL = 4.0         # 같은 줄로 묶는 top 차이(pt). 위첨자는 본문보다 2~3pt 올라간다


def _markers_from_words(words: list[tuple[str, float, float]]) -> set[str]:
    """(단어, 글자크기, top) 목록 → 위첨자 각주 표시 집합. 한 쪽 분량을 받는다.

    top이 _FOOTNOTE_LINE_TOL 안이면 같은 줄로 묶고, 줄 최대 크기의 _FOOTNOTE_SUP_RATIO 미만인 'N)' 토큰만
    각주 표시로 본다. 정의 줄("2) PM 없는 경우 생략")은 줄 전체가 작은 글자라 비율 1.0 → 걸리지 않는다(의도 —
    정의 줄은 본문에 남긴다). I/O 없는 순수 함수라 튜플만으로 테스트한다(_sections_from_docling_elements와 같은 규율).
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


def _footnote_markers(file_path: str | Path) -> dict[int, set[str]]:
    """쪽 → 위첨자 각주 표시 집합. pdfplumber 사이드카(쪽당 수십 ms) — docling 변환과 별개로 한 번 더 연다.

    **best-effort**: pdfplumber가 못 열거나 글자 크기를 못 읽으면 경고만 남기고 빈 dict — 각주 표시가 본문에
    남는 것은 #204 이전 동작 그대로라 저품질 색인이 아니고, docling이 이미 연 파일을 보조 파서 오류로
    failed 처리하는 쪽이 더 나쁘다. 실측(2026-10-08): inticube PDF 23건 중 표시가 잡힌 건 타임레포트 1건뿐.
    """
    out: dict[int, set[str]] = {}
    try:
        with pdfplumber.open(str(file_path)) as pdf:
            for page_no, page in enumerate(pdf.pages, start=1):
                words = [(w['text'], float(w.get('size') or 0.0), float(w['top']))
                         for w in page.extract_words(extra_attrs=['size'])]
                found = _markers_from_words(words)
                if found:
                    out[page_no] = found
    except Exception as e:   # noqa: BLE001 — 보조 경로. 사유는 남기고 본 경로는 계속 간다
        logger.warning('각주 사이드카 실패(각주 표시 제거 없이 진행) %s: %s', file_path, e)
    return out


def _strip_footnote_markers(text: str, markers: set[str]) -> str:
    """본문에서 각주 참조 'N)'를 뗀다 — 단어·닫는 괄호 **뒤에 붙은 것만**. 줄 머리의 정의 줄("2) PM …")과
    같은 번호가 아닌 열거, 더 큰 숫자의 일부("12)")는 그대로다. 같은 쪽에 같은 번호의 열거가 공존하면 그것도 지워진다(실측 0건, 감수)."""
    for marker in sorted(markers, key=len, reverse=True):
        # 앞의 공백은 같은 줄 안에서만([ \t]) — 줄바꿈을 허용하면 다음 줄 머리의 정의 줄 "2) PM …"까지 지운다.
        # (?<!\d): "서식 12)"의 "2)"는 각주가 아니라 숫자의 일부다 — 우리 정규식의 안전장치이지 문서에 거는 규칙이 아니다.
        pattern = (r'(?<=[\w\)\]）」])[ \t]?(?<!\d)' + re.escape(marker)
                   + r'(?=$|[\s,.;:)>\]）」→])')
        text = re.sub(pattern, '', text)
    return text


def _docling_norm(text: str) -> str:
    """docling 텍스트의 NBSP(\\xa0)를 보통 공백으로. 표 셀·헤딩에 섞여 들어와 검색 토큰을 깨뜨린다."""
    return re.sub(r'[  ]+', ' ', text).strip()


def docling_grid_markdown(table, doc=None) -> str | None:
    """docling 표 → markdown, **셀 원문(grid)에서 직접** 그린다. PDF·DOCX·XLSX 표의 정의점(#220, #252부터 PDF도).

    #252 결정 2까지 PDF는 export_to_dataframe 경로였다 — docling이 column_header 표시를 안 붙인 표(단순 표에서 흔함: 실문서
    PDF 표 40개 중 22개)에 열 이름 0,1,2…를 붙여 헤더가 `| 0 | 1 | 2 |`로 색인됐다. 이 함수는 첫 행을 헤더로 보므로 그 결함이
    없다. 대가로 2단 헤더를 점으로 잇던 것('국내 출장.일비')이 사라지고 둘째 헤더 행이 데이터 행으로 들어간다(사용자 결정).
    옛 dataframe 경로와 다른 두 가지:
      - **병합 칸은 범위의 모든 칸에 값을 둔다(가로·세로 모두).** grid는 병합 범위의 모든 칸에 같은 셀을 넣으므로 그대로 그린다.
        #220~#252 결정 9 전까지 가로 병합은 첫 열에만 두었다 — 양식형 xlsx에서 '업무 인수 인계서'가 칸마다 반복됐기
        때문이다(실문서 2건에서 23·67회, 텍스트 1.1·3.0배). 그런데 PDF까지 이 규칙을 쓰자(#252 결정 2) '근무기간 7년 이상·
        7년 미만' 두 칸에 걸친 `1 년`이 `1 년 |  `이 되어 7년 미만 칸이 빈칸으로 읽혔다(휴직 안내 3행, 정답 라벨 1건 소실).
        제목 반복의 잡음보다 값이 빈칸으로 읽히는 쪽이 위험하다고 보고 전부 반복한다(사용자 결정, 2026-10-17).
        세로는 #220부터 반복한다 — 첫 행에만 두면 표가 청크 경계에서 잘릴 때 아래 행이 분류어를 잃는다(장애관리 지침서 37행 중
        7행, 인사규정 18행 중 3행). 대가는 긴 병합 셀의 중복이다(인사규정 +13%·543자 셀 5회, 교통비 양식 +17%) —
        글자가 늘더라도 행이 제 분류를 지니는 쪽을 택했다(#220 결정 10, 2026-10-11). 짧은 셀만 반복하는 길이 임계안은 기각.
      - **export_to_dataframe을 쓰지 않는다.** Word의 서식 있는 셀(RichTableCell)을 '**항목**'처럼 마크다운 굵게로
        내보낸다 — 셀 원문(cell.text)에는 '**'가 없다.
    값이 하나도 없는 행은 뺀다(xlsx _iter_sheets의 빈 행 스킵과 같은 원리). 첫 행이 헤더다.
    빈 셀은 자리를 지킨다(to_markdown_table — #137 결함 5). 열이 없으면 None.
    **칸 안의 그림은 그 칸에 설명을 붙인다**(doc가 있을 때) — 서식 있는 칸(RichTableCell)의 ref가 가리키는 묶음 안의
    그림을 찾아 캡션(없으면 자리표시)을 칸 글자 뒤에 한 줄로 편다. 매뉴얼의 '화면 | 캡처' 표에서 캡처 설명이 제 행과
    붙어 있게 하려는 것이다(#220 리뷰, 결정 2026-10-11). 표 아래로 빼는 안은 어느 행의 그림인지를 잃어 기각했다.
    대가: 여러 줄 캡션의 줄바꿈이 사라지고(마크다운 표는 행=한 줄), 긴 캡션이면 그 행이 길어진다.
    이 처리 전에는 VLM이 캡션을 만들어 놓고도 칸 글자에 없어 버려졌다(실측: 'LOGIN ID / PASSWORD' 캡션이 색인 0).
    """
    rows = docling_grid_rows(table, doc)
    return to_markdown_table(rows[0], rows[1:]) if rows else None


def docling_grid_rows(table, doc=None) -> list[list[str]]:
    """docling 표 → 행 목록(첫 행 = 헤더). docling_grid_markdown의 행 단계 — 규칙은 그 docstring(병합·칸 안 그림·빈 행).

    따로 둔 이유(#222 리뷰): 엑셀 행 묶음은 **글로 바꾸기 전에 행 목록에서** 나눠야 한다. 렌더한 markdown을 줄 단위로
    다시 해석하면 칸 안 줄바꿈("구분⏎(원)" 헤더, "첫줄⏎| 둘째줄" 데이터)에서 헤더가 깨지고 행이 중간에서 잘렸다(실측).
    """
    rows = [[_grid_cell_text(cell, doc) for cell in row] for row in table.data.grid]
    return [row for row in rows if any(row)]


_XLSX_MIDNIGHT_RE = re.compile(r'^(\d{4}-\d{2}-\d{2}) 00:00:00$')


def _cell_text_norm(text: str) -> str:
    """칸 글자 정리(#252 C·F): 줄바꿈은 공백으로 — markdown 표는 행 = 한 줄이라 `\n`이 남으면 그 행이 깨져 뒤 열이 다음
    줄로 밀린다(장애관리 docx 24칸·xlsx 3칸 실측). 엑셀 날짜 칸의 `00:00:00` 꼬리는 뗀다(docling이 datetime을 str로 바꾸며
    붙인다 — 교통비 양식 2칸). 자정이 아닌 실제 시각은 그대로 둔다."""
    text = _docling_norm(text.replace('\n', ' '))
    return _XLSX_MIDNIGHT_RE.sub(r'\1', text)


def _grid_cell_text(cell, doc) -> str:
    """표 칸 하나의 글자 — 칸 원문(_cell_text_norm) + 칸 안 그림의 설명(docling_grid_markdown 참조)."""
    from docling_core.types.doc import PictureItem, RichTableCell   # 지연 import — docling은 톱레벨에 안 들인다
    parts = [_cell_text_norm(cell.text)]
    if doc is not None and isinstance(cell, RichTableCell):
        parts += [re.sub(r'\s+', ' ', docling_picture_text(item))
                  for item, _ in doc.iterate_items(root=cell.ref.resolve(doc)) if isinstance(item, PictureItem)]
    return ' '.join(p for p in parts if p)


def docling_picture_text(element) -> str:
    """그림 요소의 본문 한 줄 — 캡션, 없으면 자리표시(캡션 대상이 아닌 그림). DOCX·XLSX 경로가 공유한다(#220)."""
    return _docling_norm(_docling_picture_caption(element)) or _PICTURE_PLACEHOLDER


def docling_block_text(item, doc) -> str:
    """docling 요소 하나 → 본문 텍스트(내놓을 것이 없으면 ''). DOCX·XLSX가 공유하는 요소 종류 분기의 정의점(#220).

    표는 docling_grid_markdown, 그림은 캡션 또는 자리표시, 목록 항목은 번호를 되붙인 글자(_docling_elements와 같은 규칙 —
    빠뜨리면 '2.1 얼리버드…'가 번호 없이 색인된다), 그 밖의 글자 요소는 글자. 헤딩 판정은 형식마다 달라 호출부가 한다.
    """
    from docling_core.types.doc import ListItem, PictureItem, TableItem, TextItem   # 지연 import
    match item:
        case TableItem():
            return docling_grid_markdown(item, doc) or ''
        case PictureItem():
            return docling_picture_text(item)
        case ListItem(marker=marker) if marker:
            return _docling_norm(f'{marker} {item.text}')
        case TextItem():
            return _docling_norm(item.text)
    return ''


def count_picture_placeholders(chunks) -> int:
    """청크들에 남은 그림 자리표시 수(#162). 캡션이 안 붙은 그림의 개수다.

    #223 전에는 VLM 실패 신호였다(docling이 실패를 삼켜 자리표시로 남겼다). 이제 VLM 실패는 _PictureGuard가
    문서 재시도로 올리므로, 이 값은 **캡션 대상에서 빠진 그림**(면적 하한 미만, XLSX #249) 수다. 지표 기록은 호출부(rag/documents.py)가 한다:
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


def _docling_elements(doc, footnote_markers: dict[int, set[str]] | None = None):
    """docling 문서 → (label, text, page, table_rows) 튜플 흐름. 읽기 순서. table_rows는 행 목록(첫 행 헤더, docling_grid_rows —
    #252부터 PDF도 DOCX·XLSX와 같은 렌더)이고 표가 아니면 None.

    _sections_from_docling_elements가 소비한다. 둘로 나눈 이유: 매핑 규칙(헤딩 층·표·그림)을
    docling 없이 튜플만으로 단위 테스트하기 위해서다 — 모델 다운로드가 필요한 테스트와 분리.
    footnote_markers: 쪽 → 위첨자 각주 표시(_footnote_markers, #204). 텍스트 요소에서 그 쪽의 표시만 뗀다 —
    표(셀 구조)·그림(캡션)은 대상이 아니다.
    """
    footnote_markers = footnote_markers or {}
    for element, _depth in doc.iterate_items():
        label = element.label.value if hasattr(element.label, 'value') else str(element.label)
        page = element.prov[0].page_no if getattr(element, 'prov', None) else None
        if label == 'table':
            yield label, '', page, docling_grid_rows(element, doc) or None     # #252 결정 2 — DOCX·XLSX와 같은 렌더
        elif label == 'picture':
            # 캡션(#162)은 element.text가 아니라 annotations에 붙는다 — text만 보면 조용히 버려진다.
            # 캡션 대상이 아닌 그림(면적 하한 미만)은 annotations가 비어 빈 문자열이 나가고, 소비자가 자리표시로 되돌린다.
            yield label, _docling_picture_caption(element), page, None
        else:
            text = getattr(element, 'text', '') or ''
            # 목록 항목의 번호·글머리('2.1', '-', '(1)')는 text에서 떼어 marker에 둔다. 빠뜨리면
            # '2.1 얼리버드 할인율 상향'이 '얼리버드 할인율 상향'으로 색인된다(모의 코퍼스 33건 대조에서
            # 발견). docling의 export_to_markdown과 같은 방식으로 다시 붙인다.
            marker = getattr(element, 'marker', '') or ''
            if label == 'list_item' and marker:
                text = f'{marker} {text}'
            if page in footnote_markers:
                text = _strip_footnote_markers(text, footnote_markers[page])
            yield label, text, page, None


def _join_split_article_titles(elements):
    """② 닫는 괄호만 다음 요소로 떨어진 조 제목을 잇는다 — '제 7 조 ( 징계종류' + ')' → '제 7 조 ( 징계종류 )'(징계규정 3건, #241)."""
    elements = list(elements)
    i = 0
    while i < len(elements):
        label, text, page, table = elements[i]
        nxt = elements[i + 1] if i + 1 < len(elements) else None
        t = _docling_norm(text or '')
        if (label in ('section_header', 'text') and nxt is not None and _docling_norm(nxt[1] or '') in (')', '）')
                and (head := split_heading_line(t)) and head[0] == t and not any(c in t for c in ')）')):
            yield label, f'{t} {_docling_norm(nxt[1])}', page, table
            i += 2
            continue
        yield label, text, page, table
        i += 1


def _sections_from_docling_elements(elements) -> list[_Section]:
    """(label, text, page, table) 흐름 → _Section. 헤딩 층 복원 규칙의 정의점. table은 행 목록(첫 행 헤더, #252 — _TableBlock으로
    운반)이거나 옛 계약의 markdown 문자열(그대로 글 블록).

    docling은 장·조를 구분하지 않으므로(_CHAPTER_RE 참조) 여기서 층을 만든다:
      첫 헤딩 = 문서 제목(L1) / '제 N 장' = L2 / 그 외 헤딩 = L3 (장이 아직 없으면 L2).
    그러면 _pack_sections의 joinable(공통 조상 깊이 ≥2)이 현행과 같이 동작해 같은 장 아래
    조항끼리 묶인다. 이 규칙 없이 두면 26쪽 규정이 73→170청크로 갈렸다(#143 실측).
    표는 _TableBlock(행 목록 — chunk_file이 행 단위로 나눈다, #252), 그림은 캡션(없으면 _PICTURE_PLACEHOLDER), 나머지 텍스트
    요소는 본문 줄.

    조 제목 보정(#240·#241) — docling 레이아웃 모델은 본문과 같은 글꼴·크기의 조 제목을 들쭉날쭉 판정한다(회계 규정: 조 제목 97줄이
    글꼴·크기·들여쓰기·앞 간격까지 같은데 61줄만 section_header, 35줄은 text(그중 3줄은 다음 문장과 한 덩어리), 1줄은 쪽 경계에서
    순서가 섞였다. 다음 요소 종류와도 무관 — 판정 규칙을 찾을 수 없었다.
    레이아웃 모델 6종 비교에서도 최선 egret-medium이 35→8에 그치고 다른 문서에서 새로 놓쳤다, 2026-10-21). 두 꼴만 고친다:
      ① text 요소가 조 제목 꼴(rag/identifiers.split_heading_line)로 시작하면 제목으로 올리고, 같은 요소에 이어진 글은 그 조의
         본문으로 둔다 — '제1조 (목적) 이 규정은 …'처럼 제목과 첫 문장을 한 덩어리로 내는 경우가 흔하다(조 제목 뒤 간격이 좁은
         조판: 실문서 회계 규정 3건, 실측 간격에 맞춘 합성 규정은 30조 전부). 판정은 DOCX 조문 폴백(#205)과 같은 함수지만 범위가
         넓다 — DOCX는 헤딩 스타일이 0개인 문서에서만 돌고 여기는 문서 제목 뒤 모든 text 요소에 돈다. 그래서 '제98조(보상) 규정에
         따라 …'로 시작하는 본문 문단(쪽·단 경계에서 문장 중간부터 시작한 요소, 목차 줄)은 제목으로 오인된다(#233 결정 2, 글은
         남는다). 실문서 PDF 20건에서 이 규칙이 올린 줄은 35건 전부 진짜 조 제목이었다(오탐 0, 2026-10-22).
      ② 닫는 괄호가 다음 요소로 떨어진 조 제목('제 7 조 ( 징계종류' + ')')은 하나로 잇는다 — 안 이으면 ')'가 제목 경로에 들어갔다.
         제목 쪽이 text로 나와도(①이 올릴 꼴) 잇는다
    올린 조 제목은 문서 제목이 될 수 없다 — 문서 제목(첫 section_header)이 나온 뒤에만 올린다. docling이 문서 제목까지 본문으로
    낸 문서(실측 간격 합성 규정)에서는 보정이 걸리지 않는다 — 운영하며 실문서로 대응한다(#240, 2026-10-22 사용자 결정).
    조문 체계 밖의 제목('1. 목적', 번호 없는 절 제목)은 글자로 가를 단서가 없어 보정하지 않는다(모델 한계로 남김).
    """
    items: list[_Item] = []
    seen_title = False
    seen_chapter = False
    for label, text, page, table_markdown in _join_split_article_titles(elements):
        if label == 'text' and seen_title and (head := split_heading_line(_docling_norm(text))):
            items.append((page, head[0], 3 if seen_chapter else 2))     # ① 조 제목 꼴 본문 요소 → 제목(+ 이어진 글은 본문)
            if head[1]:
                items.append((page, head[1], None))
            continue
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
            if not table_markdown:
                continue
            if isinstance(table_markdown, str):               # 옛 계약(markdown 문자열) — 튜플만 먹이는 단위 테스트 호환
                items.append((page, table_markdown, None))
                continue
            block = _TableBlock(table_markdown, [page] * len(table_markdown))
            # 쪽을 넘긴 표 잇기(#252 결정 5 = E): 바로 앞 항목이 표이고(사이에 글·그림 없음) 첫 행(헤더)이 같으면 뒤 표의
            # 헤더를 떼고 행을 잇는다. docling은 쪽마다 표를 따로 내므로 45행 표가 36 + 10행으로 갈렸다(#246). 헤더가 같을
            # 때만 — 열 수만 같은 연속 표를 붙이는 안은 오판 위험으로 기각. 행별 쪽 번호는 각자 것을 지킨다(row_pages).
            if items and isinstance(items[-1][1], _TableBlock) and items[-1][1].rows[0] == block.rows[0]:
                prev_page, prev_block, _ = items[-1]
                items[-1] = (prev_page, _TableBlock(prev_block.rows + block.rows[1:],
                                                    prev_block.row_pages + block.row_pages[1:]), None)
                continue
            items.append((page, block, None))
        elif label == 'picture':
            # 캡션이 있으면 그것이 본문이다(#162). 없으면 자리표시 — 둘 다 한 줄 항목으로 들어간다.
            caption = _docling_norm(text)
            items.append((page, caption or _PICTURE_PLACEHOLDER, None))
        else:
            text = _docling_norm(text)
            if text:
                items.append((page, text, None))
    return _build_sections(items)


def _convert_local(file_path: str | Path, **kwargs):
    """이 프로세스에서 docling 변환 한 번 → DoclingDocument. 불완전 결과 가드의 정의점.

    변환 프로세스(_convert_child_main)가 부른다 — 워커 본체는 docling_convert를 쓴다.
    **raises_on_error=True만으로는 부족하다** — docling은 PARTIAL_SUCCESS(원격 OCR 실패 등)에 예외를 안
    던진다(document_converter.py: status not in {SUCCESS, PARTIAL_SUCCESS}일 때만). 그래서 status와
    errors를 직접 보고 올린다(#168). 캡션 실패·그림 장수는 _PictureGuard가 변환 도중에 올린다(#223) —
    docling이 그 예외를 RuntimeError('Pipeline … failed')로 감싸므로 원래 예외로 풀어 다시 던진다
    (감싼 채면 outbox.is_transient가 둘 다 결정적으로 본다).
    kwargs는 converter.convert로 넘긴다(PDF의 max_num_pages — 쪽 개념이 없는 형식엔 주지 않는다).
    """
    converter = _docling_runtime()
    for guard in _picture_guards:
        guard.arm()
    try:
        result = converter.convert(str(file_path), raises_on_error=True, **kwargs)
    except Exception as e:
        cause = e.__cause__
        while cause is not None:
            if isinstance(cause, (PictureLimitError, CaptionFailedError)):
                raise cause from None
            cause = cause.__cause__
        raise
    from docling.datamodel.base_models import ConversionStatus   # 지연 import — docling은 톱레벨에 안 들인다
    if result.status != ConversionStatus.SUCCESS or result.errors:
        detail = '; '.join(f'{e.category.value}/{e.module_name}: {(e.error_message or "")[:100]}'
                           for e in result.errors[:3])
        raise RuntimeError(f'docling 변환 불완전 — status={result.status.value}, '
                           f'errors={len(result.errors)}: {detail}')
    return result.document


def _picklable(exc: BaseException) -> BaseException:
    """부모로 보낼 수 있는 예외 — 피클이 안 되면 같은 문구의 RuntimeError(결정적)로 바꾼다."""
    try:
        pickle.dumps(exc)
        return exc
    except Exception:            # noqa: BLE001 — 직렬화 불가 예외는 종류를 잃는다(문구는 남긴다)
        return RuntimeError(f'{type(exc).__name__}: {exc}')


def _convert_child_main(fd: int) -> None:
    """변환 프로세스의 본체 — 요청 (경로, kwargs)를 받아 ('ok', DoclingDocument) 또는 ('err', 예외)를 돌려준다.

    첫 메시지는 부모의 설정 스냅숏이다(변환에 쓰는 설정만 — _CHILD_SETTING_PREFIXES) — 자식은 config를 .env에서 새로 읽으므로, 부모가 바꾼 값
    (테스트의 monkeypatch 등)이 자식에 닿게 하는 유일한 길이다. 모델 로딩은 'ready' 전에 끝낸다 —
    기동 시간(첫 실행이면 모델 다운로드까지)이 첫 문서의 마감을 깎지 않게.
    """
    from multiprocessing.connection import Connection
    conn = Connection(fd)
    for key, value in conn.recv().items():
        setattr(settings, key, value)
    try:
        _docling_runtime()
    except Exception as e:       # noqa: BLE001 — 기동 실패도 부모에 사유로 돌려준다
        conn.send(('err', _picklable(e)))
        return
    conn.send(('ready', None))
    while True:
        try:
            file_path, kwargs = conn.recv()
        except EOFError:             # 부모가 끊었다(재기동·종료) — 부모가 죽어도 여기서 빠진다
            return
        try:
            conn.send(('ok', _convert_local(file_path, **kwargs)))
        except Exception as e:       # noqa: BLE001 — 문서 단위 실패는 부모가 분류한다(outbox.is_transient)
            conn.send(('err', _picklable(e)))


_CHILD_START_TIMEOUT_SECONDS = 300   # 변환 프로세스 기동(모델 로딩) 상한. 맥 실측 콜드 4~5.5초 + 첫 변환 3~4초(#223) —
                                     # 300은 첫 실행의 모델 다운로드(docling_artifacts_path 미지정)까지 품는 여유다.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# 변환 프로세스가 쓰는 설정의 접두사 — 부모는 이 설정만 자식에 넘기고, 이것이 바뀌었을 때만 자식을 다시 띄운다(#267).
# 설정 전체를 비교하던 때는 테스트가 임베딩 주소·한도처럼 변환과 무관한 값을 바꿔 끼울 때마다 콜드 기동(맥 실측 8.5초)이 났다 —
# MD가 변환 프로세스를 타면서 통합 테스트 75개가 2분 8초 → 5분 55초(2026-10-19). 운영은 설정이 고정이라 차이 없다.
# 이 모듈이 읽는 settings 필드가 모두 이 접두사 안인지는 tests/test_chunking_md.py가 검사한다 — 새 접두사의 설정을 자식에서
# 읽으면서 여기 안 넣으면, 값을 바꿔도 자식이 옛 값으로 계속 돈다.
_CHILD_SETTING_PREFIXES = ('docling_', 'ocr_kserve_', 'vlm_caption_')


def _child_settings() -> dict:
    return {k: v for k, v in settings.model_dump().items() if k.startswith(_CHILD_SETTING_PREFIXES)}


class _ConvertProcess:
    """docling 변환을 맡는 상주 자식 프로세스 하나(#223 결정 1). 이 모듈을 쓰는 프로세스(워커)마다 하나다.

    왜 프로세스인가: docling에는 취소 API가 없다. 스레드(to_thread)로 돌리면 마감을 넘겨도 멈출 수 없어,
    arq가 잡을 취소한 뒤에도 고아 스레드가 변환을 계속하며 세마포어를 쥐고 있었다(#223 탐색). 프로세스는
    죽일 수 있다 — 마감을 넘기면 kill하고 다음 요청 때 새로 띄운다. 자식은 **변환만** 한다: 임베딩·엔진 쓰기·
    커밋은 부모에서 그 뒤에 일어나므로, 변환 중 kill해도 반쪽 색인은 생기지 않는다.

    - 마감: settings.docling_document_timeout_seconds — PDF·DOCX·XLSX 전 형식, 캡션 단계 포함(결정 2).
      넘기면 TimeoutError(outbox.is_transient가 일시로 분류 — 결정 5).
    - 동시성 1: 잠금 하나로 요청을 직렬화한다(옛 docling_max_concurrency 세마포어 대체).
    - 띄우는 법: multiprocessing(spawn·forkserver)이 아니라 `python -c`로 직접 띄우고 소켓 한 쌍 위에
      multiprocessing Connection(피클 메시지)을 얹는다. spawn은 자식에서 부모의 __main__ 스크립트를 다시
      import한다 — `if __name__` 보호가 없는 eval 스크립트면 그 본문이 자식에서 한 번 더 돈다(stdin 실행은
      아예 죽는다, 2026-10-14 실측). fork는 부모의 스레드·잠금 상태를 복제해 교착 위험이 있다.
    - 설정이 바뀌면(스냅숏 불일치) 재기동한다 — 자식은 기동 때의 설정으로 변환기를 만들기 때문이다.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._conn = None
        self._snapshot: dict | None = None

    def convert(self, file_path: str | Path, kwargs: dict):
        with self._lock:
            snapshot = _child_settings()
            if self._proc is None or self._proc.poll() is not None or snapshot != self._snapshot:
                self._start(snapshot)
            deadline = settings.docling_document_timeout_seconds
            self._conn.send((str(Path(file_path).resolve()), kwargs))   # 자식의 cwd는 저장소 루트다
            if not self._conn.poll(deadline):
                self._stop()
                raise TimeoutError(f'docling 변환이 {deadline:.0f}초를 넘겨 중단했다(변환 프로세스 재기동)')
            try:
                kind, value = self._conn.recv()
            except EOFError:             # 자식이 죽었다(OOM kill 등) — poll은 EOF에도 True다
                code = self._exit_code()
                self._stop()
                raise RuntimeError(f'docling 변환 프로세스가 비정상 종료했다(exitcode={code})') from None
        if kind == 'ok':
            return value
        raise value

    def _start(self, snapshot: dict) -> None:
        from multiprocessing.connection import Connection
        self._stop()
        parent_sock, child_sock = socket.socketpair()
        env = {**os.environ, 'PYTHONPATH': os.pathsep.join(
            p for p in (str(_PROJECT_ROOT), os.environ.get('PYTHONPATH')) if p)}
        try:
            self._proc = subprocess.Popen(
                [sys.executable, '-c',
                 'import sys; from rag.chunking import _convert_child_main; _convert_child_main(int(sys.argv[1]))',
                 str(child_sock.fileno())],
                pass_fds=(child_sock.fileno(),), cwd=_PROJECT_ROOT, env=env)
        except BaseException:
            parent_sock.close()
            raise
        finally:
            child_sock.close()
        self._conn = Connection(parent_sock.detach())
        self._conn.send(snapshot)
        if not self._conn.poll(_CHILD_START_TIMEOUT_SECONDS):
            self._stop()
            raise TimeoutError(f'docling 변환 프로세스 기동이 {_CHILD_START_TIMEOUT_SECONDS}초를 넘겼다')
        try:
            kind, value = self._conn.recv()
        except EOFError:
            code = self._exit_code()
            self._stop()
            raise RuntimeError(f'docling 변환 프로세스가 기동 중 종료했다(exitcode={code})') from None
        if kind != 'ready':
            self._stop()
            raise value
        self._snapshot = snapshot

    def abort(self) -> None:
        """진행 중인 변환을 끊는다 — **잠금 없이** 부른다(잠금은 기다리는 스레드가 쥐고 있다). 다른 스레드에서 불려도
        안전하다: kill만 하고, 기다리던 convert()가 EOF로 깨어나 정리(_stop)하고 잠금을 푼다."""
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.kill()

    def _exit_code(self):
        try:
            return self._proc.wait(5)
        except subprocess.TimeoutExpired:
            return None

    def _stop(self) -> None:
        if self._conn is not None:
            self._conn.close()
        if self._proc is not None and self._proc.poll() is None:
            self._proc.kill()            # 자식은 지킬 상태가 없다 — C 코드에 묶여 있어도 끊기는 SIGKILL로
            self._proc.wait(10)
        self._proc = self._conn = self._snapshot = None


_convert_process = _ConvertProcess()


def abort_conversion() -> None:
    """진행 중인 docling 변환을 끊는다(변환 프로세스 kill). 색인 잡이 취소될 때 rag/documents.py가 부른다(#223)."""
    _convert_process.abort()


def docling_convert(file_path: str | Path, **kwargs):
    """docling 변환 한 번 → DoclingDocument. PDF·DOCX·XLSX가 공유한다(#220). 변환 프로세스에 맡긴다(#223).

    실패는 예외로 올린다 — 분류(일시/결정적)는 outbox.is_transient: 마감 초과 TimeoutError·캡션 실패
    CaptionFailedError는 일시, 그림 장수 PictureLimitError·불완전 변환 RuntimeError는 결정적.
    rag/xlsx_chunking이 이 이름을 import한다.
    """
    return _convert_process.convert(file_path, kwargs)


def _docling_sections(file_path: str | Path) -> list[_Section]:
    """PDF → docling(레이아웃 ML + TableFormer) → _Section. PDF 인제스션의 기본 경로 (#143).

    pdfplumber 휴리스틱(#141)이 손으로 막던 것을 레이아웃 모델이 구조적으로 대신한다:
      - 인쇄 헤더·URL 푸터는 page_header/page_footer로 분류돼 iterate_items()에 나오지 않는다 (결함 1)
      - 제목이 폰트 혼용으로 쪼개지지 않는다 (결함 2)
      - 표는 셀 구조로 나와 빈 셀·병합 셀이 보존된다 (결함 5)
    임계 상수가 0개다. 헤딩 층 복원은 _sections_from_docling_elements가 한다.

    실패(예외·타임아웃)는 그대로 올린다 — pdfplumber 폴백 없음. 저품질 색인을 조용히 만들지 않는다
    (strict-grounded). 호출부(index_pending_document)가 failed로 기록한다. 불완전 결과 가드는 _convert_local.
    이미지 도표는 VLM 캡션으로 읽고(#162), 격자 표 이미지(TableItem)는 원격 한국어 OCR이 셀을 채운다(#168).
    """
    document = docling_convert(file_path, max_num_pages=settings.docling_max_num_pages)
    # 위첨자 각주 표시(#204) — 변환이 성공한 뒤에만 연다(변환 프로세스 밖, 쪽당 수십 ms). 가드보다 뒤에 두는 이유:
    # 변환 실패 사유가 pdfplumber 오류에 가려지지 않게, 그리고 가짜 컨버터 테스트가 파일 없이도 가드를 검증하게.
    footnote_markers = _footnote_markers(file_path)
    return _sections_from_docling_elements(_docling_elements(document, footnote_markers))


def _docx_items(file_path: str | Path):
    """DOCX body를 **문서 원래 순서대로** (is_heading, level, text, from_table)로 흘린다. python-docx — 채팅 첨부
    (_docx_text) 전용. 색인은 _docx_docling_items(#220).

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
            yield (bool(m), int(m.group(1)) if m else 0, text, False)
        elif tag == 'tbl':                         # docx 표는 구조화 포맷 → 셀 값 안정 추출
            for row in DocxTable(child, doc).rows:
                # 빈 셀도 자리를 지킨다 — 걸러내면 뒷 컬럼이 앞으로 당겨져 헤더와 어긋난다.
                # ['제주','','5000원','도서산간 별도'] → '제주 | 5000원 | 도서산간 별도'가 되어
                # LLM이 '조건=5000원'으로 읽는다. 완전 빈 행만 스킵 (xlsx 파서와 같은 원리).
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    yield (False, 0, ' | '.join(cells), True)     # from_table — 폴백 헤딩 승격 대상에서 제외


def _docx_text(file_path: str | Path) -> str:
    """DOCX 전체를 평문 하나로 (채팅 첨부 — 컨텍스트 직접 주입용). 헤딩 포함, 문서 순서 보존."""
    return '\n'.join(text for _, _, text, _ in _docx_items(file_path))


_DOCX_TITLE_MAX = 60


def _split_heading_head(text: str) -> tuple[str, str]:
    """조문·장 문단 → (헤딩 머리, 나머지 본문). 조는 첫 줄의 '제N조(제목)'까지(identifiers.split_heading_line, #233),
    장은 첫 줄 전체.

    문단 전체를 헤딩으로 올리면 "제8조(역할과 책임) ① 운영관리자는 …"처럼 본문이 같은 문단에 이어진 조문의
    본문이 heading_path로 들어가고, 뒤에 본문 줄이 없으면 섹션이 생기지 않아 **색인에서 사라진다**(리뷰 지적).
    괄호 짝을 세므로 '제2조(시향 예약(단체))'가 안쪽 닫는 괄호에서 잘리지 않는다(예전 정규식은 잘랐다).
    """
    first, _, rest_lines = text.partition('\n')
    head, rest = split_heading_line(first) or (first.strip(), '')     # 장('제 N 장 …')은 첫 줄 전체가 머리
    return head, (rest + ('\n' + rest_lines if rest_lines else '')).strip()


def _docx_fallback_headings(items: list[tuple[bool, int, str, bool]]) -> list[tuple[bool, int, str, bool]]:
    """Heading 스타일이 **하나도 없는** docx의 조문 패턴 승격 (#205). 입력에 헤딩이 하나라도 있으면 그대로 돌려준다.

    실문서(장애관리 지침서)는 Heading 스타일 0개(Normal·List Paragraph뿐)에 '제N조(…)' 문단 13개로 절을 나눴다.
    그러면 문서 전체가 섹션 하나가 되고 SentenceSplitter가 표 행 한가운데를 자른다(#203 b2_sf021: "1차 대응하며," /
    "대응 불가 시 복구 담당자를 지정함"이 두 청크로). 층은 docling 경로(_sections_from_docling_elements)와 같다:
      제목 = 첫 문단의 첫 줄(표 행이 아니고 조문 패턴이 아니며 _DOCX_TITLE_MAX자 이하일 때) → L1
      '제 N 장' → L2 / '제 N 조' → L3(장이 아직 없으면 L2). 제목이 없으면 전부 한 단계 위.
    'N.'·'N.N' 번호 문단은 승격하지 않는다 — 매뉴얼의 목록 항목("1. 출근시간 선택")을 헤딩으로 오인한다(결정 1=a).
    표 행(from_table)은 셀 안에 조문이 있어도 승격하지 않는다 — 셀 단위 구조는 이 폴백의 범위 밖.
    조 제목은 rag.identifiers.split_heading_line(조 번호 + 괄호 제목, 정의점)이 판정하고, 헤딩은 그 **머리**('제N조(제목)')까지만,
    같은 문단에 이어진 본문은 본문 줄로 분리한다(_split_heading_head). 괄호 없이 적은 조('제8조 목적')는 인용과 가를 수 없어
    승격하지 않는다(#233 결정 2 — 본문으로 남는다).
    문서 제목 판정(첫 문단 첫 줄 ≤ _DOCX_TITLE_MAX자)은 휴리스틱이다 — "목 차" 같은 줄도 제목이 된다.
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
        if i == 0 and len(first_line) <= _DOCX_TITLE_MAX and not _CHAPTER_RE.match(first_line) and not is_heading_line(first_line):
            has_title = True
            out.append((True, 1, first_line, False))
            rest = text[len(text.split('\n', 1)[0]):].strip()
            if rest:
                out.append((False, 0, rest, False))
            continue
        # 조 제목 판정은 identifiers.split_heading_line 한 곳(#233). 예전 `_ARTICLE_RE`(조 번호 뒤 공백·괄호·줄끝)는
        # "제98조 및 제99조에 따라"·"제98조 제2항의"·"제98조(이하 …)에 따라" 같은 인용을 제목으로 올려, 뒤에 곧바로 다른
        # 제목이 오면 그 문장이 색인에서 사라졌다(#233 합성 문서 실측 4건 소실·2건 문장 절단).
        if _CHAPTER_RE.match(text) or is_heading_line(first_line):
            if _CHAPTER_RE.match(text):
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
    """DOCX 항목 — _docx_items(python-docx)·_docx_docling_items(docling)가 같은 꼴로 내놓아 #205 폴백·섹션 조립을 공유한다.

    튜플이라 기존 코드·테스트의 (is_heading, level, text, from_table) 풀어 쓰기가 그대로 통한다.
    """
    is_heading: bool
    level: int              # 헤딩 레벨(본문 줄은 0)
    text: str
    from_table: bool        # 표·그림 — 폴백 헤딩 승격에서 제외하는 표지


def _docx_table_rows(table, doc) -> list[list[str]]:
    """DOCX 표 → 행 목록(첫 행 헤더), **표 안의 표는 평탄화**(#252 결정 4 = D).

    docling은 안쪽 표를 바깥 표의 후손 항목으로 따로 내놓는데, 우리는 표 안 요소를 건너뛰므로(아래 _table_descendants) 안쪽
    표가 통째로 사라졌다(#246: 청크에 `평가 항목 | (빈칸)`). PDF는 TableFormer가 같은 표를 '안쪽 행 수만큼 펴고 바깥 칸 글자를
    행마다 반복'한 5×3으로 내 값이 보존됐고 질문도 맞혔다 — 그 모양을 그대로 따른다: 안쪽 표가 든 열은 안쪽 열 수만큼 넓어지고,
    그 행은 안쪽 행 수만큼 늘며, 같은 행의 다른 칸은 세로로 반복, 다른 행의 그 열 칸은 넓어진 열마다 반복한다
    (docling_grid_rows의 병합 반복 규칙 — #252 결정 9). 안쪽 표가 둘 이상인 칸은 위아래로 이어 붙인다.
    """
    from docling_core.types.doc import RichTableCell, TableItem   # 지연 import
    grid = table.data.grid
    nested: dict[tuple[int, int], list[list[str]]] = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if not (isinstance(cell, RichTableCell) and cell.start_row_offset_idx == r and cell.start_col_offset_idx == c):
                continue
            inner = [it for it, _ in doc.iterate_items(root=cell.ref.resolve(doc)) if isinstance(it, TableItem)]
            deeper = {d.self_ref for it in inner for d, _ in doc.iterate_items(root=it) if d is not it}
            inner = [it for it in inner if it.self_ref not in deeper]        # 3단 중첩은 재귀가 맡는다 — 중복 금지(리뷰 지적)
            rows = [row2 for it in inner for row2 in _docx_table_rows(it, doc)]
            if rows:
                own = _cell_text_norm(cell.text)                            # 칸에 안쪽 표와 함께 적힌 글자(라벨)는 첫 행으로 남긴다
                nested[(r, c)] = ([[own] + [''] * (len(rows[0]) - 1)] if own else []) + rows
    base = [[_grid_cell_text(cell, doc) for cell in row] for row in grid]
    if not nested:
        return [row for row in base if any(row)]
    width = {c: max(len(x) for (_, cc), rows in nested.items() if cc == c for x in rows)        # 열 c → 안쪽 열 수(최대)
             for c in {c for (_, c) in nested}}
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
                        line.extend((cells + [''] * width[c])[:width[c]])
                    else:
                        line.extend([text] * width[c])     # 넓어진 열에 걸친 칸 = 가로 병합 — 칸마다 반복(결정 9)
                else:
                    line.append(text)                      # 안쪽 행마다 반복 — 세로 병합 반복 규칙(docling_grid_rows)과 같다
            out.append(line)
    return [row for row in out if any(row)]


def _table_descendants(doc) -> set[str]:
    """표 칸 안에 든 요소들의 self_ref. 표 자신은 빼고, 중첩 표는 바깥 표의 후손이라 들어간다(_docx_table_rows가 평탄화해 싣는다)."""
    return {item.self_ref for table in doc.tables
            for item, _ in doc.iterate_items(root=table) if item is not table}


def _docx_docling_items(doc):
    """docling Word 문서 → _DocxItem 흐름(#220).

    _docx_items(python-docx)와 같은 형태로 내놓아 #205 폴백·섹션 조립을 그대로 재사용한다. 다른 점은 실측으로 정했다
    (보유 docx 32건 대조, 2026-10-11):
      - **헤딩 레벨은 docling의 section_header.level 그대로** — Heading 스타일 문서 29건 전부 스타일 수·레벨과 일치했다.
        PDF 매핑(_sections_from_docling_elements)의 제목/장 휴리스틱을 쓰면 3단이 2단으로 눌린다.
        'title' 라벨은 본문 줄이다 — python-docx 경로도 Title 스타일을 헤딩으로 보지 않았다.
      - **표 셀 안의 요소는 건너뛴다** — docling은 서식 있는 셀의 내용을 표와 별도 텍스트 요소로 **한 번 더** 내놓는다
        (표 있는 문서에서 텍스트류의 34~47%가 표 안). 셀 내용은 표 markdown(docling_grid_markdown)에 이미 들어 있다.
        칸 안의 그림도 여기서 건너뛰지만 버려지지 않는다 — docling_grid_markdown이 그 칸에 설명을 붙인다.
      - 워드 자동 번호('1.', '제2장')는 python-docx가 잃던 것이다 — docling은 읽는다(인사규정 2건에서 문장 373·435개가
        python-docx 경로에 없었다). 표 셀 안의 중첩 표는 #252 전까지 건너뛰기에 걸려 통째로 사라졌다 — 이제 _docx_table_rows가
        바깥 표에 평탄화해 싣는다.
    표는 _TableBlock(행 목록)으로 내놓는다(#252 — chunk_file이 행 단위로 나눈다). 표와 그림은 from_table=True — 폴백 헤딩
    승격에서 제외하는 표지다(그림 캡션이 '제1장 …'로 시작해도 헤딩이 되면 안 된다). 그림은 캡션, 없으면 자리표시.
    """
    from docling_core.types.doc import PictureItem, SectionHeaderItem, TableItem   # 지연 import
    inside_tables = _table_descendants(doc)
    for item, _ in doc.iterate_items():
        if item.self_ref in inside_tables:
            continue
        if isinstance(item, SectionHeaderItem):
            if text := _docling_norm(item.text):
                yield _DocxItem(True, item.level, text, False)
        elif isinstance(item, TableItem):
            if rows := _docx_table_rows(item, doc):
                yield _DocxItem(False, 0, _TableBlock(rows), True)
        elif text := docling_block_text(item, doc):
            yield _DocxItem(False, 0, text, isinstance(item, PictureItem))


def _docx_sections(file_path: str | Path) -> list[_Section]:
    """DOCX를 (heading_path, 본문) 섹션 목록으로 — Heading 스타일이 경계. 스타일이 하나도 없으면
    조문 패턴 폴백(_docx_fallback_headings, #205).

    읽기는 docling(#220, _docx_docling_items — 그림 캡션·중첩 표·자동 번호). 채팅 첨부의 python-docx 읽기(_docx_items)와
    항목 형태가 같아 #205 폴백은 공유한다.

    헤딩 텍스트는 본문에 넣지 않는다. heading_path가 들고 있고, 인덱스 입력엔
    rag/index_text가 앞에 붙이므로 본문에까지 넣으면 같은 문구가 두 번 들어간다.

    조립은 PDF와 같은 `_build_sections`다(#233) — 본문 없는 상위 헤딩은 섹션을 만들지 않고(하위 섹션의 heading_path에 조상으로
    남는다), 본문 없이 같은 레벨 헤딩에 밀린 헤딩('제7조(삭제)')과 문서 끝 헤딩은 본문으로 되살린다. 따로 접던 때는 이 보존이
    PDF에만 있어서 워드에서 본문 없는 조가 색인에서 사라졌다. page는 docx에 없다(None).
    """
    items = list(_docx_docling_items(docling_convert(file_path)))
    return _build_sections([(None, text, level if is_heading else None)
                            for is_heading, level, text, _ in _docx_fallback_headings(items)])


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
        heads: list[str] = []
        blocks: list[str | _TableBlock] = []
        for s in buf:
            extra = s.heading_path[len(common):]   # 공통 조상보다 깊은 헤딩은 본문에 보존
            lines.extend(extra)
            heads.extend(extra)
            pages.extend([s.page] * len(extra))
            blocks.extend(extra)
            blocks.extend(_blocks_of(s))
            lines.append(s.body)
            # 줄별 page를 그대로 이어붙인다. 없으면(docx·md·txt) 섹션 page로 채운다 —
            # 묶음 page를 buf[0].page 하나로만 두면 뒤 섹션 본문이 앞 쪽 번호를
            # 물려받는다 (#137 결함 6의 두 근원 중 하나).
            body_lines = s.body.split('\n')
            if s.line_pages and len(s.line_pages) == len(body_lines):
                pages.extend(s.line_pages)
            else:
                pages.extend([s.page] * len(body_lines))
        packed.append(_Section(common, buf[0].page, '\n'.join(lines), pages,
                               headings=heads, blocks=_join_blocks(blocks)))
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
        from rag.xlsx_chunking import xlsx_plain_text   # 첨부는 openpyxl — docling 변환 프로세스를 색인과 나누지 않는다(#220 결정 6)
        return xlsx_plain_text(str(file_path)).strip()  # 첨부는 행 묶음 없이 시트 전체(#222)
    return _read_text(file_path).strip()  # txt/md/기타


def _txt_sections(file_path: str | Path) -> list[_Section]:
    """평문 txt — 헤딩 개념이 없으므로 파일 전체가 섹션 하나다.

    빈 파일은 섹션 0개로 떨어뜨린다. split_text('')가 ['']를 반환해 빈 청크가
    새는 것을 막는다 (다른 파서의 `if buf:` 가드와 같은 목적).
    """
    text = _read_text(file_path)
    return [_Section([], None, text)] if text.strip() else []


# MD 표 칸에서 docling(2.126)이 망가뜨리는 두 표기 — 변환 전에 유니코드 사용자 영역 문자로 숨겼다가 결과에서 되돌린다(#267).
# 이스케이프 '\|'는 칸 구분으로 읽어 넘친 칸을 **버리고**(실측: '협업 라벨 홀로그램 확인' 칸이 출력 어디에도 없음), '<br>'은 그 행의
# 값을 표 밖 글 한 줄로 내보낸다(구조 손상). 되돌릴 때 '<br>' 변형('<br/>'·'<BR />')은 '<br>' 하나로 적는다 — 뜻이 같다.
_MD_PIPE_MARK, _MD_BR_MARK = '\uE000', '\uE001'
_MD_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)


def _md_unmask(text: str) -> str:
    return text.replace(_MD_PIPE_MARK, '\\|').replace(_MD_BR_MARK, '<br>')


def _md_sections(file_path: str | Path) -> list[_Section]:
    """마크다운 → docling(MD 백엔드) → 섹션(#267). PDF·DOCX와 같은 변환 프로세스(docling_convert — 마감·실패 분류 공유).

    #42의 '#' 정규식 파서를 대신한다. 정규식은 '#' 헤딩·코드 펜스만 알고 나머지를 글로 넘겨, 큰 MD 표가 행 한가운데서 잘리고
    뒤 조각에 열 이름이 없었다(#245 B6). 밑줄식 헤딩('===·---')은 헤딩이 아니었고 닫는 '##'은 제목에 남았다. docling은 마크다운을
    문법대로 읽으므로 표는 _TableBlock(행 묶음·열 이름 반복·#257 표 전체 확장)으로, 헤딩은 문법대로 나온다.
    A/B(main 1750ba2, 합성 27문항 2회): Hit@1 0.852→0.963, EPCov 0.85→0.91, 정규식보다 나빠진 문항 0. 기존 MD 48문항은 노이즈 안.

    docling 출력을 그대로 쓰면 지는 곳이 있어 세 가지를 보정한다(보정 없는 docling은 같은 측정에서 EPCov 0.80):
      ① 표 칸 안 '\\|'·'<br>' — 위 _MD_PIPE_MARK 주석
      ② 링크 — docling은 주소를 글에서 빼 hyperlink 속성에만 둔다. '글(주소)'로 붙인다(주소를 묻는 질문에 답이 없었다)
      ③ 굵게·링크 같은 인라인 서식 — docling은 한 문장을 서식 단위 조각(같은 InlineGroup)으로 쪼갠다. 한 줄로 다시 잇고, 조각
         사이 띄어쓰기는 원문을 보고 정한다('**접수**로만' → '접수로만')
    레벨: '#'은 TitleItem(헤딩 1단), '##'부터는 SectionHeaderItem(level 1부터) → +1 — 정규식 파서의 '#' 개수 레벨과 같다.
    DOCX(_docx_docling_items)는 'title'을 본문으로 보지만 MD의 '#'은 문서 제목이라 헤딩이다. 빈 헤딩(YAML 앞머리)은 버린다.
    본문 없는 헤딩·문서 끝 헤딩 처리는 PDF·DOCX와 같은 _build_sections다(#233). 글자 소실은 테스트가 지킨다
    (tests/test_chunking_md.py — 운영 가드는 두지 않았다, #267 결정 4). 코드 블록은 들여쓰기를 그대로 둔다.

    알려진 한계(#267 리뷰 — 모양만 원문과 다르고 내용은 남아 두었다):
      - ③ 띄어쓰기는 원문에서 조각을 찾아 정한다. 못 찾으면 띄운다 — 원문이 문장 중간에서 줄을 바꾼 경우(docling이 공백으로
        바꿔 준다)와, 표 바로 뒤 조각이 표 안의 같은 낱말을 먼저 맞힌 경우 조사 앞에 빈칸이 생긴다('확인 은')
      - 서식이 섞인 목록 항목은 docling이 번호만 든 항목과 조각으로 나눠 줘 번호가 따로 한 줄이 된다('1.' / '양식(…) 받기')
      - 그림('![대체 글](주소)')은 파일 안에 그림이 없어(주소일 뿐) 캡션하지 않는다 — 대체 글 한 줄과 자리표시가 남는다.
        자리표시 수 지표(count_picture_placeholders)에 MD 그림도 섞인다
      - 큰 MD 표도 PDF·DOCX와 같이 TABLE_MAX_TOKENS를 넘으면 색인하지 않는다(#257 — 정규식 시절에는 글로 색인됐다)
    """
    import tempfile
    from docling_core.types.doc import (CodeItem, InlineGroup, SectionHeaderItem, TableItem, TextItem,   # 지연 import
                                        TitleItem)
    source = _MD_BR_RE.sub(_MD_BR_MARK, _read_text(file_path).replace('\\|', _MD_PIPE_MARK))
    with tempfile.NamedTemporaryFile('w', suffix='.md', encoding='utf-8', delete=False) as tmp:
        tmp.write(source)
    try:
        doc = docling_convert(tmp.name)
    finally:
        os.unlink(tmp.name)

    inside = _table_descendants(doc)
    items: list[_Item] = []
    cursor = 0              # 원문에서 직전 글 조각이 끝난 자리 — ③ 띄어쓰기 판정
    group = None            # 직전 조각이 속한 InlineGroup
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
            if text := _md_unmask(_docling_norm(item.text)):
                items.append((None, text, 1 if isinstance(item, TitleItem) else item.level + 1))
        elif isinstance(item, TableItem):
            if rows := _docx_table_rows(item, doc):
                items.append((None, _TableBlock([[_md_unmask(c) for c in row] for row in rows]), None))
        elif isinstance(item, CodeItem):
            # 코드 블록은 들여쓰기가 구조다(YAML·Python) — 공백을 접는 _docling_norm을 태우지 않는다(#267 리뷰 A)
            if text := _md_unmask('\n'.join(line.rstrip() for line in item.text.strip('\n').split('\n'))):
                items.append((None, text, None))
        else:
            text = docling_block_text(item, doc)
            if isinstance(item, TextItem) and item.hyperlink:
                text = f'{text}({item.hyperlink})'
            if text := _md_unmask(text):
                if inline and inline == group and items and items[-1][2] is None and isinstance(items[-1][1], str):
                    items[-1] = (None, items[-1][1] + (' ' if spaced else '') + text, None)
                else:
                    items.append((None, text, None))
        group = inline
    return _build_sections(items)


# 형식 → 섹션 파서. 이 넷은 _Section으로 수렴하므로 뒤 단계(묶기·분할)를 공유한다.
# xlsx만 빠져 있다 — 행 묶음 + 시트 완성본·크기 상한 초과 시 거절이라 계약이 근본적으로 다르다(#222).
_SECTION_PARSERS = {
    '.pdf': _docling_sections,  # docling — 레이아웃 ML + TableFormer + 원격 OCR(#143)
    '.docx': _docx_sections,    # docling SimplePipeline — Heading 스타일 레벨, 없으면 조문 폴백(#220·#205)
    '.md': _md_sections,        # docling MD 백엔드 + 보정 3개(#267)
    '.txt': _txt_sections,      # 헤딩 없음 — 통째로 섹션 1개
}


def chunk_file(file_path: str | Path, *, description: str = '') -> list[ChunkData]:
    """파일 한 개를 청크 리스트로. **형식 분기는 여기 한 곳뿐이다.**

    분기가 두 곳(여기 + 호출부)에 있던 게 CLI로 xlsx를 넣으면 ZIP 바이너리가 색인되던
    버그의 원인이었다(#42). 호출부는 이제 확장자를 보지 않는다.

    xlsx는 `chunk_xlsx`로 위임한다 — 행 묶음 청크 + 시트 완성본, 크기 상한 초과 시 업로드 거절,
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
    table_keys: dict[tuple[int, int], str] = {}     # (섹션 번호, 섹션 안 블록 번호) → 문서 안 표 순번(#257)
    for si, section in enumerate(_pack_sections(parser(file_path))):
        for text, offset, table in _split_section(section, splitter):
            # 이 조각에 실린 헤딩만 — 색인이 식별자 사전으로, 근거 준비가 구간 경계로 쓴다(#238 결정 15).
            lines = {line.strip() for line in text.split('\n')}
            heads = [h for h in section.headings if h.strip() in lines]
            meta = {'headings': heads} if heads else {}
            if table is not None:
                key = table_keys.setdefault((si, table.block), f't{len(table_keys) + 1}')
                meta |= {'is_table': True, 'table_key': key, 'table_text': table.text}
            out.append(ChunkData(text=text, heading_path=list(section.heading_path),
                                 page=_page_at(section, offset), chunk_index=len(out),
                                 meta=meta or None))
    return out


_TABLE_LEAD_MAX = 150   # 큰 표 묶음마다 붙이는 '표 바로 위 한 줄'의 최대 길이 — 이보다 길면 표 설명이 아니라 본문 문단으로 본다

# 표 전체(table_text)에 함께 싣는 표 앞뒤 글(#268 결정 1·2) — 묶음 청크 본문(검색 입력)은 그대로 두고 모델이 받는 표 전체에만 넣는다.
# 설명이 여러 줄이면 마지막 줄만(B1 '단위: kg당 원' 유실), 150자를 넘으면 아예(B2) 표와 떨어졌고, 표 아래 '※ 단위 ‰' 주석은 다른
# 청크였다(B3) — 모델이 숫자만 보고 단위를 모른 채 답했다(#245 합성 재현: 거절 2·오답 1).
#   앞: 표 바로 위 글의 끝에서 최대 _TABLE_CONTEXT_LINES줄·합계 _TABLE_CONTEXT_CHARS자, 제목 줄을 만나면 멈춘다(#245 진단 도구의
#       '표 앞 최대 3줄'과 같은 기준 — 그래야 그 도구로 0건을 확인할 수 있다). 200자 = 실측 결함을 고치는 최소선(B2 한 문단 196자,
#       B1 세 줄 106자)이자 무관한 글이 섞일 때의 상한(#268 결정 3 — 처음 600자에서 낮춤). 한계: 설명이 200자를 넘으면 안 붙는다.
#       알려진 잡음: 규정의 별표가 부칙 바로 뒤에 오면 부칙 글(시행일 등)이 설명으로 잡힌다(실문서 징계 양정표 184자) — 답을 틀리게
#       할 내용이 아니라 받아들였다(결정 3 = a, '부칙'에서 멈추는 단어 규칙은 두지 않음).
#   뒤: 표 바로 아래 글의 첫 줄부터 TABLE_NOTE 표지로 시작하는 줄이 이어지는 동안 최대 _TABLE_CONTEXT_LINES줄·합계 _TABLE_NOTE_CHARS자
#       (#268 리뷰 1 — 상한이 없으면 긴 '※' 문단이 표 전체를 키우고 크기 검사에 합산돼 업로드 거절까지 바꾼다. 실측 주석은 25자). 표지 없는 문장은
#       잡지 않는다 — '표 아래 첫 줄은 무조건'으로 넓히면 다음 조 본문이 딸려 온다.
# 원래 자리의 줄은 그대로 남는다(복사본이 하나 더 생길 뿐 — 원문 소실 없음).
_TABLE_CONTEXT_LINES = 3
_TABLE_CONTEXT_CHARS = 200
_TABLE_NOTE_CHARS = 100
# '* 항목' 글머리표 목록도 걸린다 — 띄어 쓴 별표 주석('* 재검사는 1회')이 흔해 받아들였다(#268 리뷰 2: 잘못 잡혀도 100자 상한 안의 복사본뿐).
TABLE_NOTE = re.compile(r'^\s*(※|\*(?!\*)|주\s*[)）:：]|[(（]\s*단위|단위\s*[:：]|출처\s*[:：])')   # 표 주석 표지 — eval/chunk_boundary가 import한다


def _table_lead_lines(text: str, headings: set[str]) -> list[str]:
    """표 바로 위 글 블록 → 표 설명 줄(문서 순). 끝에서부터 모으고 제목 줄·줄 수·글자 수 한도에서 멈춘다(#268 결정 2)."""
    out: list[str] = []
    for line in reversed([l.strip() for l in text.split('\n') if l.strip()]):
        if line in headings or is_heading_line(line) or len(out) >= _TABLE_CONTEXT_LINES \
                or sum(map(len, out)) + len(line) > _TABLE_CONTEXT_CHARS:
            break
        out.append(line)
    return out[::-1]


def _table_note_lines(text: str) -> list[str]:
    """표 바로 아래 글 블록 → 주석 줄. 첫 줄부터 TABLE_NOTE 표지로 시작하는 줄이 이어지는 동안만(#268 결정 2)."""
    out: list[str] = []
    for line in (l.strip() for l in text.split('\n') if l.strip()):
        if not TABLE_NOTE.match(line) or len(out) >= _TABLE_CONTEXT_LINES or sum(map(len, out)) + len(line) > _TABLE_NOTE_CHARS:
            break
        out.append(line)
    return out


class _TableRef(NamedTuple):
    """큰 표(묶음 둘 이상) 묶음이 어느 표의 것인가 — chunk_file이 문서 안 표 순번(table_key)으로 바꾼다(#257)."""
    block: int      # 섹션 안 블록 번호
    text: str       # 표 전체 — 표 앞 설명 줄(최대 3줄) + 헤더 한 번 + 전체 행 + 표 아래 주석 줄(#268). 근거 준비가 걸린 묶음을 이것으로 바꾼다


class _Piece(NamedTuple):
    text: str
    offset: int                     # section.body 안의 위치
    table: _TableRef | None = None  # 큰 표의 묶음일 때만


def _check_table_size(block: _TableBlock, table_text: str, page: int | None) -> None:
    """표 하나가 TABLE_MAX_TOKENS를 넘으면 TableTooLargeError — 크기는 모델이 볼 글(table_text)로 잰다(#257 결정 2)."""
    tokens = estimate_tokens_conservative(table_text)
    if tokens > TABLE_MAX_TOKENS:
        header = block.rows[0]
        raise TableTooLargeError(page, header, -(-tokens * 100 // TABLE_MAX_TOKENS), len(block.rows) - 1, len(header))


def _split_section(section: _Section, splitter) -> list[_Piece]:
    """섹션 → 조각(청크 글, 본문 오프셋, 큰 표면 표 정보). **표 블록은 행 묶음으로, 글 블록만 SentenceSplitter로** 나눈 뒤
    블록 사이를 다시 잇는다(#252 결정 1 = b′, 결정 7 = c).

    #252 전에는 표 markdown이 글줄에 섞인 채 분할기를 타서 45행 표가 셀 한가운데(`30,` / `148원 | 5 |`)에서 잘리고 뒷 청크엔
    헤더가 없었다(#246 L2 실측 — XLSX만 행 묶음이라 무사). 이제 표는 batch_table_rows가 행 경계에서만 자르고 묶음마다 헤더를
    반복한다. 그런데 블록마다 따로 자르기만 하면 표 위의 소제목·단위 줄('2. 사이즈 스펙 (단위: cm)')이 혼자 남는 작은 청크가
    되고 표는 그 맥락을 잃었다(실문서 98건 diff — 청크 1,269→1,408). 그래서 두 규칙으로 다시 잇는다:
      - **작은 표(묶음 하나)**: 앞뒤 조각과 합쳐 TABLE_BATCH_CHARS 안이면 한 청크로 — #252 전과 같은 모양.
        같은 글 블록에서 나온 조각끼리는 합치지 않는다(분할기 겹침 50토큰이 중복되므로).
      - **큰 표(묶음 둘 이상)**: 표 바로 위 글의 마지막 줄(_TABLE_LEAD_MAX자 이하)을 모든 묶음 머리에 붙인다 — XLSX 묶음마다
        설명을 반복하는 것과 같은 방식. 큰 표의 묶음은 이웃과 합치지 않는다(이미 예산에 가깝다).
    오프셋은 section.body 안의 위치다 — 쪽 매핑(_page_at)과 목록 번호 복원의 근거. 표 묶음은 첫 데이터 행 위치(행별 쪽).
    큰 표의 묶음에는 표 전체(_TableRef)를 함께 돌려준다(#257) — 작은 표는 이미 한 조각 안에 다 있으므로 붙이지 않는다.
    표 전체에는 표 앞 설명 여러 줄과 표 아래 주석 줄도 싣는다(#268 — _table_lead_lines·_table_note_lines). 묶음 본문은 그대로다.
    표 하나가 TABLE_MAX_TOKENS를 넘으면 TableTooLargeError(#257 결정 2).
    """
    pieces: list[tuple[str, int, int, bool, _TableRef | None]] = []   # (글, 오프셋, 블록 번호, 이웃과 합칠 수 있나, 표 정보)
    cursor = 0
    last_text = ''
    blocks = _blocks_of(section)
    heading_set = {h.strip() for h in section.headings}
    for bi, block in enumerate(blocks):
        rendered = block.render() if isinstance(block, _TableBlock) else block
        start = section.body.find(rendered, cursor)
        start = start if start >= 0 else cursor
        if isinstance(block, _TableBlock):
            batches = batch_table_rows(block.rows[0], block.rows[1:])
            head_len = len(to_markdown_table(block.rows[0], [])) + 1
            pos = start + head_len
            lead = last_text.strip().split('\n')[-1].strip() if last_text.strip() else ''
            lead = lead if len(batches) > 1 and len(lead) <= _TABLE_LEAD_MAX else ''
            if lead and pieces and pieces[-1][0].strip() == lead:
                pieces.pop()        # 그 줄만 든 조각은 묶음마다 머리로 들어가므로 혼자 남기지 않는다(실문서: '[ 별표 1] 징계 양정 기준')
            after = blocks[bi + 1] if bi + 1 < len(blocks) and isinstance(blocks[bi + 1], str) else ''
            table_text = '\n'.join([*_table_lead_lines(last_text, heading_set), rendered, *_table_note_lines(after)])
            _check_table_size(block, table_text, block.row_pages[0] if block.row_pages else section.page)
            ref = _TableRef(bi, table_text) if len(batches) > 1 else None
            for batch in batches:
                pieces.append((f'{lead}\n{batch}' if lead else batch, pos, bi, len(batches) == 1, ref))
                pos += len(batch) - head_len + 1
            last_text = ''
        else:
            inner = start
            for chunk in splitter.split_text(rendered):
                # 청크의 실제 시작 위치로 쪽을 고른다. 섹션 page 하나를 모든 청크에
                # 물려주면 여러 쪽에 걸친 섹션의 뒤쪽 청크가 앞 쪽을 가리킨다 (#137 결함 6).
                # 오프셋은 목록 번호 복원(결함 3)의 판정 근거이기도 하므로 손대기 전에 구한다.
                found = section.body.find(chunk, inner)
                offset = found if found >= 0 else inner
                inner = max(inner, offset)
                text = _strip_orphan_marker(_restore_leading_marker(section.body, offset, chunk))
                if text.strip():
                    pieces.append((text, offset, bi, True, None))
            last_text = rendered
        cursor = start + len(rendered)
    out: list[_Piece] = []
    prev_block, prev_mergeable = None, False
    for text, offset, bi, mergeable, ref in pieces:
        if (out and mergeable and prev_mergeable and bi != prev_block
                and len(out[-1].text) + 1 + len(text) <= TABLE_BATCH_CHARS):
            out[-1] = out[-1]._replace(text=f'{out[-1].text}\n{text}')   # 합칠 수 있는 조각은 큰 표가 아니다(ref 없음)
        else:
            out.append(_Piece(text, offset, ref))
        prev_block, prev_mergeable = bi, mergeable
    return _carry_tail_headings(section, out)


def _carry_tail_headings(section: _Section, pieces: list[_Piece]) -> list[_Piece]:
    """**기록된 헤딩은 조각의 마지막 줄이 될 수 없다** (#238 결정 15-2).

    SentenceSplitter는 헤딩을 모른다. 경계가 헤딩 바로 뒤에 떨어지면 헤딩만 앞 조각에 남고 본문은 다음
    조각으로 가 — 식별자로 그 조를 찾아도 넘길 내용이 없다(실측 1/427, 2026-10-17). 그런 꼬리 헤딩을
    다음 조각 맨 앞으로 옮긴다. 다음 조각이 (겹침 때문에) 이미 그 줄을 갖고 있으면 앞에서만 뗀다.
    #252 이후로는 다음 조각이 표 묶음일 수도 있다 — 헤딩이 표 앞으로 가는 것이므로 같은 규칙이 맞다.
    옮긴 줄이 다음 조각의 머리가 되면 그 조각의 오프셋도 헤딩 위치로 당긴다(쪽 매핑이 헤딩 쪽을 따르게).
    """
    heads = {h.strip() for h in section.headings}
    if not heads:
        return pieces
    out = list(pieces)
    for i in range(len(out) - 1):
        lines = out[i][0].split('\n')
        while lines and not lines[-1].strip():
            lines.pop()
        if not lines or lines[-1].strip() not in heads:
            continue
        head = lines.pop()
        out[i] = out[i]._replace(text='\n'.join(lines))
        nxt, off = out[i + 1].text, out[i + 1].offset
        if head.strip() not in {line.strip() for line in nxt.split('\n')}:
            found = section.body.rfind(head, 0, off + 1)
            out[i + 1] = out[i + 1]._replace(text=head + '\n' + nxt, offset=found if found >= 0 else off)
    return [p for p in out if p.text.strip()]
