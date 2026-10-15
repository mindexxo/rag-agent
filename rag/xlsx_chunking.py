"""xlsx 청킹 (F1a → #222 행 묶음 + 시트 확장).

## 행 묶음 청크 + 시트 확장 (#222)

시트 표를 **행 묶음 청크**(약 `XLSX_BATCH_TOKENS`)로 나눠 색인하고, 묶음마다 표 설명과 헤더를 반복한다 — 검색은 작은
묶음으로 정밀하게 걸리게. 묶음마다 **시트 전체 완성본**(`meta['sheet_text']`, 설명·헤더 한 번)을 함께 실어 두고, 검색
후 근거 준비 단계(`rag.retriever.prepare_evidence`)가 걸린 묶음을 그 완성본으로 바꿔 넣는다 — 모델은 언제나 시트
전체를 본다("부분 표" 위험 차단, 기획 F1a 원안). 완성본을 묶음마다 중복 저장하는 대가는 저장 공간뿐이다(묶음 수 ×
시트 크기, 상한 시트 ≈ 6.6천 자 × 약 6). 재색인은 문서 단위로 지우고 다시 쓰므로 완성본과 묶음이 어긋나지 않는다.
시트 하나가 한 묶음에 다 들어가면 청크 본문 = 완성본이다(#222 이전과 같은 모양).

## 크기 상한 — 행 수가 아니라 내용의 양 (#222 결정 7·8·9)

시트 표 하나당 **약 `XLSX_MAX_TABLE_TOKENS`(6,000) 생성 모델 토큰**. 근거(2026-10-12 실측, 14B 운영 설정, 현실형 합성
단가표, 단순 조회 크기마다 60문항): 5,070토큰 행 조회 58/60 · **7,638토큰 60/60** · 10,140토큰 55/60(표 중간 행 15/19)
— 7.6K~10K 사이에서 꺾인다. 처음엔 7,500으로 잡았으나 골드 큰 표 문항(`*_lt*`, 100·200·300행, 2026-10-13)에서
300행 표(실제 5,954토큰)의 4/5 지점 단가를 3,600→36000으로 옮겨 적는 오답이 나와(17/18) 여유를 두어 6,000으로 낮췄다. 행당 토큰은 표 모양에 따라 33배 차이(2열 ≈ 10, 15열 긴 셀 ≈ 165)라 옛 150행 상한은 좁은
표엔 과소, 넓은 표엔 과대였다(15열 긴 셀 150행 ≈ 25K로 임베딩 상한 8,192도 넘어 조용히 잘렸다).
토큰은 **렌더한 표 글자 수 ÷ `CHARS_PER_TOKEN`(1.1)**으로 추정한다 — 실측 글자/토큰의 최소(1.142)보다 낮게 잡아 어떤
표든 추정 ≥ 실제가 되는 보수적 추정이고(근거는 상수 주석), 업로드가 GPU 서버(vLLM /tokenize)에 묶이지 않는다.
화면 도움말의 "셀 글자 수 합계 약 6천 자"는 이 계수로 환산한 값을 내림한 것이다(6,000 × 1.1 = 6,600).
사용자에게는 토큰이라는 말을 쓰지 않는다 — 거절 문구는 허용 크기 대비 %와 현재 행×열(`XlsxTableTooLarge`).
**시트·파일로 나눠 올리라고 안내하지 않는다**: 검색 후 표는 시트 하나만 남아(`rag.retriever`) 나눈 표를 온전히 쓰지 못한다.
상한보다 큰 표는 지금 구조로 받을 방법이 없다(필요해지면 행→문장·text-to-SQL — 기획 문서 승격 단계).

## 읽기는 docling, 행 세기는 openpyxl (#220)

- **내용**: docling이 기본(`_docling_sheet_blocks`). 보유 xlsx 24건 대조(2026-10-11)에서 셀 누락은 openpyxl·docling
  모두 0이었고, 깨끗한 표(20건)는 내용이 같았다. 차이는 어려운 시트에서 났다 — 안내문 한 줄 + 표 둘인 시트는
  docling이 표 둘을 각자 헤더로 나눴고 openpyxl은 안내문을 헤더로 잡아 두 표를 섞었다. 양식형 실문서 2건도
  docling이 블록을 의미 있는 표로 나눴다. 시트 위 그림은 VLM 캡션으로 들어간다(openpyxl 경로는 그림을 버린다).
  옛 판단 "docling은 헤더행을 가끔 유실"의 정체는 **완전히 빈 행·열을 표 경계로 보는 것**이다(gap_tolerance=0 기본).
  표 중간에 완전히 빈 행이 있으면 표가 갈리고 아랫부분은 첫 데이터 행이 헤더가 된다 — 값은 빠지지 않고 같은 시트
  청크에 이어진다. 한 칸이라도 값이 있는 행은 가르지 않는다(실측). 허용치 1은 이것을 막지만 안내문·양식 시트를
  openpyxl처럼 잘못 읽는다(실문서 2건 모두 해당) — 0을 유지한다(#220 결정 9). FE 업로드 도움말의 '엑셀 작성 규격'
  (시트당 표 1개·첫 행 헤더)은 안내일 뿐 강제하지 않으며, docling이 나았던 3건은 전부 그 규격을 어긴 파일이었다.
- **크기 판정**: openpyxl `_iter_sheets`로 렌더한 표 하나로 잰다 — 업로드 검증과 색인이 같은 기준이어야 하고(아래 절),
  업로드 검증이 docling 변환 대기(변환 프로세스)에 서면 안 된다.
- `docling_enabled=False`(비상 스위치, `chunk_xlsx_plain`)와 채팅 첨부(`xlsx_plain_text`)는 openpyxl로 읽는다 — 첨부는
  사용자 대기 경로라 색인과 docling 변환 프로세스를 나누지 않는다(#220 결정 6). 첨부는 묶지 않고 시트 전체 텍스트 그대로다.

청크 meta에 {"is_table": True, "sheet": 시트명, "sheet_text": 시트 완성본}을 싣는다 — retriever가 표 청크를 식별하고
'한 시트만 참조' 필터(시트 = 문서 id + heading_path[0])와 시트 확장을 한다.

## 상한 위반이 어디서 걸리는가 — 두 층, 정의점은 하나 (#194)

같은 규칙(`XLSX_MAX_TABLE_TOKENS`·`validate_xlsx_upload`)이 두 시점에 강제된다:

- **웹 업로드**: 라우터가 blob을 쓰기 전에 `validate_xlsx_upload`로 걸러 **400**을 준다.
  사용자가 즉시 사유를 받고, 디스크에 찌꺼기도 DB 행도 남지 않는다.
- **그 밖의 경로**(CLI·`rag.os_reconcile` 재등재·검증 도입 이전에 올라온 문서의 재색인):
  라우터를 안 거치므로 `chunk_xlsx`에서 예외가 올라가 워커가 문서를 **failed**로 굳힌다
  (결정적 실패 — `rag.outbox.is_transient`가 재시도하지 않는다).

두 층이 같은 함수를 거치므로 "검증은 통과했는데 색인에서 거절"되는 드리프트가 없다.
(#139로 파싱이 워커로 옮겨간 뒤 한동안 웹 업로드도 **색인 시점 failed**였다 — 이 모듈
docstring의 옛 "업로드 거절"은 그 기간 동안 사실이 아니었고, #194로 다시 사실이 됐다.)
"""
import logging
import math
from contextlib import closing
from io import BytesIO

import openpyxl

from config import settings
from rag.tokens import CONSERVATIVE_CHARS_PER_TOKEN, estimate_tokens_conservative

# 표 markdown 조립은 chunking.py가 단일 정의점 — PDF 표(#137 결함 5)도 같은 형태를
# 내놓아야 프롬프트·리랭커·어휘 채널이 형식에 따라 다른 모양을 보지 않는다.
from rag.chunking import (ChunkData, docling_block_text, docling_convert, docling_grid_rows,
                          to_markdown_table as _to_markdown)

logger = logging.getLogger(__name__)

XLSX_MAX_TABLE_TOKENS = 6000   # 시트 표 하나당 생성 모델 토큰 상한 — 근거는 모듈 docstring(#222 결정 7)
# 글자 → 토큰 추정 계수(#222 결정 8) — 정의점은 rag.tokens(근거 예산과 같은 추정기). 처음 고른 1.35는 행 본문만 잰 값
# (1.36~1.46)이라 보수적이지 않았다(tests/test_xlsx_token_estimate가 잡음). 대가: 보통 표(글자/토큰 ≈ 1.35)의 실제 상한이
# 약 4,900토큰으로 준다(6,000 × 1.1 ÷ 1.35).
CHARS_PER_TOKEN = CONSERVATIVE_CHARS_PER_TOKEN
# 행 묶음 하나의 크기. 측정 근거가 없다 — 보유 엑셀은 시트가 대부분 1천 토큰 미만이라 묶음 크기의 효과를 잴 표가 없었다
# (#222 결정 11). 기획 메모의 "20~30행씩 + 헤더 반복"과 같은 규모(5열 표 ≈ 30행)로 잡았다. 큰 표가 실제로 들어오면 잴 것.
XLSX_BATCH_TOKENS = 1000


class XlsxUploadRejected(Exception):
    """xlsx 업로드 사전검증 실패의 공통 베이스 (#194) — 라우터는 이것 하나만 잡아 400으로 바꾼다.

    개별 타입을 라우터가 나열하지 않게 하려는 것이다. 사유는 `str(e)`에 들어 있고 그대로
    사용자에게 나간다 — 하위 클래스의 메시지는 사용자가 읽는 문장이다.
    """


class XlsxTableTooLarge(XlsxUploadRejected):
    """시트 표가 크기 상한을 넘었다(#222). 문구는 사용자 결정 그대로 — 토큰을 말하지 않고, 나눠 올리라고 하지 않는다."""

    def __init__(self, sheet: str, percent: int, rows: int, cols: int):
        self.sheet, self.percent, self.rows, self.cols = sheet, percent, rows, cols
        super().__init__(f"시트 '{sheet}'의 표가 허용 크기의 {percent}%입니다(현재 {rows}행 × {cols}열). "
                         "상담에 필요 없는 열이나 긴 설명을 줄여 다시 올려 주세요.")


class XlsxDescriptionRequired(XlsxUploadRejected):
    """xlsx인데 표 설명이 최종적으로 비었다 (#194).

    '최종적으로'가 핵심이다 — 이번 요청에 안 보냈어도 직전 버전에서 계승되면 통과한다.
    그 판정은 호출부(라우터)가 `rag/documents.py`의 `latest_alive`로 끝낸 뒤 결과만 넘긴다.
    """

    def __init__(self):
        super().__init__('xlsx 업로드에는 표 설명이 필요합니다 (검색 정확도를 위해 필수).')


def _cell(value) -> str:
    """셀 값을 문자열로. None은 빈칸, 나머지는 그대로 (수치는 숫자 형태 유지)."""
    return '' if value is None else str(value)


def _has_value(value) -> bool:
    """셀에 내용이 있는가 — 공백만 든 문자열('', ' ')은 빈칸이다(#220).

    docling도 그런 칸을 내용으로 보지 않는다. 기준이 다르면 openpyxl은 시트를 세고 docling은 비워 내서 문서 전체가
    '시트를 비워 냈다'로 실패한다(리뷰 지적, 재현 확인 2026-10-11).
    """
    return value is not None and str(value).strip() != ''


def description_missing(description: str | None) -> bool:
    """"표 설명이 비었는가"의 **정의점** — 업로드 사전검증과 PATCH 거절이 같은 판정을 쓴다 (#194).

    None·빈 문자열·공백만을 전부 "없음"으로 본다. `chunk_xlsx`가 병합 여부를 `strip()`으로
    가르므로, 공백만 통과시키면 "필수인데 검색에 아무 기여 없는" 값이 들어온다.
    """
    return not (description or '').strip()


def estimate_tokens(text: str) -> int:
    """렌더한 표의 생성 모델 토큰 추정 — rag.tokens.estimate_tokens_conservative(글자 ÷ 1.1, 올림). 추정 ≥ 실제."""
    return estimate_tokens_conservative(text)


def _check_sheet_size(title: str, header: list, body: list) -> None:
    """크기 상한 비교식 — 색인(`chunk_xlsx`)과 사전검증(`validate_xlsx_upload`)이 공유한다(openpyxl 렌더 기준)."""
    tokens = estimate_tokens(_to_markdown(header, body))
    if tokens > XLSX_MAX_TABLE_TOKENS:
        raise XlsxTableTooLarge(title, math.ceil(tokens * 100 / XLSX_MAX_TABLE_TOKENS), len(body), len(header))


def _iter_sheets(wb):
    """시트별 (시트명, 헤더행, 데이터행들) — 완전 빈 행은 제외한다.

    `chunk_xlsx`(색인)와 `validate_xlsx_upload`(업로드 사전검증)가 **같은 기준으로 재게** 하려고
    뗀 것이다. 크기 재는 규칙이 두 벌이 되면 "업로드는 통과했는데 색인에서 거절"(또는 반대)이
    난다 — 빈 행 스킵 하나만 달라도 경계값에서 갈린다.

    **숨긴 시트(hidden·veryHidden)는 읽지 않는다**(#220 리뷰, 서비스 결정 2026-10-11) — 업로드 검증·색인·첨부가 전부
    이 함수를 거치므로 한 곳에서 빼면 셋이 같이 따른다. 숨긴 참조용 시트가
    검색 잡음이 되지 않게 하려는 것이다. (#220 이전 openpyxl 경로는 숨긴 시트도 색인했다.) 크기 상한에도 세지 않는다.
    docling도 숨긴 시트를 INVISIBLE 층에 두어 기본 순회에서 빼므로 두 경로의 시트 목록이 맞는다.
    """
    for ws in wb.worksheets:
        if ws.sheet_state != 'visible':
            continue
        rows = [
            [_cell(c.value) for c in row]
            for row in ws.iter_rows()
            if any(_has_value(c.value) for c in row)   # 완전 빈 행 스킵 — 공백만 든 칸도 빈칸
        ]
        if rows:
            yield ws.title, rows[0], rows[1:]


def validate_xlsx_upload(content: bytes, description: str | None) -> None:
    """xlsx 업로드 사전검증 — **정의점**. 라우터가 blob을 쓰기 전에 부른다 (#194).

    통과하면 None, 아니면 `XlsxUploadRejected` 하위 예외. HTTP 변환은 하지 않는다 —
    이 모듈은 0층 leaf라 starlette을 들이지 않는다(`rag/documents.py`와 같은 규율).

    싼 검사(문자열)부터 — 설명이 비었으면 워크북을 열지 않는다.

    `content`는 디스크를 거치지 않은 업로드 바이트다 — `BytesIO`로 연다(실측 확인:
    openpyxl이 파일류 객체를 그대로 받고 행 수도 경로 입력과 같게 센다, 2026-10-03).
    색인 시점에 워커가 `blob_path`에서 **다시** 연다 — 업로드는 동기 HTTP, 색인은 비동기
    워커라 한 번으로 묶을 수 없다. 열리는 횟수는 둘이지만 판정 기준은 `_iter_sheets` 하나다.

    `chunk_xlsx`를 부르지 않는 이유: docling 변환이 **업로드 응답 지연에 그대로 들어간다**. openpyxl 렌더 글자 수만 잰다.
    """
    if description_missing(description):
        raise XlsxDescriptionRequired()
    with closing(openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)) as wb:
        for title, header, body in _iter_sheets(wb):
            _check_sheet_size(title, header, body)


def _batch_rows(header: list[str], body: list[list[str]], budget_chars: int) -> list[str]:
    """표(헤더 + 행 목록) → 행 묶음 markdown들. 묶음마다 헤더 전체와 구분선을 반복한다.

    **글로 바꾸기 전에 행 목록에서 나눈다**(#222 리뷰) — 렌더한 글을 줄 단위로 다시 해석하던 첫 구현은 칸 안 줄바꿈에서
    헤더가 깨지고(헤더 칸 "구분⏎(원)") 행이 중간에서 잘렸다(데이터 칸 "첫줄⏎| 둘째줄", 8묶음 중 6개). 행 목록 단계에서
    나누면 경계가 항상 행과 행 사이다. 묶음 크기는 렌더한 길이로 잰다(행 하나를 따로 렌더해 더한다 — 행마다 독립적으로
    렌더되므로 정확하다). 행 하나가 예산보다 길면 그 행 혼자 한 묶음이다(자르지 않는다). 데이터 행이 없으면 헤더만.
    """
    head_len = len(_to_markdown(header, []))
    batches, cur, size = [], [], head_len
    for row in body:
        row_len = len(_to_markdown(header, [row])) - head_len      # '\n' + 그 행 한 줄(들)
        if cur and size + row_len > budget_chars:
            batches.append(_to_markdown(header, cur))
            cur, size = [], head_len
        cur.append(row)
        size += row_len
    batches.append(_to_markdown(header, cur))
    return batches


# 시트 블록: ('table', 행 목록[첫 행 = 헤더]) 또는 ('text', 본문) — 표는 행 목록 그대로 들고 다닌다(행 묶음은 행 단계에서 나눈다).
Block = tuple[str, object]


def _render(block: Block) -> str:
    kind, value = block
    return _to_markdown(value[0], value[1:]) if kind == 'table' else value


def _sheet_chunks(title: str, blocks: list[Block], description: str, start: int) -> list[ChunkData]:
    """시트 하나 → 행 묶음 청크들. 두 읽기 경로(docling·openpyxl)가 같은 모양을 내게 하는 단일 조립점(#222).

    blocks: 읽기 순서. 표는 행 묶음으로 나누고, 표가 아닌 블록(그림 캡션·표 밖 문구)은 그대로 한 묶음.
    묶음마다 표 설명을 앞에 붙인다(검색 보강 — 작은 묶음일수록 필요). meta['sheet_text']는 시트 완성본(설명 한 번 +
    블록들을 빈 줄로 이은 것) — 근거 준비 단계가 걸린 묶음을 이것으로 바꾼다.
    """
    prefix = f'[{description}]\n' if description.strip() else ''
    sheet_text = prefix + '\n\n'.join(_render(b) for b in blocks)
    if (est := estimate_tokens(sheet_text)) > XLSX_MAX_TABLE_TOKENS:
        # 크기 판정은 openpyxl 렌더로 한다(업로드·색인이 같은 기준) — docling 완성본은 그림 캡션·세로 병합 반복·설명으로 더
        # 클 수 있다. 거절하면 "업로드 통과·색인 거절"이 생기므로 경고만 남긴다(#222 리뷰). 근거 예산이 안전장치다.
        logger.warning("시트 '%s' 완성본이 상한을 넘는다(추정 %d > %d토큰) — 색인은 계속한다", title, est, XLSX_MAX_TABLE_TOKENS)
    budget = int(XLSX_BATCH_TOKENS * CHARS_PER_TOKEN)
    parts = [p for kind, value in blocks
             for p in (_batch_rows(value[0], value[1:], budget) if kind == 'table' else [value])]
    return [ChunkData(text=prefix + p, heading_path=[title], page=None, chunk_index=start + k,
                      meta={'is_table': True, 'sheet': title, 'sheet_text': sheet_text})
            for k, p in enumerate(parts)]


def _plain_sheets(file_path, *, check_size: bool = True) -> list[tuple[str, list[list[str]]]]:
    """openpyxl로 (시트명, 행 목록[첫 행 = 헤더]) — check_size면 크기 상한 위반 시 XlsxTableTooLarge."""
    with closing(openpyxl.load_workbook(file_path, read_only=True, data_only=True)) as wb:
        out = []
        for title, header, body in _iter_sheets(wb):
            if check_size:
                _check_sheet_size(title, header, body)
            out.append((title, [header, *body]))
        return out


def chunk_xlsx_plain(file_path, description: str = '') -> list[ChunkData]:
    """openpyxl 경로 — 시트별 행 묶음 청크(첫 행 헤더). 비상 스위치(`docling_enabled=False`)용. 그림은 읽지 않는다."""
    chunks: list[ChunkData] = []
    for title, rows in _plain_sheets(file_path):
        chunks += _sheet_chunks(title, [('table', rows)], description, len(chunks))
    return chunks


def xlsx_plain_text(file_path) -> str:
    """채팅 첨부용 — 시트별 표 markdown을 묶지 않고 그대로 잇는다(#222: 첨부는 행 묶음 대상이 아니다).

    색인용 크기 상한은 걸지 않는다(#222 리뷰) — 첨부에는 더 좁은 자체 상한(schemas.kms.ATTACHMENT_MAX_TEXT_CHARS, 6천 자)이
    있고, 라우터가 그 초과를 "필요한 부분만 잘라 첨부" 안내로 돌려준다. 여기서 먼저 막으면 라우터의 일반 예외 처리에 걸려
    "추출할 수 없는 파일"로 사유가 가려졌다.
    """
    return '\n'.join(_render(('table', rows)) for _, rows in _plain_sheets(file_path, check_size=False))


def _check_workbook_size(file_path) -> list[str]:
    """크기 상한 검사(openpyxl) — 위반이면 XlsxTableTooLarge. 셀이 있는 시트 이름을 문서 순서로 돌려준다."""
    return [title for title, _ in _plain_sheets(file_path)]


def _docling_sheet_blocks(doc) -> dict[str, list[Block]]:
    """docling 엑셀 문서 → {시트명: [블록, …]}. 시트는 docling의 'sheet' 그룹이다(그룹 이름 = 시트명, 실측).

    블록은 읽기 순서다. 표는 docling_grid_rows의 행 목록(가로 병합 반복 제거·세로 병합은 행마다), 그 밖은
    docling_block_text의 글자(그림=캡션 또는 자리표시, 표 밖 단일 셀 문구). 한 시트에 표가 여럿이면(시트 1개 = 표 1개는
    안내만 하고 강제하지 않는다, #222 결정 2) 표마다 자기 헤더로 묶음을 만든다. 숨긴 시트의 그룹은 INVISIBLE 층이라 기본
    순회가 비워 내고, 블록이 없는 시트는 넣지 않는다.
    """
    from docling_core.types.doc import GroupLabel, TableItem   # 지연 import — docling은 톱레벨에 안 들인다
    sheets: dict[str, list[Block]] = {}
    for group in doc.groups:
        if group.label != GroupLabel.SHEET:
            continue
        blocks: list[Block] = []
        for item, _ in doc.iterate_items(root=group):
            if isinstance(item, TableItem):
                if rows := docling_grid_rows(item, doc):
                    blocks.append(('table', rows))
            elif text := docling_block_text(item, doc):
                blocks.append(('text', text))
        if blocks:
            sheets[group.name] = blocks
    return sheets


def chunk_xlsx(file_path, description: str = '') -> list[ChunkData]:
    """시트별 행 묶음 청크(#222). description(표 설명)이 있으면 각 묶음 앞에 병합 — 빈약한 표의 검색 보강.

    크기 상한을 넘는 시트가 있으면 XlsxTableTooLarge — openpyxl로 **먼저** 잰다(docling 변환 전, 모듈 docstring).
    웹 업로드는 라우터가 먼저 막으므로 여기까지 오지 않는다("상한 위반이 어디서 걸리는가").
    시트 순서·포함 여부: openpyxl이 셀을 본 시트는 전부 나간다 — docling이 그 시트를 비워 내면 **조용히 빠뜨리지 않고
    실패로 올린다**(strict-grounded, PDF 경로와 같은 규율). 셀 없이 그림만 있는 시트는 docling 결과가 있으면 뒤에 붙인다.
    """
    if not settings.docling_enabled:
        return chunk_xlsx_plain(file_path, description)
    titles = _check_workbook_size(file_path)
    blocks = _docling_sheet_blocks(docling_convert(file_path))
    if missing := [t for t in titles if t not in blocks]:
        raise RuntimeError(f'docling이 셀이 있는 시트를 비워 냈다: {missing}')
    chunks: list[ChunkData] = []
    for t in titles + [t for t in blocks if t not in titles]:
        chunks += _sheet_chunks(t, blocks[t], description, len(chunks))
    return chunks
