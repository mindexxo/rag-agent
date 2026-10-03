"""xlsx 청킹 (F1a).

openpyxl로 직접 읽어 헤더(1행)를 확실히 잡는다 (docling은 헤더행을 가끔 유실).
시트 = 표 1개 규격 전제. 150행 이하면 시트 하나가 청크 하나로 통째 담긴다
(BGE-M3 임베딩 8192토큰 상한 내 — 확장 재조회 불필요).

청크 meta에 {"is_table": True, "sheet": 시트명}을 실어 retriever가
'표 청크'를 식별하고 '한 시트만 참조' 필터를 적용한다.

## 상한 위반이 어디서 걸리는가 — 두 층, 정의점은 하나 (#194)

같은 규칙(`XLSX_MAX_ROWS`·`validate_xlsx_upload`)이 두 시점에 강제된다:

- **웹 업로드**: 라우터가 blob을 쓰기 전에 `validate_xlsx_upload`로 걸러 **400**을 준다.
  사용자가 즉시 사유를 받고, 디스크에 찌꺼기도 DB 행도 남지 않는다.
- **그 밖의 경로**(CLI·`rag.os_reconcile` 재등재·검증 도입 이전에 올라온 문서의 재색인):
  라우터를 안 거치므로 `chunk_xlsx`에서 예외가 올라가 워커가 문서를 **failed**로 굳힌다
  (결정적 실패 — `rag.outbox.is_transient`가 재시도하지 않는다).

두 층이 같은 함수를 거치므로 "검증은 통과했는데 색인에서 거절"되는 드리프트가 없다.
(#139로 파싱이 워커로 옮겨간 뒤 한동안 웹 업로드도 **색인 시점 failed**였다 — 이 모듈
docstring의 옛 "업로드 거절"은 그 기간 동안 사실이 아니었고, #194로 다시 사실이 됐다.)
"""
from io import BytesIO

import openpyxl

# 표 markdown 조립은 chunking.py가 단일 정의점 — PDF 표(#137 결함 5)도 같은 형태를
# 내놓아야 프롬프트·리랭커·어휘 채널이 형식에 따라 다른 모양을 보지 않는다.
from rag.chunking import ChunkData, to_markdown_table as _to_markdown

XLSX_MAX_ROWS = 150   # 헤더 제외 데이터 행. 초과 시 거절 (통째 주입 + 임베딩 상한)


class XlsxUploadRejected(Exception):
    """xlsx 업로드 사전검증 실패의 공통 베이스 (#194) — 라우터는 이것 하나만 잡아 400으로 바꾼다.

    개별 타입을 라우터가 나열하지 않게 하려는 것이다. 사유는 `str(e)`에 들어 있고 그대로
    사용자에게 나간다 — 하위 클래스의 메시지는 사용자가 읽는 문장이다.
    """


class XlsxTooManyRows(XlsxUploadRejected):
    def __init__(self, sheet: str, rows: int):
        self.sheet, self.rows = sheet, rows
        super().__init__(f"시트 '{sheet}'가 {rows}행으로 상한({XLSX_MAX_ROWS})을 초과")


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


def description_missing(description: str | None) -> bool:
    """"표 설명이 비었는가"의 **정의점** — 업로드 사전검증과 PATCH 거절이 같은 판정을 쓴다 (#194).

    None·빈 문자열·공백만을 전부 "없음"으로 본다. `chunk_xlsx`가 병합 여부를 `strip()`으로
    가르므로, 공백만 통과시키면 "필수인데 검색에 아무 기여 없는" 값이 들어온다.
    """
    return not (description or '').strip()


def _check_row_limit(title: str, body: list) -> None:
    """행 상한 비교식 — 색인(`chunk_xlsx`)과 사전검증(`validate_xlsx_upload`)이 공유한다."""
    if len(body) > XLSX_MAX_ROWS:
        raise XlsxTooManyRows(title, len(body))


def _iter_sheets(wb):
    """시트별 (시트명, 헤더행, 데이터행들) — 완전 빈 행은 제외한다.

    `chunk_xlsx`(색인)와 `validate_xlsx_upload`(업로드 사전검증)가 **같은 기준으로 세게** 하려고
    뗀 것이다. 행 세는 규칙이 두 벌이 되면 "업로드는 통과했는데 색인에서 거절"(또는 반대)이
    난다 — 빈 행 스킵 하나만 달라도 경계값에서 갈린다.
    """
    for ws in wb.worksheets:
        rows = [
            [_cell(c.value) for c in row]
            for row in ws.iter_rows()
            if any(c.value is not None for c in row)   # 완전 빈 행 스킵
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

    markdown 조립까지 가는 `chunk_xlsx`를 부르지 않는 이유: 결과를 버릴 거면서 표를 문자열로
    만드는 비용이 **업로드 응답 지연에 그대로 들어간다**. 행만 세고 끝낸다.
    """
    if description_missing(description):
        raise XlsxDescriptionRequired()
    wb = openpyxl.load_workbook(BytesIO(content), read_only=True, data_only=True)
    try:
        for title, _header, body in _iter_sheets(wb):
            _check_row_limit(title, body)
    finally:
        wb.close()


def chunk_xlsx(file_path, description: str = '') -> list[ChunkData]:
    """시트별로 markdown 표 1청크. 시트당 1표·첫 행 헤더 규격 전제.

    description(표 설명)이 있으면 각 청크 앞에 병합 — 빈약한 표의 검색 보강.
    150행 초과 시트가 있으면 XlsxTooManyRows — 웹 업로드는 라우터가 먼저 막으므로 여기까지
    오지 않는다(모듈 docstring "상한 위반이 어디서 걸리는가").
    """
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    chunks, idx = [], 0
    try:
        for title, header, body in _iter_sheets(wb):
            _check_row_limit(title, body)
            md = _to_markdown(header, body)
            text = f'[{description}]\n{md}' if description.strip() else md
            chunks.append(ChunkData(
                text=text,
                heading_path=[title],
                page=None,
                chunk_index=idx,
                meta={'is_table': True, 'sheet': title},
            ))
            idx += 1
    finally:
        wb.close()
    return chunks
