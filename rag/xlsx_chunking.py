"""xlsx 청킹 (F1a).

시트 하나가 청크 하나로 통째 담긴다. 150행 이하(BGE-M3 임베딩 8192토큰 상한 내 — 확장 재조회 불필요).

## 읽기는 docling, 행 세기는 openpyxl (#220)

- **내용**: docling이 기본(`_docling_sheet_texts`). 보유 xlsx 24건 대조(2026-10-11)에서 셀 누락은 openpyxl·docling
  모두 0이었고, 깨끗한 표(20건)는 내용이 같았다. 차이는 어려운 시트에서 났다 — 안내문 한 줄 + 표 둘인 시트는
  docling이 표 둘을 각자 헤더로 나눴고 openpyxl은 안내문을 헤더로 잡아 두 표를 섞었다. 양식형 실문서 2건도
  docling이 블록을 의미 있는 표로 나눴다. 시트 위 그림은 VLM 캡션으로 들어간다(openpyxl 경로는 그림을 버린다).
  옛 판단 "docling은 헤더행을 가끔 유실"의 정체는 **완전히 빈 행·열을 표 경계로 보는 것**이다(gap_tolerance=0 기본).
  표 중간에 완전히 빈 행이 있으면 표가 갈리고 아랫부분은 첫 데이터 행이 헤더가 된다 — 값은 빠지지 않고 같은 시트
  청크에 이어진다. 한 칸이라도 값이 있는 행은 가르지 않는다(실측). 허용치 1은 이것을 막지만 안내문·양식 시트를
  openpyxl처럼 잘못 읽는다(실문서 2건 모두 해당) — 0을 유지한다(#220 결정 9). FE 업로드 도움말의 '엑셀 작성 규격'
  (시트당 표 1개·첫 행 헤더)은 안내일 뿐 강제하지 않으며, docling이 나았던 3건은 전부 그 규격을 어긴 파일이었다.
- **행 상한**: openpyxl `_iter_sheets` 하나로 센다 — 업로드 검증과 색인이 같은 기준이어야 하고(아래 절), 업로드
  검증이 docling 변환 대기(세마포어)에 서면 안 된다.
- `docling_enabled=False`(비상 스위치)와 채팅 첨부(`chunk_xlsx_plain`)는 openpyxl로 읽는다 — 첨부는 사용자
  대기 경로라 색인과 docling 세마포어를 나누지 않는다(#220 결정 6).

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
from contextlib import closing
from io import BytesIO

import openpyxl

from config import settings

# 표 markdown 조립은 chunking.py가 단일 정의점 — PDF 표(#137 결함 5)도 같은 형태를
# 내놓아야 프롬프트·리랭커·어휘 채널이 형식에 따라 다른 모양을 보지 않는다.
from rag.chunking import ChunkData, docling_block_text, docling_convert, to_markdown_table as _to_markdown

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


def _check_row_limit(title: str, body: list) -> None:
    """행 상한 비교식 — 색인(`chunk_xlsx`)과 사전검증(`validate_xlsx_upload`)이 공유한다."""
    if len(body) > XLSX_MAX_ROWS:
        raise XlsxTooManyRows(title, len(body))


def _iter_sheets(wb):
    """시트별 (시트명, 헤더행, 데이터행들) — 완전 빈 행은 제외한다.

    `chunk_xlsx`(색인)와 `validate_xlsx_upload`(업로드 사전검증)가 **같은 기준으로 세게** 하려고
    뗀 것이다. 행 세는 규칙이 두 벌이 되면 "업로드는 통과했는데 색인에서 거절"(또는 반대)이
    난다 — 빈 행 스킵 하나만 달라도 경계값에서 갈린다.

    **숨긴 시트(hidden·veryHidden)는 읽지 않는다**(#220 리뷰, 서비스 결정 2026-10-11) — 업로드 검증·색인·첨부가 전부
    이 함수를 거치므로 한 곳에서 빼면 셋이 같이 따른다. 숨긴 시트는 행 상한에도 세지 않는다. 숨긴 참조용 시트가
    검색 잡음이 되지 않게 하려는 것이다. (#220 이전 openpyxl 경로는 숨긴 시트도 색인했다.)
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


def _sheet_chunk(title: str, body_text: str, description: str, idx: int) -> ChunkData:
    """시트 하나 → 청크 하나. 두 읽기 경로(docling·openpyxl)가 같은 모양을 내게 하는 단일 조립점."""
    text = f'[{description}]\n{body_text}' if description.strip() else body_text
    return ChunkData(text=text, heading_path=[title], page=None, chunk_index=idx,
                     meta={'is_table': True, 'sheet': title})


def chunk_xlsx_plain(file_path, description: str = '') -> list[ChunkData]:
    """openpyxl 경로 — 시트별 markdown 표 1청크, 첫 행 헤더. 비상 스위치(`docling_enabled=False`)·채팅 첨부용.

    150행 초과 시트가 있으면 XlsxTooManyRows. 그림은 읽지 않는다.
    """
    with closing(openpyxl.load_workbook(file_path, read_only=True, data_only=True)) as wb:
        chunks = []
        for title, header, body in _iter_sheets(wb):
            _check_row_limit(title, body)
            chunks.append(_sheet_chunk(title, _to_markdown(header, body), description, len(chunks)))
        return chunks


def _check_workbook_rows(file_path) -> list[str]:
    """행 상한 검사(openpyxl) — 위반이면 XlsxTooManyRows. 셀이 있는 시트 이름을 문서 순서로 돌려준다."""
    with closing(openpyxl.load_workbook(file_path, read_only=True, data_only=True)) as wb:
        titles = []
        for title, _header, body in _iter_sheets(wb):
            _check_row_limit(title, body)
            titles.append(title)
        return titles


def _docling_sheet_texts(doc) -> dict[str, str]:
    """docling 엑셀 문서 → {시트명: 본문}. 시트는 docling의 'sheet' 그룹이다(그룹 이름 = 시트명, 실측).

    한 시트 안의 블록(표·그림 캡션·표 밖 단일 셀 문구)을 읽기 순서대로 빈 줄로 잇는다 — 시트 하나에 표가 여럿이어도
    청크는 하나다(retriever `_keep_single_table`의 '표 청크 1개 = 시트 1개' 전제 유지). 블록 하나의 글자는
    docling_block_text(표=가로 병합 반복 제거·세로 병합은 행마다, 그림=캡션 또는 자리표시).
    숨긴 시트의 그룹은 INVISIBLE 층이라 기본 순회가 비워 내고, 블록이 없는 시트는 넣지 않는다.
    """
    from docling_core.types.doc import GroupLabel   # 지연 import — docling은 톱레벨에 안 들인다
    sheets: dict[str, str] = {}
    for group in doc.groups:
        if group.label != GroupLabel.SHEET:
            continue
        if blocks := [b for item, _ in doc.iterate_items(root=group) if (b := docling_block_text(item, doc))]:
            sheets[group.name] = '\n\n'.join(blocks)
    return sheets


def chunk_xlsx(file_path, description: str = '') -> list[ChunkData]:
    """시트별 1청크. description(표 설명)이 있으면 각 청크 앞에 병합 — 빈약한 표의 검색 보강.

    150행 초과 시트가 있으면 XlsxTooManyRows — openpyxl로 **먼저** 센다(docling 변환 전, 모듈 docstring).
    웹 업로드는 라우터가 먼저 막으므로 여기까지 오지 않는다("상한 위반이 어디서 걸리는가").
    시트 순서·포함 여부: openpyxl이 셀을 본 시트는 전부 나간다 — docling이 그 시트를 비워 내면 **조용히 빠뜨리지 않고
    실패로 올린다**(strict-grounded, PDF 경로와 같은 규율). 셀 없이 그림만 있는 시트는 docling 결과가 있으면 뒤에 붙인다.
    """
    if not settings.docling_enabled:
        return chunk_xlsx_plain(file_path, description)
    titles = _check_workbook_rows(file_path)
    texts = _docling_sheet_texts(docling_convert(file_path).document)
    if missing := [t for t in titles if t not in texts]:
        raise RuntimeError(f'docling이 셀이 있는 시트를 비워 냈다: {missing}')
    order = titles + [t for t in texts if t not in titles]
    return [_sheet_chunk(t, texts[t], description, i) for i, t in enumerate(order)]
