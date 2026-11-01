"""xlsx 테스트 입력을 메모리에서 만드는 순수 헬퍼 — 청커 단위 테스트와 업로드 통합 테스트가 함께 쓴다."""

from collections.abc import Callable
from io import BytesIO

import openpyxl

from rag.chunking import TABLE_MAX_TOKENS, to_markdown_table
from rag.tokens import estimate_tokens_conservative

XLSX_MIME = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'


def max_rows_at_limit() -> int:
    """헤더 'col' + 숫자 1열 표에서 추정 토큰이 상한 이하인 최대 데이터 행 수 — 경계 테스트용(off-by-one 회귀 방지)."""
    n = 1
    while (
        estimate_tokens_conservative(to_markdown_table(['col'], [[str(i)] for i in range(n + 1)])) <= TABLE_MAX_TOKENS
    ):
        n += 1
    return n


def xlsx_bytes(n_rows: int, *, blank_in_middle: int = 0) -> bytes:
    """헤더 1행 + 데이터 n_rows행을 메모리에서 만든다 — 라우터가 받는 것과 같은 '디스크 안 거친 bytes'."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['col'])
    for i in range(n_rows):
        ws.append([i])
        if blank_in_middle and i == n_rows // 2:
            for _ in range(blank_in_middle):
                ws.append([None])
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


def wb_bytes(build: Callable[[openpyxl.Workbook], None]) -> bytes:
    """빈 통합 문서를 build로 채워 bytes로 돌려준다."""
    wb = openpyxl.Workbook()
    build(wb)
    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()
