"""xlsx 청킹 (F1a).

openpyxl로 직접 읽어 헤더(1행)를 확실히 잡는다 (docling은 헤더행을 가끔 유실).
시트 = 표 1개 규격 전제. 150행 이하면 시트 하나가 청크 하나로 통째 담긴다
(BGE-M3 임베딩 8192토큰 상한 내 — 확장 재조회 불필요).

청크 meta에 {"is_table": True, "sheet": 시트명}을 실어 retriever가
'표 청크'를 식별하고 '한 시트만 참조' 필터를 적용한다.
"""
import openpyxl

# 표 markdown 조립은 chunking.py가 단일 정의점 — PDF 표(#137 결함 5)도 같은 형태를
# 내놓아야 프롬프트·리랭커·어휘 채널이 형식에 따라 다른 모양을 보지 않는다.
from rag.chunking import ChunkData, to_markdown_table as _to_markdown

XLSX_MAX_ROWS = 150   # 헤더 제외 데이터 행. 초과 시 거절 (통째 주입 + 임베딩 상한)


class XlsxTooManyRows(Exception):
    def __init__(self, sheet: str, rows: int):
        self.sheet, self.rows = sheet, rows
        super().__init__(f"시트 '{sheet}'가 {rows}행으로 상한({XLSX_MAX_ROWS})을 초과")


def _cell(value) -> str:
    """셀 값을 문자열로. None은 빈칸, 나머지는 그대로 (수치는 숫자 형태 유지)."""
    return '' if value is None else str(value)


def chunk_xlsx(file_path, description: str = '') -> list[ChunkData]:
    """시트별로 markdown 표 1청크. 시트당 1표·첫 행 헤더 규격 전제.

    description(표 설명)이 있으면 각 청크 앞에 병합 — 빈약한 표의 검색 보강.
    150행 초과 시트가 있으면 XlsxTooManyRows (업로드 거절).
    """
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    chunks, idx = [], 0
    try:
        for ws in wb.worksheets:
            rows = [
                [_cell(c.value) for c in row]
                for row in ws.iter_rows()
                if any(c.value is not None for c in row)   # 완전 빈 행 스킵
            ]
            if not rows:
                continue
            header, body = rows[0], rows[1:]
            if len(body) > XLSX_MAX_ROWS:
                raise XlsxTooManyRows(ws.title, len(body))

            md = _to_markdown(header, body)
            text = f'[{description}]\n{md}' if description.strip() else md
            chunks.append(ChunkData(
                text=text,
                heading_path=[ws.title],
                page=None,
                chunk_index=idx,
                meta={'is_table': True, 'sheet': ws.title},
            ))
            idx += 1
    finally:
        wb.close()
    return chunks
