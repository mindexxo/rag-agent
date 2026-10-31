"""표 직렬화·행 묶음 — 전 형식(PDF·DOCX·MD·XLSX)이 같은 표를 같은 글로 만들게 하는 정의점."""

from rag.tokens import CONSERVATIVE_CHARS_PER_TOKEN

# 표 행 묶음 하나의 크기 — 전 형식 공통. 글자 수 환산은 보수적 계수(rag/tokens).
TABLE_BATCH_TOKENS = 1000
TABLE_BATCH_CHARS = int(TABLE_BATCH_TOKENS * CONSERVATIVE_CHARS_PER_TOKEN)

# 표 하나당 토큰 상한 — 전 형식 공통. 근거 준비(rag/search/retriever.prepare_evidence)는 표를 통째로 넣거나 빼므로 넘는 표는
# 일부만 보여 줄 수밖에 없고, 그러면 모델이 나머지 행을 지어낸다. 그래서 받지 않는다(엑셀은 업로드 거절, PDF·DOCX·MD는 색인 실패).
# 값은 답의 정확도로 정했다 — 근거 예산(EVIDENCE_BUDGET_TOKENS)보다 작아야 하고, 생성 모델을 바꾸면 다시 잰다.
TABLE_MAX_TOKENS = 6000


def batch_table_rows(header: list[str], body: list[list[str]], budget_chars: int = TABLE_BATCH_CHARS) -> list[str]:
    """표 → 행 묶음 markdown들, 묶음마다 헤더 반복. 표 행 묶음의 형식 공통 정의점(xlsx가 import한다).

    글로 바꾸기 전에 행 목록에서 나누므로 경계는 항상 행 사이다(렌더한 글을 다시 나누면 칸 안 줄바꿈에서 깨진다).
    크기는 렌더한 길이로 잰다. 예산보다 긴 행은 혼자 한 묶음(자르지 않는다), 데이터 행이 없으면 헤더만.
    """
    head_len = len(to_markdown_table(header, []))
    batches: list[str] = []
    cur: list[list[str]] = []
    size = head_len
    for row in body:
        row_len = len(to_markdown_table(header, [row])) - head_len  # '\n' + 그 행 한 줄(들)
        if cur and size + row_len > budget_chars:
            batches.append(to_markdown_table(header, cur))
            cur, size = [], head_len
        cur.append(row)
        size += row_len
    batches.append(to_markdown_table(header, cur))
    return batches


def to_markdown_table(header: list[str], body: list[list[str]]) -> str:
    """헤더 + 데이터행 → markdown 표. 표 직렬화의 정의점(xlsx가 import한다).

    빈 셀은 자리를 지킨다 — 지우면 뒷 열이 앞으로 당겨져 값이 다른 열로 읽힌다. 행은 헤더 길이에 맞춰 채우거나 자른다.
    """
    lines = ['| ' + ' | '.join(header) + ' |', '| ' + ' | '.join('---' for _ in header) + ' |']
    for row in body:
        cells = (row + [''] * len(header))[: len(header)]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)
