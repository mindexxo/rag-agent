"""크기 분할 — 섹션을 청크 조각으로. 표는 행 묶음으로, 글은 SentenceSplitter로 나눈 뒤 블록 사이를 다시 잇는다."""

import re
from typing import TYPE_CHECKING, NamedTuple

from rag.chunking.errors import TableTooLargeError
from rag.chunking.sections import restore_leading_marker, strip_orphan_marker
from rag.chunking.tables import TABLE_BATCH_CHARS, TABLE_MAX_TOKENS, batch_table_rows, to_markdown_table
from rag.chunking.types import Section, TableBlock, blocks_of
from rag.identifiers import is_heading_line
from rag.tokens import estimate_tokens_conservative

if TYPE_CHECKING:
    from llama_index.core.node_parser import SentenceSplitter

_TABLE_LEAD_MAX = (
    150  # 큰 표 묶음마다 붙이는 '표 바로 위 한 줄'의 최대 길이 — 이보다 길면 표 설명이 아니라 본문 문단으로 본다
)

# 표 전체(table_text)에 함께 싣는 표 앞뒤 글 — 묶음 청크 본문(검색 입력)은 그대로 두고 모델이 받는 표 전체에만 넣는다
# (단위·설명이 다른 청크로 떨어지면 모델이 숫자만 보고 답한다). 원래 자리의 줄도 그대로 남는다.
#   앞: 표 바로 위 글의 끝에서 최대 _TABLE_CONTEXT_LINES줄·_TABLE_CONTEXT_CHARS자, 제목 줄을 만나면 멈춘다.
#       eval/chunk_boundary의 '표 앞 최대 3줄'과 같은 기준이어야 그 도구로 확인할 수 있다.
#   뒤: 표 바로 아래 글의 첫 줄부터 TABLE_NOTE 표지 줄이 이어지는 동안, 최대 _TABLE_CONTEXT_LINES줄·_TABLE_NOTE_CHARS자 —
#       상한이 없으면 긴 '※' 문단이 표 전체를 키워 크기 검사(TABLE_MAX_TOKENS)까지 바꾼다. 표지 없는 줄은 다음 조 본문이라 잡지 않는다.
_TABLE_CONTEXT_LINES = 3
_TABLE_CONTEXT_CHARS = 200
_TABLE_NOTE_CHARS = 100
# '* 항목' 글머리표 목록도 걸린다 — 띄어 쓴 별표 주석이 흔해 받아들였다(잘못 잡혀도 상한 안의 복사본뿐).
TABLE_NOTE = re.compile(
    r'^\s*(※|\*(?!\*)|주\s*[)）:：]|[(（]\s*단위|단위\s*[:：]|출처\s*[:：])'
)  # 표 주석 표지 — eval/chunk_boundary가 import한다


def _table_lead_lines(text: str, headings: set[str]) -> list[str]:
    """표 바로 위 글 블록 → 표 설명 줄(문서 순). 끝에서부터 모으고 제목 줄·줄 수·글자 수 한도에서 멈춘다."""
    out: list[str] = []
    for line in reversed([ln.strip() for ln in text.split('\n') if ln.strip()]):
        if (
            line in headings
            or is_heading_line(line)
            or len(out) >= _TABLE_CONTEXT_LINES
            or sum(map(len, out)) + len(line) > _TABLE_CONTEXT_CHARS
        ):
            break
        out.append(line)
    return out[::-1]


def _table_note_lines(text: str) -> list[str]:
    """표 바로 아래 글 블록 → 주석 줄. 첫 줄부터 TABLE_NOTE 표지로 시작하는 줄이 이어지는 동안만."""
    out: list[str] = []
    for line in (ln.strip() for ln in text.split('\n') if ln.strip()):
        if (
            not TABLE_NOTE.match(line)
            or len(out) >= _TABLE_CONTEXT_LINES
            or sum(map(len, out)) + len(line) > _TABLE_NOTE_CHARS
        ):
            break
        out.append(line)
    return out


class TableRef(NamedTuple):
    """큰 표(묶음 둘 이상) 묶음이 어느 표의 것인가 — chunk_file이 문서 안 표 순번(table_key)으로 바꾼다."""

    block: int  # 섹션 안 블록 번호
    text: str  # 표 전체 — 앞 설명 줄 + 헤더 한 번 + 전체 행 + 아래 주석 줄. 근거 준비가 걸린 묶음을 이것으로 바꾼다


class Piece(NamedTuple):
    text: str
    offset: int  # section.body 안의 위치
    table: TableRef | None = None  # 큰 표의 묶음일 때만


def _check_table_size(block: TableBlock, table_text: str, page: int | None) -> None:
    """표 하나가 TABLE_MAX_TOKENS를 넘으면 TableTooLargeError — 크기는 모델이 볼 글(table_text)로 잰다."""
    tokens = estimate_tokens_conservative(table_text)
    if tokens > TABLE_MAX_TOKENS:
        header = block.rows[0]
        raise TableTooLargeError(page, header, -(-tokens * 100 // TABLE_MAX_TOKENS), len(block.rows) - 1, len(header))


def split_section(section: Section, splitter: 'SentenceSplitter') -> list[Piece]:
    """섹션 → 조각(청크 글, 본문 오프셋, 큰 표면 표 정보). 표 블록은 행 묶음(batch_table_rows)으로, 글 블록만
    SentenceSplitter로 나눈 뒤 블록 사이를 다시 잇는다 — 따로 자르기만 하면 표 위 소제목·단위 줄이 혼자 남는 작은 청크가 된다:
      - 작은 표(묶음 하나): 이웃 조각과 합쳐 TABLE_BATCH_CHARS 안이면 한 청크로. 같은 글 블록의 조각끼리는 합치지 않는다
        (분할기 겹침이 중복되므로).
      - 큰 표(묶음 둘 이상): 표 바로 위 글의 마지막 줄(_TABLE_LEAD_MAX자 이하)을 모든 묶음 머리에 붙이고, 이웃과 합치지 않는다.
        묶음마다 표 전체(TableRef — 앞 설명·아래 주석 포함)를 함께 돌려준다.
    오프셋은 section.body 안의 위치다(쪽 매핑·목록 번호 복원의 근거). 표 묶음은 첫 데이터 행 위치.
    표 하나가 TABLE_MAX_TOKENS를 넘으면 TableTooLargeError.
    """
    pieces: list[
        tuple[str, int, int, bool, TableRef | None]
    ] = []  # (글, 오프셋, 블록 번호, 이웃과 합칠 수 있나, 표 정보)
    cursor = 0
    last_text = ''
    blocks = blocks_of(section)
    heading_set = {h.strip() for h in section.headings}
    for bi, block in enumerate(blocks):
        rendered = block.render() if isinstance(block, TableBlock) else block
        start = section.body.find(rendered, cursor)
        start = start if start >= 0 else cursor
        if isinstance(block, TableBlock):
            batches = batch_table_rows(block.rows[0], block.rows[1:])
            head_len = len(to_markdown_table(block.rows[0], [])) + 1
            pos = start + head_len
            lead = last_text.strip().split('\n')[-1].strip() if last_text.strip() else ''
            lead = lead if len(batches) > 1 and len(lead) <= _TABLE_LEAD_MAX else ''
            if lead and pieces and pieces[-1][0].strip() == lead:
                pieces.pop()  # 그 줄만 든 조각은 묶음마다 머리로 들어가므로 혼자 남기지 않는다
            after = next((b for b in blocks[bi + 1 : bi + 2] if isinstance(b, str)), '')  # 바로 다음 글 블록만
            table_text = '\n'.join([*_table_lead_lines(last_text, heading_set), rendered, *_table_note_lines(after)])
            _check_table_size(block, table_text, block.row_pages[0] if block.row_pages else section.page)
            ref = TableRef(bi, table_text) if len(batches) > 1 else None
            for batch in batches:
                pieces.append((f'{lead}\n{batch}' if lead else batch, pos, bi, len(batches) == 1, ref))
                pos += len(batch) - head_len + 1
            last_text = ''
        else:
            inner = start
            for chunk in splitter.split_text(rendered):
                # 오프셋은 쪽 매핑과 목록 번호 복원의 근거라 청크를 손대기 전에 원문에서 구한다.
                found = section.body.find(chunk, inner)
                offset = found if found >= 0 else inner
                inner = max(inner, offset)
                text = strip_orphan_marker(restore_leading_marker(section.body, offset, chunk))
                if text.strip():
                    pieces.append((text, offset, bi, True, None))
            last_text = rendered
        cursor = start + len(rendered)
    out: list[Piece] = []
    prev_block, prev_mergeable = None, False
    for text, offset, bi, mergeable, ref in pieces:
        if (
            out
            and mergeable
            and prev_mergeable
            and bi != prev_block
            and len(out[-1].text) + 1 + len(text) <= TABLE_BATCH_CHARS
        ):
            out[-1] = out[-1]._replace(text=f'{out[-1].text}\n{text}')  # 합칠 수 있는 조각은 큰 표가 아니다(ref 없음)
        else:
            out.append(Piece(text, offset, ref))
        prev_block, prev_mergeable = bi, mergeable
    return _carry_tail_headings(section, out)


def _carry_tail_headings(section: Section, pieces: list[Piece]) -> list[Piece]:
    """기록된 헤딩은 조각의 마지막 줄이 될 수 없다 — 꼬리 헤딩을 다음 조각 맨 앞으로 옮긴다.

    분할기는 헤딩을 모르므로 경계가 헤딩 바로 뒤에 떨어지면 식별자로 그 조를 찾아도 본문이 없다. 다음 조각이 (겹침으로)
    이미 그 줄을 가졌으면 앞에서만 뗀다. 옮긴 줄이 머리가 되면 그 조각의 오프셋도 헤딩 위치로 당긴다(쪽 매핑).
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
