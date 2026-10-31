"""청킹 자료형 — 파서 → 묶기 → 크기 분할 → 색인 사이의 계약(ChunkData·Section·TableBlock·Item)."""

from dataclasses import dataclass, field
from typing import NamedTuple

from rag.chunking.tables import to_markdown_table


@dataclass
class ChunkData:
    """청킹 결과 한 조각(임베딩·적재 전)."""

    text: str
    heading_path: list[str]  # 예: ["3. 배송지연 보상", "3.2 지급 기준"]
    page: int | None  # PDF만 채운다
    chunk_index: int  # 문서 내 순서 (0부터)
    meta: dict | None = (
        None  # 큰 표 묶음은 {"is_table", "table_key", "table_text"}, 조각에 실린 하위 헤딩은 {"headings"}
    )


@dataclass
class Section:
    """형식별 파서 → _pack_sections 사이의 내부 계약. 뒤 단계(묶기 → 크기 분할)를 형식과 무관하게 한 번만 쓰게 한다."""

    heading_path: list[str]
    page: int | None  # PDF만 채운다
    body: str
    # body를 '\n'으로 나눈 줄과 1:1인 쪽 번호(PDF만) — 여러 쪽에 걸친 섹션의 뒤쪽 청크가 앞 쪽 번호를 물려받지 않게.
    line_pages: list[int | None] | None = None
    # 묶음(_pack_sections)이 본문 줄로 내려보낸 하위 헤딩(원문 그대로). 색인이 식별자를 추정 없이 뽑고,
    # 분할이 헤딩을 본문과 같은 조각에 두는 근거다. 묶지 않은 섹션은 비어 있다.
    headings: list[str] = field(default_factory=list)
    # 본문 블록 목록 — 글(str) 또는 표(TableBlock). body·line_pages는 여기서 파생된 값이다. None이면 본문 전체가
    # 글 블록 하나다. 크기 분할은 이 목록을 보고 표만 행 단위로 나눈다 — 표를 글줄에서 추측하지 않는다.
    blocks: list['str | TableBlock'] | None = None


class TableBlock(NamedTuple):
    """표 하나 — 첫 행이 헤더인 행 목록. 파서가 글로 바꾸지 않고 이대로 크기 분할까지 운반한다.

    글로 바꾸는 곳(섹션 body·행 묶음 청크)은 둘 다 to_markdown_table을 쓰므로 같은 표는 어디서든 같은 글이다.
    row_pages — 행마다의 쪽 번호(헤더 포함, PDF만). 쪽을 넘겨 이은 표의 뒷 쪽 행이 앞 쪽 번호를 물려받지 않게 한다.
    None이면 블록 전체가 한 쪽.
    """

    rows: list[list[str]]
    row_pages: list[int | None] | None = None

    def render(self) -> str:
        return to_markdown_table(self.rows[0], self.rows[1:])

    def line_pages(self, page: int | None) -> list[int | None]:
        """렌더한 줄(헤더·구분선·행)마다의 쪽 번호 — 구분선은 헤더 쪽을 따른다."""
        rp = self.row_pages or [page] * len(self.rows)
        return [rp[0], rp[0]] + list(rp[1:])


def blocks_of(section: 'Section') -> list['str | TableBlock']:
    return section.blocks if section.blocks is not None else [section.body]


def join_blocks(blocks: list['str | TableBlock']) -> list['str | TableBlock']:
    """이웃한 글 블록을 '\n'으로 합친다 — 글 블록 경계는 의미가 없고(분할기가 다시 나눈다) 표 블록 경계만 의미가 있다."""
    out: list[str | TableBlock] = []
    for b in blocks:
        if isinstance(b, str) and out and isinstance(out[-1], str):
            out[-1] = f'{out[-1]}\n{b}'
        else:
            out.append(b)
    return out


# (page, text, level) — PDF·DOCX·MD 경로가 _build_sections에 흘리는 공통 단위.
# level이 None이면 본문, 정수면 그 레벨의 헤딩. text는 여러 줄일 수 있고 표는 TableBlock으로 온다.
Item = tuple[int | None, 'str | TableBlock', int | None]
