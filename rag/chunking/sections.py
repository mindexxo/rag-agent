"""섹션 조립과 묶기 — 형식별 파서가 흘린 (쪽, 글/표, 헤딩 레벨) 항목을 섹션으로 조립하고 예산까지 묶는다."""

import re

from rag.chunking.types import Item, Section, TableBlock, blocks_of, join_blocks

# 섹션 묶음 예산(문자, 소프트). 실제로는 pack_sections의 경계 규칙이 먼저 걸리고, 토큰 상한은 SentenceSplitter가 건다.
_PACK_CHARS = 500

# 청크 꼬리에 남은 목록 번호 — strip_orphan_marker·restore_leading_marker가 쓴다.
# 번호 뒤 개행 하나까지 허용한다(번호가 항목과 같은 줄일 때와 독립 줄일 때가 둘 다 있다).
_ORPHAN_MARKER_RE = re.compile(r'(?:^|\n)[ \t]*(\d{1,2}\.|\(\d{1,2}\))[ \t]*\n?$')

# 한국 규정 문서의 장(章) 제목. docling은 장과 조를 같은 레벨로 내놓아 이 패턴으로 층을 복원한다.
CHAPTER_RE = re.compile(r'^제\s*\d+\s*장\b')


# ── 섹션 조립 — PDF·DOCX·MD 공통 후단 ──────────────────────────────────────────────


def build_sections(items: list[Item]) -> list[Section]:
    """(page, text, level) 흐름을 헤딩 스택으로 Section 목록에 접는다. PDF·DOCX·MD 공통 조립.

    스택 규약(`del stack[level-1:]` + append) 위에 셋을 더한다:
      - 줄별 쪽 번호(line_pages) — 여러 줄 text는 줄 수만큼 같은 쪽을 채워 body 줄과 1:1을 지킨다.
      - 본문 없는 헤딩(고아) — 같은 레벨 헤딩에 밀려 지워질 헤딩은 다음 섹션 본문 앞에 문서 순서로 되살린다.
        이미 섹션 경로로 나간 스택 앞부분(emitted)은 고아가 아니다. 한계: 본문 없는 '제3조(삭제)'는 다음 조 본문 앞에
        붙어 경로가 제4조가 된다.
      - 문서 끝 헤딩 — 문서가 본문 없는 헤딩으로 끝나면 그 헤딩을 본문으로 남긴다(경로는 이미 나간 앞부분).
    본문이 하나도 없는 문서는 섹션 0개 — 호출부(rag/ingestion)가 "추출된 텍스트 없음"으로 실패시킨다.
    """
    sections: list[Section] = []
    stack: list[str] = []
    stack_pos: list[int] = []  # stack 원소마다 items 속 위치 — 고아를 되살릴 때 문서 순서로 놓으려고
    orphans: list[tuple[int, str]] = []  # (items 속 위치, 헤딩)
    emitted = 0  # stack 앞에서 몇 개가 이미 섹션 경로로 나갔나 — 그 헤딩들은 고아가 될 수 없다
    buf: list[str] = []
    buf_pages: list[int | None] = []
    buf_blocks: list[str | TableBlock] = []
    start_page: int | None = items[0][0] if items else None
    last_page: int | None = start_page

    def add(text: 'str | TableBlock', page: int | None) -> None:
        if isinstance(text, TableBlock):
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
        sections.append(
            Section(list(path), start_page, '\n'.join(buf), list(buf_pages), blocks=join_blocks(buf_blocks))
        )
        emitted = len(path)
        buf.clear()
        buf_pages.clear()
        buf_blocks.clear()

    for pos, (page, text, level) in enumerate(items):
        if page is not None:
            last_page = page
        if level is not None:
            flush(stack)
            cut = max(emitted, level - 1)  # 걷어낼 헤딩 중 경로로 한 번도 안 나간 것
            orphans.extend(zip(stack_pos[cut:], stack[cut:]))
            emitted = min(emitted, level - 1)
            del stack[level - 1 :], stack_pos[level - 1 :]
            stack.append(text)
            stack_pos.append(pos)
        else:
            # 섹션의 page는 첫 본문 줄의 쪽이다 — 헤딩이 쪽 끝에 걸려도 인용은 본문이 있는 쪽을 가리켜야 한다.
            if not buf:
                start_page = page
                for _, orphan in sorted(orphans):  # 빠진 순서가 아니라 문서 순서 — 상위가 하위보다 늦게 빠질 수 있다
                    add(orphan, page)
                orphans.clear()
            add(text, page)
    if buf:
        flush(stack)
    elif sections:  # 문서가 본문 없는 헤딩으로 끝났다 — 버리지 않고 본문으로(경로는 이미 나간 앞부분)
        start_page = last_page
        for _, heading in sorted([*orphans, *zip(stack_pos[emitted:], stack[emitted:])]):
            add(heading, last_page)
        flush(stack[:emitted])
    return sections


# ── 섹션 묶기 ───────────────────────────────────────────────────────────────────


def _common_prefix(paths: list[list[str]]) -> list[str]:
    """heading_path들의 공통 조상 경로."""
    common = list(paths[0])
    for path in paths[1:]:
        while common != path[: len(common)]:
            common.pop()
    return common


def pack_sections(
    sections: list[Section],
    max_chars: int = _PACK_CHARS,
) -> list[Section]:
    """연속 섹션을 예산까지 묶는다 — 섹션 하나가 청크 하나면 너무 잘아지므로. 묶음의 page는 첫 섹션 것.

    묶음의 heading_path는 참여 섹션들의 공통 조상이고, 그보다 깊은 헤딩은 본문 줄로 남긴다(headings에도 기록).
      예: '2.1 신청 채널'과 '2.2 최소 후원 금액'을 묶으면 heading_path=[문서명, '2. 정기후원 신청'], 본문 앞에 소제목이 한 줄씩.
    묶기를 실제로 막는 건 예산보다 경계다 — joinable이 공통 조상 깊이 2(문서명 아래 같은 대분류)를 요구한다.
    무작정 묶으면 공통 조상이 문서명 하나로 떨어져 heading_path가 무의미해진다.
    size는 body 길이만 세므로(덧붙는 헤딩 줄 제외) max_chars는 소프트 예산이다.
    """
    packed: list[Section] = []
    buf: list[Section] = []
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
        blocks: list[str | TableBlock] = []
        for s in buf:
            extra = s.heading_path[len(common) :]  # 공통 조상보다 깊은 헤딩은 본문에 보존
            lines.extend(extra)
            heads.extend(extra)
            pages.extend([s.page] * len(extra))
            blocks.extend(extra)
            blocks.extend(blocks_of(s))
            lines.append(s.body)
            # 줄별 page를 이어붙인다(없으면 섹션 page) — 묶음 page 하나로 두면 뒤 섹션이 앞 쪽 번호를 물려받는다.
            body_lines = s.body.split('\n')
            if s.line_pages and len(s.line_pages) == len(body_lines):
                pages.extend(s.line_pages)
            else:
                pages.extend([s.page] * len(body_lines))
        packed.append(Section(common, buf[0].page, '\n'.join(lines), pages, headings=heads, blocks=join_blocks(blocks)))
        buf.clear()
        size = 0

    for section in sections:
        if buf and (size + len(section.body) > max_chars or not joinable(section.heading_path)):
            flush()
        buf.append(section)
        size += len(section.body)
    flush()
    return packed


def page_at(section: Section, offset: int) -> int | None:
    """섹션 본문의 문자 오프셋이 실제로 몇 쪽에서 왔는지. line_pages가 없으면 섹션 page."""
    if not section.line_pages:
        return section.page
    position = 0
    for index, line in enumerate(section.body.split('\n')):
        position += len(line) + 1  # '\n' 포함
        if offset < position:
            return section.line_pages[index] if index < len(section.line_pages) else section.page
    return section.line_pages[-1]


def strip_orphan_marker(chunk: str) -> str:
    """청크 꼬리에 목록 번호만 남았으면 떼어낸다(항목 쪽에 되붙이는 것은 restore_leading_marker).

    SentenceSplitter의 기본 secondary_chunking_regex가 마침표를 문장 끝으로 봐서 '3. 항목'을 '3.' / ' 항목'으로 끊는다.
    """
    match = _ORPHAN_MARKER_RE.search(chunk)
    return chunk[: match.start()].rstrip() if match and match.end() == len(chunk) else chunk


def restore_leading_marker(body: str, offset: int, chunk: str) -> str:
    """청크가 목록 번호 바로 뒤에서 시작하면 그 번호를 머리에 되살린다.

    판정은 원문 오프셋으로만 한다 — 청크 앞에 붙어 있던 것이 번호뿐일 때만. overlap으로 번호를 이미 가진 청크는
    offset이 번호보다 앞이라 걸리지 않는다. 청크 글자로 추측하면 엉뚱한 자리에 번호를 중복 주입한다.
    """
    prefix = body[:offset]
    match = _ORPHAN_MARKER_RE.search(prefix)
    if match and match.end() == len(prefix):
        return f'{match.group(1)} {chunk.lstrip()}'
    return chunk
