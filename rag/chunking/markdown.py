"""MD 파서 — docling MD 백엔드(같은 변환 프로세스) → Section; 표 칸의 '\\|'·<br> 마스킹 보정."""

import os
import re
from pathlib import Path

from rag.chunking import convert_process, docling_elements, docx, plain, sections
from rag.chunking.types import Item, Section, TableBlock

# MD 표 칸에서 docling이 망가뜨리는 두 표기 — 변환 전에 유니코드 사용자 영역 문자로 숨겼다가 결과에서 되돌린다.
# '\|'는 칸 구분으로 읽어 넘친 칸을 버리고, '<br>'은 그 행의 값을 표 밖 글로 내보낸다. '<br>' 변형은 '<br>' 하나로 되돌린다.
_MD_PIPE_MARK, _MD_BR_MARK = '\ue000', '\ue001'
_MD_BR_RE = re.compile(r'<br\s*/?>', re.IGNORECASE)


def md_unmask(text: str) -> str:
    return text.replace(_MD_PIPE_MARK, '\\|').replace(_MD_BR_MARK, '<br>')


def md_sections(file_path: str | Path) -> list[Section]:
    """마크다운 → docling(MD 백엔드, 같은 변환 프로세스) → 섹션. 표는 TableBlock으로, 헤딩은 문법대로(밑줄식 포함) 나온다.

    docling 출력을 그대로 쓰면 지는 곳 셋을 보정한다:
      ① 표 칸 안 '\\|'·'<br>' — 위 _MD_PIPE_MARK 주석
      ② 링크 — docling은 주소를 hyperlink 속성에만 둔다. '글(주소)'로 붙인다
      ③ 인라인 서식 — docling은 한 문장을 서식 단위 조각(같은 InlineGroup)으로 쪼갠다. 한 줄로 다시 잇고, 조각 사이
         띄어쓰기는 원문을 보고 정한다('**접수**로만' → '접수로만')
    레벨: '#'은 TitleItem(1단), '##'부터는 SectionHeaderItem(level 1부터) → +1. DOCX와 달리 MD의 '#'은 헤딩이다.
    빈 헤딩(YAML 앞머리)은 버리고, 코드 블록은 들여쓰기를 그대로 둔다. 글자 소실은 tests/test_chunking_md.py가 지킨다.

    알려진 한계(모양만 다르고 내용은 남는다):
      - ③ 원문에서 조각을 못 찾으면 띄운다 — 조사 앞에 빈칸이 생길 수 있다('확인 은')
      - 서식이 섞인 목록 항목은 번호가 따로 한 줄이 된다('1.' / '양식(…) 받기')
      - 그림('![대체 글](주소)')은 파일 안에 그림이 없어 캡션하지 않는다 — 대체 글과 자리표시가 남는다
      - 큰 표는 다른 형식과 같이 TABLE_MAX_TOKENS를 넘으면 색인하지 않는다
    """

    import tempfile

    from docling_core.types.doc import (
        CodeItem,
        InlineGroup,
        SectionHeaderItem,
        TableItem,
        TextItem,  # 지연 import
        TitleItem,
    )

    source = _MD_BR_RE.sub(_MD_BR_MARK, plain.read_text(file_path).replace('\\|', _MD_PIPE_MARK))
    with tempfile.NamedTemporaryFile('w', suffix='.md', encoding='utf-8', delete=False) as tmp:
        tmp.write(source)
    try:
        doc = convert_process.docling_convert(tmp.name)
    finally:
        os.unlink(tmp.name)

    inside = docling_elements.table_descendants(doc)
    items: list[Item] = []
    cursor = 0  # 원문에서 직전 글 조각이 끝난 자리 — ③ 띄어쓰기 판정
    group = None  # 직전 조각이 속한 InlineGroup
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
            if text := md_unmask(docling_elements.docling_norm(item.text)):
                items.append((None, text, 1 if isinstance(item, TitleItem) else item.level + 1))
        elif isinstance(item, TableItem):
            if rows := docx.docx_table_rows(item, doc):
                items.append((None, TableBlock([[md_unmask(c) for c in row] for row in rows]), None))
        elif isinstance(item, CodeItem):
            # 코드 블록은 들여쓰기가 구조다 — 공백을 접는 docling_elements.docling_norm을 태우지 않는다
            if text := md_unmask('\n'.join(line.rstrip() for line in item.text.strip('\n').split('\n'))):
                items.append((None, text, None))
        else:
            text = docling_elements.docling_block_text(item, doc)
            if isinstance(item, TextItem) and item.hyperlink:
                text = f'{text}({item.hyperlink})'
            if text := md_unmask(text):
                if inline and inline == group and items and items[-1][2] is None and isinstance(items[-1][1], str):
                    items[-1] = (None, items[-1][1] + (' ' if spaced else '') + text, None)
                else:
                    items.append((None, text, None))
        group = inline
    return sections.build_sections(items)
