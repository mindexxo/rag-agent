"""검색 인덱스용 텍스트 조립 — 청크 본문 앞에 '파일명 > 헤딩' 한 줄을 붙여, 본문에 없는 단어로도 잡히게 한다.

임베딩(rag/ingestion.py)·리랭커(rag/reranker.py)·어휘 필드(rag/os_index.py)가 같은 형태를 보도록 조립을 여기 한 곳에 둔다.
`folder`(폴더 설명)는 리랭커에만 넣는다 — 임베딩에 넣으면 폴더 안 청크 벡터가 서로 가까워지고, 설명을 고칠 때마다 재색인이다.
저장되는 청크 본문은 원문 그대로다(인용·프롬프트가 쓰는 값).
"""

from pathlib import Path


def build_index_text(text: str, filename: str | None, heading_path: list[str] | None, folder: str | None = None) -> str:
    """본문 앞에 '[폴더] > 파일명 > 헤딩 > 헤딩' 한 줄을 붙인다. 붙일 게 없으면 본문 그대로.

    확장자는 뗀다(의미 벡터의 잡음). 본문 첫 줄과 같은 조각·조각끼리의 중복은 뺀다.
    """
    parts = []
    if folder:  # 리랭커 전용 (모듈 docstring)
        parts.append(folder.strip())
    if filename:
        parts.append(Path(filename).stem)
    parts.extend(heading_path or [])

    # 완전일치만 뺀다 — 부분일치로 빼면 본문에 스친 단어 때문에 상위 섹션 컨텍스트까지 날아간다.
    first_line = text.lstrip().split('\n', 1)[0].lstrip('#').strip()
    unique = []
    for part in parts:
        part = part.strip()
        if part and part != first_line and part not in unique:
            unique.append(part)
    parts = unique

    if not parts:
        return text
    return ' > '.join(parts) + '\n' + text
