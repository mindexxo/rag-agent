"""검색 인덱스용 텍스트 조립.

청크 본문 앞에 문서 컨텍스트('파일명 > 헤딩 > 헤딩')를 한 줄 붙인 문자열을 만든다.
파일명·heading_path는 지금까지 DB에만 있고 벡터엔 들어가지 않아, 본문에 없는 단어로는
그 청크가 잡히지 않았다 (예: 본문에 '환불'이 없는 «환불정책» 청크).

임베딩(인제스션)과 리랭커(검색 시)가 **같은 형태**를 보도록 조립을 이 한 곳으로 모은다.
- 임베딩: rag/documents.py(워커), rag/ingestion.py(CLI)
- 리랭커: rag/reranker.py

DB의 chunks.text는 원문 그대로 둔다 — 인용·프롬프트 컨텍스트·본문 미리보기가 쓰는 값이라
파생 정보를 섞지 않는다. 프롬프트는 이미 build_context_blocks가 '[파일명 vN] 섹션: ...'
라벨을 따로 찍으므로 본문에까지 넣으면 중복이다.
"""
from pathlib import Path


def build_index_text(text: str, filename: str | None, heading_path: list[str] | None) -> str:
    """본문 앞에 '파일명 > 헤딩 > 헤딩' 한 줄을 붙인다. 붙일 게 없으면 본문 그대로.

    확장자는 뗀다 — '.pdf'/'.docx'는 의미 벡터에 잡음만 얹는다.
    FAQ 청크는 본문이 'Q: ...'로 이미 자기설명적이고 파일 개념이 없어 대상이 아니다
    (호출부가 filename=None으로 넘긴다).

    본문 첫 줄과 겹치는 조각은 뺀다. MarkdownNodeParser는 자기 헤딩을 청크 본문
    첫 줄에 그대로 남기므로(rag/chunking._extract_heading_path), 안 빼면 md 청크마다
    같은 헤딩이 두 번 들어가 본문 대비 헤딩 비중이 부풀고 '헤딩만 매칭되는' 쪽으로 기운다.
    """
    parts = []
    if filename:
        parts.append(Path(filename).stem)
    parts.extend(heading_path or [])

    # 본문 첫 줄(마크다운 '#' 제거)과 정확히 같은 조각은 중복 — 부분일치로 빼면
    # 본문에 스쳐 나온 단어 때문에 상위 섹션 컨텍스트까지 날아가므로 완전일치만.
    # 조각끼리도 중복 제거 — 파일명 == 헤딩('배송지연대응.docx' + '# 배송지연대응') 대비.
    # 실측 0건(2026-08-03, heading 보유 92청크)이라 지금은 값싼 예방책일 뿐. 발생 가능성은
    # DOCX 헤딩 추출이 들어가면 올라간다 — 현재 heading_path는 해시 파일명인 md에만 있어서 0.
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
