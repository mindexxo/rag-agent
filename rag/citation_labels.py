"""인용 라벨 문자열의 단일 정의점 (#56).

같은 라벨 문자열을 세 곳이 소비한다 — ① 프롬프트 컨텍스트 블록(rag/prompts.py),
② guided decoding 문법 후보(rag/prompts.py), ③ 출처 꼬리 파서(rag/citation_tail.py).
셋이 각자 f-string을 들면 표기가 어긋나는 순간 인용 매칭이 조용히 깨진다 — 여기 한 곳만 고친다.

FAQ는 filename='FAQ'인 가짜 SourceCitation(document_id=None)으로 이미 통일돼 있어(retriever
F99 원칙: 컨텍스트 라벨 = 인용 형식) 별도 분기가 없다. 첨부는 SourceCitation이 아니라서
(검색 출처가 아님) 라벨 함수가 따로 있다 — 인용되면 FAQ와 같은 방식의 가짜 인용 객체가 된다
(rag/citation_tail.resolve_citations).
"""

# 꼬리 마커의 정의점은 rag/prompt_texts.py다(프롬프트가 f-string으로 보간해야 해서 문구와
# 같은 파일). 여기서 재노출만 한다 — 문법·파서는 이 모듈만 보면 된다.
from rag.prompt_texts import TAIL_END, TAIL_START  # noqa: F401 (재노출)


def source_label(source) -> str:
    """검색 출처의 인용 라벨 — 컨텍스트 블록 라벨과 같은 문자열이어야 모델이 그대로 복사한다."""
    return '[FAQ]' if source.filename == 'FAQ' else f'[{source.filename} v{source.version}]'


def attachment_label(filename: str) -> str:
    """첨부 문서의 인용 라벨 — 첨부는 prepared.sources에 없어 후보 목록에서 빠뜨리기 쉽다(#41의 함정)."""
    return f'[첨부: {filename}]'
