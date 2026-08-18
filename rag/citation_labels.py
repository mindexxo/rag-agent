"""인용 후보 번호의 단일 정의점 (#56).

꼬리가 번호 목록(««1,3»»)이 되면서(파일명 라벨 ~25토큰 → ~6토큰, 각주 표시 지연 해소)
"번호 i가 어느 문서인가"의 순서 근거가 곧 계약이다. 같은 순서를 세 곳이 소비한다 —
① 프롬프트 컨텍스트 블록의 [번호](rag/prompts.py), ② guided decoding 문법의 번호 후보
(rag/prompts.build_citation_grammar), ③ 꼬리 파서의 번호→객체 매핑
(rag/citation_tail.resolve_citations). 셋이 각자 순서를 세우면 어긋나는 순간
**오귀속이 조용히**(에러 없이 엉뚱한 문서 인용) 생긴다 — 순서는 여기 한 곳만 만든다.

번호 불변식: sources_from_chunks(chunks)가 만드는 문서 순서 그대로 1부터,
그 뒤에 첨부 파일 순서로 이어진다. 첨부는 SourceCitation이 아니라서(검색 출처가 아님)
후보 목록에서 빠뜨리기 쉽다(#41의 함정) — 문법·파서 양쪽 다 반드시 포함해야 한다.
"""

# 꼬리 마커의 정의점은 rag/prompt_texts.py다(프롬프트가 f-string으로 보간해야 해서 문구와
# 같은 파일). 여기서 재노출만 한다 — 문법·파서는 이 모듈만 보면 된다.
from rag.prompt_texts import TAIL_END, TAIL_START  # noqa: F401 (재노출)
from schemas.kms import SourceCitation


def sources_from_chunks(chunks) -> list[SourceCitation]:
    """검색 청크 목록 → 문서 단위 인용 후보 (번호 순서의 원천).

    문서는 청크 첫 등장 순서(=리랭크 순위 순), FAQ 청크(F3)가 섞여 있으면 'FAQ' 인용
    1건으로 접어 마지막에 둔다 — 원본 파일이 없으므로 document_id 없이 내려가고,
    FE는 document_id 없는 인용을 비클릭으로 처리한다. filename/version은 검색 시점에
    청크에 비정규화돼 있어 DB 재조회가 필요 없다 — 구 _build_sources의 `WHERE id IN`은
    반환 순서가 비보장이라 번호의 순서 근거로 쓸 수 없었다.
    """
    seen: set[int] = set()
    out: list[SourceCitation] = []
    for chunk in chunks:
        if chunk.faq_id is not None or chunk.document_id in seen:
            continue
        seen.add(chunk.document_id)
        out.append(SourceCitation(document_id=chunk.document_id,
                                  filename=chunk.filename, version=chunk.version))
    if any(chunk.faq_id is not None for chunk in chunks):
        out.append(SourceCitation(document_id=None, filename='FAQ', version=1))
    return out


def source_display(source) -> str:
    """검색 출처의 표시명 — 컨텍스트 블록에서 [번호] 뒤에 붙는 사람용 이름 (매칭에 안 쓰임)."""
    return 'FAQ' if source.filename == 'FAQ' else f'{source.filename} v{source.version}'


def attachment_display(filename: str) -> str:
    """첨부 문서의 표시명 — 인용되면 FAQ 선례를 따라 document_id=None 가짜 인용이 된다."""
    return f'첨부: {filename}'
