"""인용 후보 번호 순서의 정의점 — "번호 i가 어느 출처인가"를 여기 한 곳에서만 만든다.

같은 순서를 컨텍스트 블록의 [번호](rag/prompts.py)·꼬리 강제 제약(build_citation_tail)·꼬리 파서
(rag/citation_tail.resolve_citations)가 소비한다. 셋이 어긋나면 에러 없이 엉뚱한 문서를 인용한다.

번호 불변식: sources_from_chunks(chunks)의 출처 순서대로 1부터, 이어서 첨부 파일 순서. 첨부는 검색 출처가
아니라 빠뜨리기 쉬운데, 꼬리 강제가 걸려 있어 후보를 적게 세면 누락이 아니라 범위 안 다른 문서로의 오답이
된다 — 후보 목록 파생은 PreparedRag.citation_candidates(rag/prepared.py) 한 곳에서 한다.
"""

from typing import NamedTuple

# 꼬리 마커의 정의점은 rag/prompt_texts.py다.
from rag.prompt_texts import TAIL_END, TAIL_START
from schemas.kms import SourceCitation


class CitationCandidates(NamedTuple):
    """인용 후보 — 검색 출처 다음 첨부 순서. `*`로 펼쳐 꼬리 제약과 파서에 같은 순서를 넘긴다."""

    sources: list[SourceCitation]
    attachment_filenames: list[str]


def citation_tail(numbers) -> str:
    """꼬리 문자열의 유일한 조립 통로 — 형식을 손으로 적은 산문·테스트가 형식 변경 때 조용히 낡지 않게.
    프롬프트 쪽은 f-string 보간이라 함수 대신 TAIL_EXAMPLE_* 상수를 쓴다."""
    return f'{TAIL_START}[{",".join(str(n) for n in numbers)}]{TAIL_END}'


def sources_from_chunks(chunks) -> list[SourceCitation]:
    """검색 청크 → 출처 단위 인용 후보 (번호 순서의 원천).

    청크 첫 등장 순서(=리랭크 순위)로 번호를 받는다 — 문서는 document_id 단위, FAQ는 항목(faq_id) 단위다
    (항목별 인용 횟수를 세기 위해). FAQ는 원본 파일이 없어 document_id=None으로 내려가고 FE는 비클릭 처리한다.
    filename/version·FAQ 질문은 청크에 비정규화된 값을 쓴다 — DB 재조회(`WHERE id IN`)는 순서를 보장하지 않는다.
    """
    seen: set[tuple] = set()
    out: list[SourceCitation] = []
    for chunk in chunks:
        key = source_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        if chunk.faq_id is not None:
            out.append(
                SourceCitation(
                    document_id=None, faq_id=chunk.faq_id, filename=faq_display(faq_question(chunk)), version=1
                )
            )
        else:
            out.append(SourceCitation(document_id=chunk.document_id, filename=chunk.filename, version=chunk.version))
    return out


def source_key(item) -> tuple:
    """청크·인용 객체 → 번호 묶음 키. 문서와 FAQ는 id 공간이 겹쳐 종류를 같이 든다."""
    faq_id = getattr(item, 'faq_id', None)
    return ('faq', faq_id) if faq_id is not None else ('doc', item.document_id)


def faq_question(chunk) -> str:
    """FAQ 청크의 질문 — 색인이 heading_path=[질문]으로 적는다(rag/os_index.index_parsed_faq).
    없으면 본문 첫 줄 'Q: …'(rag/faq_indexing.build_faq_chunk_text 형식)에서 읽는다 — 방어용."""
    if chunk.heading_path:
        return chunk.heading_path[0]
    first = (chunk.text or '').split('\n', 1)[0]
    return first[2:].strip() if first.startswith('Q:') else 'FAQ'


def faq_display(question: str) -> str:
    """'종류: 이름' 꼴(첨부와 같음)이라 FE가 접두로 종류를 안다. 질문은 줄이지 않는다 — 축약 표시는 FE 몫."""
    return f'FAQ: {question}'


def source_display(source) -> str:
    """컨텍스트 블록의 [번호] 뒤에 붙는 사람용 이름(매칭에 안 쓰임). 청크와 SourceCitation 둘 다 받는다."""
    if getattr(source, 'faq_id', None) is not None:
        return faq_display(faq_question(source)) if hasattr(source, 'text') else source.filename
    return f'{source.filename} v{source.version}'


def attachment_display(filename: str) -> str:
    """첨부 문서의 표시명 — 인용되면 document_id=None 인용 객체가 된다(FAQ와 같은 방식)."""
    return f'첨부: {filename}'
