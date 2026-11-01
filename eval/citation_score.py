"""생성 답변의 인용 정확도 채점 — 답변 끝 출처 꼬리를 운영 배관으로 읽어 기대 문서를 몇 개 인용했는지 잰다."""

import re

from rag.search.retrieval_types import RetrievedChunk
from rag.turn.citation_labels import sources_from_chunks
from rag.turn.citation_tail import TailSplitter, resolve_citations


def citation_match(core: str, stem: str) -> bool:
    """인용 토큰 코어 ↔ 기대 파일명 stem 매칭.

    허용: 정확 일치 / 언더스코어 마지막 조각 일치(kms_03_배송지연대응 ↔ 배송지연대응) /
          4자 이상일 때의 포함 관계 (짧은 토큰의 우연 매칭 오탐 방지).
    """
    if not core or not stem:
        return False
    if core == stem:
        return True
    if core == stem.split('_')[-1] or stem == core.split('_')[-1]:
        return True
    shorter, longer = sorted((core, stem), key=len)
    return len(shorter) >= 4 and shorter in longer


def citation_accuracy(answer: str, expected_docs: list[str], chunks: list[RetrievedChunk]) -> float:
    r"""기대 문서 중 답변이 실제로 인용한 비율 (0..1).

    인용은 답변 끝 출처 꼬리(TAIL_START…TAIL_END, 번호 목록)에서 읽는다. 꼬리 분리·번호 해석은
    운영 배관(TailSplitter → sources_from_chunks → resolve_citations)을 그대로 쓴다 — 채점기가
    배관을 따로 들면 운영과 다른 번호 해석이 생길 수 있다(단일 정의점 원칙). 그래서 운영의 오탐
    복구도 그대로 따른다: `««1»» 추가 설명`(END 뒤 텍스트)이나 END 없는 꼬리는 인용 None이다.
    resolve_citations의 `\d+` 추출이 대괄호·쉼표·여백을 무시하므로 `««1,3»»`·`««[1,3]»»` 모두 같다.
    꼬리가 없으면(형식 미준수) 인용 0으로 집계 — 그게 사실이다.
    eval/rescore_v2.py는 v2 라벨 꼬리 전용이라 이 시그니처로는 동작하지 않는다(의도적).

    - any-match가 아니라 커버리지 비율이다 — any-match는 multi_doc에서 문서 하나만 인용해도 만점이다.
    - FAQ 출처는 'FAQ'로 접는다 — 운영 인용은 항목별 'FAQ: 질문'이지만 이 채점기의 기대 'FAQ'는
      **어느 FAQ든 인용했나**다(골드에 FAQ 문항이 없다). 항목 단위 채점은 eval/faq_citation.py가 faq_id로 한다.
    - 확장자 스트립은 5개 형식 전체.
    """
    if not expected_docs:
        return 0.0
    # 꼬리 분리는 운영 배관 그대로 — 완성된 문자열을 한 청크로 흘려 넣는다.
    # feed/finish의 계약상 부분 스트리밍과 결과가 같다: 보류(_hold)는 finish가 비워내고,
    # 오탐 복구 판정은 누적 버퍼 기준이라 청크 경계와 무관하다.
    splitter = TailSplitter()
    splitter.feed(answer)
    splitter.finish()
    cited = resolve_citations(splitter.tail_raw, sources_from_chunks(chunks), [])
    # 기대 stem과 같은 규칙으로 확장자 제거 — '공지.txt' vs '공지'는 4자 미만이라 포함 매칭도 못 탄다.
    # FAQ 인용은 항목 표시명('FAQ: 질문')이 아니라 'FAQ'로 접는다 — 기대 'FAQ'와 정확 매칭되게.
    cores = ['FAQ' if c.faq_id is not None else re.sub(r"\.(pdf|docx|xlsx|txt|md)$", "", c.filename) for c in cited]
    # 원소가 리스트면 대체 출처 그룹 — 같은 정보가 여러 문서에 있을 때 하나만 인용해도 인정
    groups = [[d] if isinstance(d, str) else d for d in expected_docs]
    covered = 0
    for group in groups:
        stems = [re.sub(r"\.(pdf|docx|xlsx|txt|md)$", "", d) for d in group]
        if any(citation_match(core, stem) for stem in stems for core in cores):
            covered += 1
    return covered / len(groups)
