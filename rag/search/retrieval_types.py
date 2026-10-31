"""검색 결과 자료형 — `RetrievedChunk`·`RetrievalResult`·`RetrievalCandidates`·`DenseHits`·`GateVerdict`의 정의점.

엔진 쪽(os_search)과 조합 쪽(retriever)이 서로를 import하지 않고 같은 자료형을 쓰도록 0층에 둔다. 로직은 두지 않는다.
"""

from dataclasses import dataclass, field
from typing import NamedTuple


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int | None  # 부모 문서 id (FAQ 청크는 None)
    text: str  # 청크 본문 — LLM 컨텍스트로 전달
    heading_path: list[str]
    page: int | None
    filename: str  # FAQ 청크는 'FAQ' — 컨텍스트 라벨·인용 표기가 이 값을 따름
    version: int
    faq_id: int | None = None  # FAQ 출처면 항목 id (캐시 키·인용 분기용)
    is_table: bool = False  # 표 묶음 여부 — 근거 준비가 접고 확장한다
    table_key: str | None = (
        None  # 문서 안에서 어느 표인가 — 엑셀은 시트명, PDF·DOCX는 표 순번('t1'…). 접기 키 = (document_id, table_key)
    )
    table_text: str | None = None  # 표 전체 — 근거 준비가 '걸린 묶음 + 표 전체'로 바꿀 때 쓴다
    folder_name: str | None = None
    folder_description: str | None = None  # 리랭커 입력에만 사용 (임베딩엔 미포함)
    # 식별자 채널 색인 필드 그대로 — 가산 판정·구간 오리기가 읽는다.
    identifiers: list[str] = field(default_factory=list)  # '11조'·'43조의2'·'3장'
    headings: list[str] = field(default_factory=list)  # 묶음이 본문으로 내려보낸 헤딩 원문
    matched_ident: str | None = (
        None  # 질문의 식별자와 일치해 가산을 받았으면 그 토큰 — prepare_evidence가 이걸 보고 오린다
    )


# query_embedding(아래 두 자료형 공통) — 원본 쿼리의 dense 벡터. 한 턴의 쿼리 벡터는 하나여야 해서(TEI는 호출마다
# 미세하게 비결정적) rag/turn/cache.py가 다시 임베딩하지 않고 이걸 재사용한다. 확장 변형 벡터는 담지 않는다.
# repr=False: 1024차원 float가 트레이스백·로그에 통째로 새는 것을 막는다.


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]  # no_evidence여도 비어있지 않을 수 있다
    no_evidence: bool
    reason: str | None  # 'no_results' | 'low_similarity' | None
    # 판정과 무관하게 항상 실어 나른다 (사유는 위 주석 블록)
    query_embedding: list[float] | None = field(default=None, repr=False)


@dataclass
class RetrievalCandidates:
    """게이트 적용 전 후보 묶음 — eval이 이 단계를 직접 들여다본다.

    top_dense_distance: 원본 쿼리의 top-1 dense 거리(후보 없으면 999.0) — 근거 게이트의 유일한 입력.
    리랭크와 무관하게 산정되므로 이 신호가 가리키는 청크가 chunks에 없을 수 있다(sweep 해석 시 주의).
    """

    chunks: list[RetrievedChunk]
    top_dense_distance: float
    query_embedding: list[float] | None = field(default=None, repr=False)


class DenseHit(NamedTuple):
    chunk_id: int
    distance: float  # cosine distance


class DenseHits(NamedTuple):
    per_query_ids: list[list[int]]  # 쿼리별 kNN id(원본 먼저)
    top: list[DenseHit]  # 원본 쿼리 결과만 — 변형 쿼리가 게이트를 흔들지 않게


class GateVerdict(NamedTuple):
    no_evidence: bool
    reason: str | None  # 'no_results' | 'low_similarity' | None
