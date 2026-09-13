"""검색 결과 자료형 — `RetrievedChunk`·`RetrievalResult`·`RetrievalCandidates`의 정의점.

0층 leaf다(`dataclasses`만 쓴다). `rag/retriever.py`에서 뽑아낸 것은(#146) 검색 엔진 쪽과
조합 쪽이 **같은 자료형을 공유하면서도 서로를 import하지 않게** 하기 위해서다: 분리 전에는
엔진 쪽(구 rag/opensearch.py)이 `RetrievedChunk`를 만들려고 `rag/retriever.py`를 되돌아
가리켜야 했고(retriever가 엔진 모듈을 톱레벨로 물고 있었다), 그 순환을 함수 안 지연 import로
피하고 있었다. 자료형이 여기 있으면 엔진 쪽과 `rag/retriever.py` 둘 다 톱레벨에서 그냥 가져온다.

자료형만 둔다 — 판정·조립 로직은 `rag/retriever.py`다.
"""
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    """검색 결과 1건. RagService가 인용 메타 + LLM 컨텍스트 구성에 사용."""
    chunk_id: int
    document_id: int | None         # 부모 문서 id (FAQ 청크는 None)
    text: str                       # 청크 본문 — LLM 컨텍스트로 전달
    heading_path: list[str]         # 인용 표시용 ["3. 보상", "3.2 지급기준"]
    page: int | None                # PDF 페이지 (있으면)
    filename: str                   # FAQ 청크는 'FAQ' — 컨텍스트 라벨·인용 표기가 이 값을 따름
    version: int
    faq_id: int | None = None       # FAQ 출처면 항목 id (캐시 키·인용 분기용)
    is_table: bool = False          # F1a: xlsx 표 청크 여부 ('한 시트만' 필터용)
    folder_name: str | None = None          # 소속 폴더 (미분류·FAQ는 None)
    folder_description: str | None = None   # 폴더 '참조 설명' — 리랭커 입력에만 사용 (임베딩엔 미포함)

# 아래 두 자료형이 함께 실어 나르는 필드 — 검색 "결과"가 아니라 입력의 파생물이다 (#50).
#
# 원본 쿼리(queries[0])의 dense 벡터. 한 턴 안에서 같은 문자열을 검색·캐시 조회·캐시 저장이
# 각자 임베딩해 TEI를 3번 때리고 있었고, TEI가 호출마다 비결정적(실측 1.4e-4)이라 세 벡터가
# 서로 미세하게 달랐다 — "한 턴 = 하나의 쿼리 벡터"라는 불변식이 코드로 보장되지 않았다.
# 여기 담아 rag/cache.py가 재사용하게 만든다. 재사용이 안전한 근거는 cache.get_semantic 참조.
#
# 확장 변형(index 1+)은 담지 않는다 — 검색·RRF 융합 전용이고 캐시 키와 무관하다.
# repr=False: 1024차원 float가 예외 트레이스백·로그에 통째로 새는 것을 막는다.

@dataclass
class RetrievalResult:
    """검색 최종 결과. no_evidence=True면 LLM 호출 건너뜀."""
    chunks: list[RetrievedChunk]    # top-K 결과 (no_evidence여도 비어있지 않을 수 있음)
    no_evidence: bool               # True면 근거 부족으로 판정
    reason: str | None              # 'no_results' (아예 빈 결과) |
                                    # 'low_similarity' (top-1 거리 임계값 초과) |
                                    # None (정상)
    # 원본 쿼리 dense 벡터 — 판정과 무관한 pass-through (#50, 사유는 위 주석 블록)
    query_embedding: list[float] | None = field(default=None, repr=False)

@dataclass
class RetrievalCandidates:
    """게이트 적용 전 후보 묶음. 평가 스크립트가 이 단계를 직접 들여다봄.

    - chunks: 최종 정렬 순 top_n 후보 (리랭크 on이면 리랭크 후 순서)
    - top_dense_distance: 원본 쿼리의 top-1 dense distance — 근거 게이트의 유일한 입력 신호.
                          후보 없으면 999.0 (기존 'no_results' 분기와 동일 의미)
                          **운영은 게이트가 꺼져 있어(GATE_DISABLED) 이 값을 쓰지 않는다.**
                          실사용처는 eval의 threshold sweep(eval/gate.py·retrieval.py)뿐이다.
                          리랭크·표 필터와 무관하게 산정되므로, 이 신호가 가리키는 청크가
                          최종 chunks에 없을 수도 있다 (sweep 해석 시 주의).
    - query_embedding: 원본 쿼리 dense 벡터 (#50). retrieve()가 RetrievalResult로 그대로
                       옮긴다 — 사유는 위 주석 블록 참조.
    """
    chunks: list[RetrievedChunk]
    top_dense_distance: float
    query_embedding: list[float] | None = field(default=None, repr=False)
