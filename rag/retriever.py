"""dense-only 검색 + 리랭커 + 근거 게이트 (F99 — 하이브리드 원복 청사진은 본문 주석 보관).

흐름:
    query
      -> embed_query (async) -> q_dense
      -> dense top-N (cosine distance)   -- 의미 매칭 (BGE-M3)
      -> 리랭커 재정렬 (cross-encoder, settings.rerank_enabled)
      -> top-N 후보 + 본문 fetch          == retrieve_candidates()
      -> 근거 게이트 (top-1 거리 임계값)   == apply_gate()
      -> RetrievalResult                  == retrieve() 가 위 둘을 조합

평가(1.5.B)는 retrieve_candidates / apply_gate 를 직접 호출해 Recall@N·
gate 전/후·threshold sweep 을 본다. 운영 /kms/query 는 retrieve() wrapper
사용 — 동작 불변.

LLM 호출 없음. Stage D의 RagService가 이 결과를 받아 답변 생성 또는
"확인 불가" 응답으로 분기.
"""
from dataclasses import dataclass

from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag.embeddings import embed_query
from rag.models import Chunk, Document, Faq, Folder

# 검색 대상 판정 — 출처별 분기 (chunks는 document/faq 다형성, F3):
#   문서 청크: 활성 버전 + 인덱싱 완료 + 문서 on + (미분류 OR 폴더 on)
#   FAQ 청크 : 항목 on
def _searchable_condition():
    return or_(
        and_(
            Chunk.document_id.is_not(None),
            Document.is_active.is_(True),
            Document.status == 'ready',
            Document.is_searchable.is_(True),
            or_(Document.folder_id.is_(None), Folder.is_searchable.is_(True)),
        ),
        and_(Chunk.faq_id.is_not(None), Faq.is_active.is_(True)),
    )


# ===== 결과 자료형 ==================================================

@dataclass
class RetrievedChunk:
    """검색 결과 1건. RagService가 인용 메타 + LLM 컨텍스트 구성에 사용."""
    chunk_id: int
    document_id: int | None         # 부모 문서 id (FAQ 청크는 None)
    text: str                       # 청크 본문 — LLM 컨텍스트로 전달
    heading_path: list[str]         # 인용 표시용 ["3. 보상", "3.2 지급기준"]
    page: int | None                # PDF 페이지 (있으면)
    rrf_score: float                # RRF 합산 점수 (높을수록 관련)
    branches: list[str]             # ['dense'] | ['sparse'] | ['dense','sparse']  어느 검색에서 잡혔는지 — 디버깅·로깅용
    filename: str                   # FAQ 청크는 'FAQ' — 컨텍스트 라벨·인용 표기가 이 값을 따름
    version: int
    faq_id: int | None = None       # FAQ 출처면 항목 id (캐시 키·인용 분기용)
    is_table: bool = False          # F1a: xlsx 표 청크 여부 ('한 시트만' 필터용)
    folder_name: str | None = None          # 소속 폴더 (미분류·FAQ는 None)
    folder_description: str | None = None   # 폴더 '참조 설명' — 리랭커 입력에만 사용 (임베딩엔 미포함)

@dataclass
class RetrievalResult:
    """검색 최종 결과. no_evidence=True면 LLM 호출 건너뜀."""
    chunks: list[RetrievedChunk]    # top-K 결과 (no_evidence여도 비어있지 않을 수 있음)
    no_evidence: bool               # True면 근거 부족으로 판정
    reason: str | None              # 'no_results' (아예 빈 결과) |
                                    # 'low_similarity' (top-1 거리 임계값 초과) |
                                    # None (정상)

@dataclass
class RetrievalCandidates:
    """게이트 적용 전 후보 묶음. 평가 스크립트가 이 단계를 직접 들여다봄.

    - chunks: RRF 순 정렬된 top_n 후보 (Recall@N·threshold sweep용)
    - top_dense_distance: top-1 dense distance — 근거 게이트의 유일한 입력 신호
                          후보 없으면 999.0 (기존 'no_results' 분기와 동일 의미)
    """
    chunks: list[RetrievedChunk]
    top_dense_distance: float


# ===== RRF (Reciprocal Rank Fusion) =================================

def _rrf_fuse(
    dense_ids: list[int],
    sparse_ids: list[int],
    k: int = 60,
) -> dict[int, float]:
    """[보관 — 현재 미호출] 하이브리드 원복 시 사용. 두 검색의 순위 리스트를 RRF 점수로 합산.

    공식: score(doc) = sum( 1 / (k + rank) ) for each retriever where doc 출현
    - k=60: 표준값. 작을수록 상위 순위 가산 ↑, 클수록 평탄
    - 점수 자체값이 아닌 '순위'만 사용 → dense/sparse 점수 스케일 달라도 안정
    - 두 검색 모두 상위면 점수 합산 → 자동 가점

    예: dense에서 rank 1, sparse에서 rank 3인 청크의 점수
        = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
    """
    scores: dict[int, float] = {}
    # enumerate(..., start=1): 인덱스를 0이 아닌 1부터 시작 (rank가 1부터이므로)
    for rank, cid in enumerate(dense_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, cid in enumerate(sparse_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


# ===== 후보 검색 (게이트 미적용) ====================================

async def retrieve_candidates(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    top_n: int = 20,                      # 후보 수 — 평가가 Recall@N 보려면 넓게
    candidates_per_branch: int = 30,      # 각 검색에서 가져올 후보 수
) -> RetrievalCandidates:
    """하이브리드 검색까지만. 근거 게이트는 apply_gate가 담당.

    NOTE: tenant 격리는 명시적 WHERE 절로 강제 (tenant_scoped 유틸 미사용).
    유틸은 전체 row select에만 적합한데 여기선 id+점수만 가져오므로 직접 박음.
    """

    # ----- 1. query 임베딩 (dense-only, F99) -----------------
    q_emb = await embed_query(query)
    # [dense-only] sparse 제거 — 하이브리드 원복 시 해제
    # q_sparse = SparseVector(q_emb.sparse, 250002)

    # ----- 2. dense search: cosine distance 작은 순 top-N ---------
    # cosine_distance: pgvector-python이 제공하는 SQLAlchemy 헬퍼.
    # 실제 SQL은 chunks.dense <=> :q_dense 연산자로 변환됨.
    # 범위 [0, 2]: 0=완전 동일, 1=직각(무관), 2=정반대
    distance = Chunk.dense.cosine_distance(q_emb.dense).label('distance')
    # .label('distance'): SELECT 결과에 'distance' 라는 컬럼명 부여 → r.distance로 접근

    dense_stmt = (
        select(Chunk.id, distance)
        .outerjoin(Document, Chunk.document_id == Document.id)   # FAQ 청크(document NULL)도 살리는 outer join
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .outerjoin(Faq, Chunk.faq_id == Faq.id)
        .where(Chunk.tenant_id == tenant_id)         # 격리 강제
        .where(_searchable_condition())              # 출처별 참조 판정 (문서 F2 조건 / FAQ on)
        .order_by(distance)                          # 작은 순 = 유사한 순
        .limit(candidates_per_branch)
    )
    # (await session.execute(stmt)).all() → list[Row]
    dense_rows = (await session.execute(dense_stmt)).all()
    # 튜플 리스트로 정규화: [(chunk_id, distance), ...]
    dense_results = [(r.id, r.distance) for r in dense_rows]

    # ----- 3~5. [dense-only, F99] distance 순 top-N -----------------
    # 하이브리드(sparse 검색 + RRF)는 아래 주석 보관 — 원복 시 이 블록으로 되돌리면 됨.
    #   neg_ip = Chunk.sparse.max_inner_product(q_sparse).label('score')
    #   sparse_stmt = (select(Chunk.id, neg_ip)
    #       .outerjoin(Document, Chunk.document_id == Document.id)
    #       .outerjoin(Folder, Document.folder_id == Folder.id)
    #       .outerjoin(Faq, Chunk.faq_id == Faq.id)
    #       .where(Chunk.tenant_id == tenant_id).where(_searchable_condition())
    #       .order_by(neg_ip).limit(candidates_per_branch))
    #   sparse_rows = (await session.execute(sparse_stmt)).all()
    #   sparse_results = [(r.id, r.score) for r in sparse_rows]
    #   dense_ids = [cid for cid, _ in dense_results]
    #   sparse_ids = [cid for cid, _ in sparse_results]
    #   scores = _rrf_fuse(dense_ids, sparse_ids)
    #   if not scores: return RetrievalCandidates(chunks=[], top_dense_distance=999.0)
    #   top_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
    dense_ids = [cid for cid, _ in dense_results]
    top_ids = dense_ids[:top_n]
    # rrf_score 자리 채움(순위 역수) — 하이브리드 복원 시 위 블록으로 대체
    scores = {cid: 1.0 / rank for rank, cid in enumerate(dense_ids, start=1)}

    # 빈 결과 → 빈 후보 반환 (no_results 판정은 apply_gate가)
    if not top_ids:
        return RetrievalCandidates(chunks=[], top_dense_distance=999.0)

    # ----- 6. 본문 fetch -----------------------------------------
    # 위에선 id + 점수만 가져왔으니, 이제 본문 + 메타 가져옴
    # IN (...) 절로 한 번에 조회 (개별 SELECT N번 안 함)
    chunk_rows = await session.execute(
        select(Chunk, Document.filename, Document.version, Folder.name, Folder.description)
        .outerjoin(Document, Chunk.document_id == Document.id)   # FAQ 청크는 filename/version이 NULL로 옴
        .outerjoin(Folder, Document.folder_id == Folder.id)      # 미분류 문서·FAQ는 폴더가 NULL로 옴
        .where(Chunk.id.in_(top_ids))
    )
    # 결과를 dict로 만들어서 RRF 순서로 재배열할 때 O(1) 룩업.
    # FAQ 청크는 라벨을 'FAQ'로 — 컨텍스트 라벨=인용 형식 원칙에 따라 모델이 [FAQ]로 인용하게 됨
    chunk_map = {c.id: (c, filename or 'FAQ', ver or 1, fname, fdesc)
                 for c, filename, ver, fname, fdesc in chunk_rows.all()}

    # branch 판정용 set (in 연산 O(1))
    dense_id_set = set(dense_ids)
    sparse_id_set = set()          # [dense-only, F99] sparse 브랜치 없음

    result = []
    for cid in top_ids:                              # RRF 순서대로
        c, filename, version, folder_name, folder_description = chunk_map[cid]
        branches = []
        if cid in dense_id_set:
            branches.append('dense')
        if cid in sparse_id_set:
            branches.append('sparse')
        result.append(RetrievedChunk(
            chunk_id=c.id,
            document_id=c.document_id,
            text=c.text,
            heading_path=c.heading_path,
            page=c.page,
            rrf_score=scores[cid],
            branches=branches,
            filename=filename,
            version=version,
            faq_id=c.faq_id,
            is_table=bool((c.meta or {}).get('is_table')),   # F1a: xlsx 표 청크 여부
            folder_name=folder_name,
            folder_description=folder_description,
        ))

    # ----- 6.4 리랭커 재정렬 (F99, on/off = settings.rerank_enabled) ----------
    # 후보(top_n) 순서만 cross-encoder로 재정렬 → 표 필터·최종 top-K 전에 수행해
    # '가장 관련 높은 청크'가 앞·표 필터 기준이 되게 한다. 서버 실패 시 원 순서 유지(graceful).
    if settings.rerank_enabled and result:
        from rag.reranker import rerank   # 지연 import (retriever ↔ reranker 순환 방지)
        result = await rerank(query, result)

    # ----- 6.5 '한 시트만 참조' 필터 (F1a) ----------
    # 표(xlsx) 청크는 최상위 1개만 유지, 나머지 표 청크는 제외.
    # 표 간 값 교차오염 방지 (채널톡 '한 답변에 시트 1개' 정책). 산문·FAQ 청크는 무관하게 통과.
    # 시트=1청크(150행 상한)라 "최상위 표 청크 1개"가 곧 "시트 1개".
    result = _keep_single_table(result)

    # ----- 7. 게이트 신호만 챙겨 반환 (판정은 apply_gate) ----------
    # top-1 dense distance가 게이트의 유일한 입력. 여기선 계산만 하고 들려보낸다.
    top_dense_dist = dense_results[0][1] if dense_results else 999.0
    return RetrievalCandidates(chunks=result, top_dense_distance=top_dense_dist)


def _keep_single_table(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """표 청크는 RRF 최상위 1개만 남기고 나머지 표 청크 제외 (F1a '한 시트만')."""
    kept, table_seen = [], False
    for ch in chunks:
        if ch.is_table:
            if table_seen:
                continue          # 이미 상위 표 청크를 채택 → 다른 표는 버림
            table_seen = True
        kept.append(ch)
    return kept


# ===== 근거 게이트 ==================================================

def apply_gate(
    candidates: RetrievalCandidates,
    max_dense_distance: float = 0.6,      # 근거 게이트 임계값 (cosine distance)
                                          # 실측(2026-06): 정상 ~0.35, 무관 ~0.6 — 잠정값, Phase 1.5에서 튜닝
) -> tuple[bool, str | None]:
    """근거 게이트 판정만. (no_evidence, reason) 반환.

    DB·LLM 호출 없음 → 같은 후보에 임계값만 바꿔 반복 호출 가능 (1.5.B threshold sweep).

    reason: 'no_results'     (후보 0건 — dense/sparse 둘 다 빈 결과)
          | 'low_similarity' (top-1 dense 거리가 임계값 초과)
          | None             (정상)
    """
    if not candidates.chunks:
        return True, 'no_results'
    # top-1 dense distance가 임계값 초과 → "유사한 게 없다"고 판단
    if candidates.top_dense_distance > max_dense_distance:
        return True, 'low_similarity'
    return False, None


# ===== 조합 wrapper (운영 /kms/query 경로 — 동작 불변) ==============

async def retrieve(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    top_k: int = 5,                       # 최종 반환 청크 수
    candidates_per_branch: int = 30,
    max_dense_distance: float = float("inf"),  # 거리 게이트 비활성 (no_results만 유지)
                                          # 대상이 상담원 어시스턴트라 잡담/무관 질의가 드물고,
                                          # dense 거리 게이트는 rare_lexical 등 정상 질의를 오거부
                                          # (1.5.E 평가: 0.6에서 무근거 5%만 거름, 0.5는 정상 14% 오거부).
                                          # 무근거 환각 방어는 LLM 프롬프트(확인 불가)·groundedness가 담당.
                                          # 외부 공개/잡담 환경이면 0.5~0.6으로 되돌릴 것.
) -> RetrievalResult:
    """retrieve_candidates(넓게) → apply_gate(임계값 판정) → top_k로 잘라 반환.

    최종 top_k는 정렬된 후보의 상위 슬라이스라 리팩토링 전과 동일 결과·동일 순서.
    RagService 등 호출부는 이 함수만 알면 됨 (계약 불변).
    """
    candidates = await retrieve_candidates(
        session, tenant_id, query,
        top_n=20,
        candidates_per_branch=candidates_per_branch,
    )
    no_evidence, reason = apply_gate(candidates, max_dense_distance)
    return RetrievalResult(
        chunks=candidates.chunks[:top_k],
        no_evidence=no_evidence,
        reason=reason,
    )
