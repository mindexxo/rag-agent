"""dense-only 검색 + 리랭커 + 근거 게이트 (F99 — 하이브리드 원복 청사진은 본문 주석 보관).

흐름:
    query (+ 확장 변형, #5 Multi-Query)
      -> embed_texts (async, 배치 1회) -> q_dense × N
      -> 쿼리별 dense top-N (cosine distance)   -- 의미 매칭 (BGE-M3)
      -> 변형 있으면 union 후보 + 쿼리별 리랭크 max-pool (RRF는 폴백 순서)
         없으면 distance 순 그대로 -> 리랭커 재정렬 (cross-encoder, settings.rerank_enabled)
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
from rag.embeddings import embed_texts
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
    branches: list[str]             # 'dense'(원본)/'sparse'(하이브리드)/'expand'(변형 쿼리, #5) 조합 — 어느 검색에서 잡혔는지, 디버깅·로깅용
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
    rank_lists: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """N개 순위 리스트를 RRF 점수로 합산. 쿼리 확장(#5) 융합에 사용 중이며,
    하이브리드 원복 시에도 [dense_ids, sparse_ids]로 그대로 쓴다.

    공식: score(doc) = sum( 1 / (k + rank) ) for each list where doc 출현
    - k=60: 표준값. 작을수록 상위 순위 가산 ↑, 클수록 평탄
    - 점수 자체값이 아닌 '순위'만 사용 → 리스트 간 점수 스케일 달라도 안정
    - 여러 리스트에서 모두 상위면 점수 합산 → 자동 가점

    예: 리스트 A에서 rank 1, 리스트 B에서 rank 3인 청크의 점수
        = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
    """
    scores: dict[int, float] = {}
    for ids in rank_lists:
        # enumerate(..., start=1): 인덱스를 0이 아닌 1부터 시작 (rank가 1부터이므로)
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


# ===== 후보 검색 (게이트 미적용) ====================================

async def retrieve_candidates(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    top_n: int = 20,                      # 후보 수 — 평가가 Recall@N 보려면 넓게
    candidates_per_branch: int = 30,      # 각 검색에서 가져올 후보 수
    expanded_queries: list[str] | None = None,   # 쿼리 확장(#5) 변형 — 검색·RRF 융합 전용
) -> RetrievalCandidates:
    """하이브리드 검색까지만. 근거 게이트는 apply_gate가 담당.

    expanded_queries가 있으면 원본+변형을 각각 dense 검색해 union 후보를 만들고,
    쿼리별 cross-encoder 채점의 max-pool로 정렬한다 (#5). 리랭크 꺼짐/실패 시 RRF 순서 폴백.
    게이트 신호(top_dense_distance)는 항상 '원본 쿼리'의 top-1 거리 — 변형이
    의미 이탈해도 게이트 판정이 흔들리지 않는다.

    NOTE: tenant 격리는 명시적 WHERE 절로 강제 (tenant_scoped 유틸 미사용).
    유틸은 전체 row select에만 적합한데 여기선 id+점수만 가져오므로 직접 박음.
    """

    # ----- 1. query 임베딩 (dense-only, F99) -----------------
    # 원본+변형을 한 번에 배치 임베딩 (TEI 호출 1회 — embed_texts가 분할 담당)
    queries = [query, *(expanded_queries or [])]
    q_embs = await embed_texts(queries)
    # [dense-only] sparse 제거 — 하이브리드 원복 시 해제
    # q_sparse = SparseVector(q_embs[0].sparse, 250002)

    # ----- 2. dense search: cosine distance 작은 순 top-N ---------
    # 쿼리별로 순차 실행 — AsyncSession은 동시 실행 불가라 gather 금지.
    # cosine_distance: pgvector-python이 제공하는 SQLAlchemy 헬퍼.
    # 실제 SQL은 chunks.dense <=> :q_dense 연산자로 변환됨.
    # 범위 [0, 2]: 0=완전 동일, 1=직각(무관), 2=정반대
    per_query_ids: list[list[int]] = []
    dense_results: list[tuple[int, float]] = []   # 원본 쿼리 결과 — 게이트 신호용
    for i, q_emb in enumerate(q_embs):
        distance = Chunk.dense.cosine_distance(q_emb.dense).label('distance')
        # .label('distance'): SELECT 결과에 'distance' 라는 컬럼명 부여 → r.distance로 접근
        dense_stmt = (
            select(Chunk.id, distance)
            .outerjoin(Document, Chunk.document_id == Document.id)   # FAQ 청크(document NULL)도 살리는 outer join
            .outerjoin(Folder, Document.folder_id == Folder.id)
            .outerjoin(Faq, Chunk.faq_id == Faq.id)
            .where(Chunk.tenant_id == tenant_id)         # 격리 강제 — 변형 검색도 동일 경로
            .where(_searchable_condition())              # 출처별 참조 판정 (문서 F2 조건 / FAQ on)
            .order_by(distance)                          # 작은 순 = 유사한 순
            .limit(candidates_per_branch)
        )
        # (await session.execute(stmt)).all() → list[Row]
        rows = (await session.execute(dense_stmt)).all()
        if i == 0:
            # 튜플 리스트로 정규화: [(chunk_id, distance), ...]
            dense_results = [(r.id, r.distance) for r in rows]
        per_query_ids.append([r.id for r in rows])

    # ----- 3~5. 순위 결정 -----------------
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
    #   scores = _rrf_fuse([dense_ids, sparse_ids])
    #   if not scores: return RetrievalCandidates(chunks=[], top_dense_distance=999.0)
    #   top_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_n]]
    dense_ids = per_query_ids[0]
    multi = len(per_query_ids) > 1
    if not multi:
        # [dense-only, F99] 단일 쿼리 — distance 순 그대로 (기존 동작 불변)
        top_ids = dense_ids[:top_n]
        # rrf_score 자리 채움(순위 역수) — 하이브리드 복원 시 위 블록으로 대체
        scores = {cid: 1.0 / rank for rank, cid in enumerate(dense_ids, start=1)}
    else:
        # 쿼리 확장(#5): union 전체를 확보(슬라이스는 리랭크 뒤에서) — 순서는 RRF.
        # RRF는 max-pool 리랭크(6.4) 실패 시의 폴백 순서이자 rrf_score 표시값.
        # 원본이 항상 한 리스트로 들어가므로, 의미 이탈한 변형이 순위를 지배하지 못한다.
        scores = _rrf_fuse(per_query_ids)
        top_ids = [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]

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
    # 변형 쿼리에서 잡힌 청크 — 어느 검색이 기여했는지 디버깅·로깅용 (#5)
    expand_id_set = {cid for ids in per_query_ids[1:] for cid in ids}

    result = []
    for cid in top_ids:                              # RRF 순서대로
        c, filename, version, folder_name, folder_description = chunk_map[cid]
        branches = []
        if cid in dense_id_set:
            branches.append('dense')
        if cid in sparse_id_set:
            branches.append('sparse')
        if cid in expand_id_set:
            branches.append('expand')
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
    # 단일 쿼리: 후보(top_n) 순서만 cross-encoder로 재정렬 (기존 동작 불변).
    # 쿼리 확장(#5): union 후보를 쿼리별로 각자 채점해 청크별 최고점(max-pool)으로 정렬.
    #   RRF→원본 쿼리 채점 방식은 변형이 찾아온 청크를 원본 어휘로 다시 채점해 이득이
    #   소멸했다 (mt 90문항 실측: RRF+원본채점 = 풀확장+원본채점 = 개선 0, max-pool +4.5pp).
    # 표 필터·최종 top-K 전에 수행해 '가장 관련 높은 청크'가 앞·표 필터 기준이 되게 한다.
    # 서버 실패 시 원 순서(RRF) 유지(graceful) — 이때도 top_n 슬라이스는 보장.
    if settings.rerank_enabled and result:
        if multi:
            import asyncio
            from rag.reranker import rerank_scores   # 지연 import (retriever ↔ reranker 순환 방지)
            matrix = await asyncio.gather(*(rerank_scores(q, result) for q in queries))
            if all(s is not None for s in matrix):
                best = [max(col) for col in zip(*matrix)]
                order = sorted(range(len(result)), key=lambda idx: -best[idx])
                result = [result[idx] for idx in order]
        else:
            from rag.reranker import rerank
            result = await rerank(query, result)
    result = result[:top_n]   # multi는 union 전체를 들고 왔으므로 여기서 top_n 확정 (단일은 이미 top_n)

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
    expanded_queries: list[str] | None = None,   # 쿼리 확장(#5) 변형 — retrieve_candidates로 전달만

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
        expanded_queries=expanded_queries,
    )
    no_evidence, reason = apply_gate(candidates, max_dense_distance)
    return RetrievalResult(
        chunks=candidates.chunks[:top_k],
        no_evidence=no_evidence,
        reason=reason,
    )
