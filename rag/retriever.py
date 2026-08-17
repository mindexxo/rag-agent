"""dense-only 검색 + 리랭커 + 근거 게이트 (F99).

흐름 — 각 단계가 함수 하나다 (번호 주석 대신 이름으로 좇는다):

    query (+ 확장 변형, #5 Multi-Query)
      -> embed_texts                 쿼리 전체를 배치 1회로 임베딩
      -> _search_dense_per_query     쿼리별 dense top-N (cosine distance)
      -> (단일: distance 순 그대로 / 멀티: _rank_multi 로 RRF 융합)
      -> _fetch_chunk_map            본문·메타 IN절 1회 조회
      -> rerank / rerank_maxpool     cross-encoder 재정렬 (rag.reranker, settings.rerank_enabled)
      -> [:top_n]                    두 경로 공통 — 슬라이스는 항상 리랭크 뒤다
      -> _keep_single_table          표는 한 시트만 (F1a)
                                     == retrieve_candidates()
      -> apply_gate                  근거 게이트 (top-1 거리 임계값)
      -> RetrievalResult             == retrieve() 가 위 둘을 조합

**리랭커는 두 경로 모두 후보 전체를 본다** — 단일이면 dense `candidates_per_branch`개,
멀티면 union 전체. 슬라이스가 리랭크 뒤에 있어서 리랭커가 순위를 뒤집을 여지를 안 깎는다.
한때 단일 경로만 리랭크 **앞에서** top_n으로 잘라(#5가 멀티를 얹으며 생긴 비대칭)
dense 21~30위를 리랭커에게 보여주지 않았다 — #38에서 해소.

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
from rag import otel
from rag.embeddings import embed_texts
from rag.models import Chunk, Document, Faq, Folder

DEFAULT_TOP_N = 20      # 후보 수 — retrieve_candidates 기본값과 retrieve() 호출부의 단일 출처
                        # (평가가 Recall@N을 보려면 최종 top_k보다 넓어야 한다)
GATE_DISABLED = float("inf")   # 거리 게이트를 끈 상태에 붙인 이름 — retrieve()의 운영 기본값.
                               # 사유는 retrieve() 인자 주석 참조. eval은 이 상수를 쓰지 않고
                               # 자기 임계값을 apply_gate에 직접 넘긴다 (threshold sweep).


# 검색 대상 판정 — 출처별 분기 (chunks는 document/faq 다형성, F3):
#   문서 청크: 활성 버전 + 인덱싱 완료 + 문서 on + (미분류 OR 폴더 on)
#   FAQ 청크 : 항목 on
#
# 주의: 이 조건은 dense 검색의 ORDER BY/LIMIT과 함께 걸리는 **post-filter**다. HNSW가
# ef_search(기본 40)로 뽑은 뒤 여기서 걸러내므로, 비활성 문서·검색 끔 폴더가 늘면
# candidates_per_branch를 못 채우고 리콜이 **에러 없이** 떨어진다. 해소하려면
# hnsw.iterative_scan + ef_search 상향이 필요한데 값 변경 = 동작 변경이라 별건(11번 축).
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

    - chunks: 최종 정렬 순 top_n 후보 (리랭크 on이면 리랭크 후 순서)
    - top_dense_distance: 원본 쿼리의 top-1 dense distance — 근거 게이트의 유일한 입력 신호.
                          후보 없으면 999.0 (기존 'no_results' 분기와 동일 의미)
                          **운영은 게이트가 꺼져 있어(GATE_DISABLED) 이 값을 쓰지 않는다.**
                          실사용처는 eval의 threshold sweep(eval/gate.py·retrieval.py)뿐이다.
                          리랭크·표 필터와 무관하게 산정되므로, 이 신호가 가리키는 청크가
                          최종 chunks에 없을 수도 있다 (sweep 해석 시 주의).
    """
    chunks: list[RetrievedChunk]
    top_dense_distance: float


# ===== RRF (Reciprocal Rank Fusion) =================================

def _rrf_fuse(
    rank_lists: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """N개 순위 리스트를 RRF 점수로 합산. 쿼리 확장(#5) 융합에 사용.

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


# ===== 단계별 함수 ==================================================

async def _search_dense_per_query(
    session: AsyncSession,
    tenant_id: str,
    q_embs: list,
    candidates_per_branch: int,
) -> tuple[list[list[int]], list[tuple[int, float]]]:
    """쿼리별 dense top-N 검색. (쿼리별 id 리스트, 원본 쿼리의 (id, distance)) 반환.

    두 번째 반환값은 게이트 신호 전용 — 원본 쿼리(index 0) 결과만 담는다.
    변형 쿼리가 의미를 이탈해도 게이트 판정이 흔들리지 않게 하기 위함이다.

    부수효과 없음 (DB 읽기 전용). 쿼리 수만큼 **순차** 실행한다 —
    AsyncSession은 한 세션에 동시 쿼리를 못 보내므로 gather 금지.
    (별도 세션을 빌리면 생성 1건당 커넥션이 늘어 풀 문제를 악화시킨다 — 3번 축 실측.)

    NOTE: tenant 격리는 명시적 WHERE 절로 강제 (프로젝트 규약 — rag/models.py 참조).
    유틸은 전체 row select에만 적합한데 여기선 id+점수만 가져오므로 직접 박음.
    """
    per_query_ids: list[list[int]] = []
    dense_results: list[tuple[int, float]] = []
    for i, q_emb in enumerate(q_embs):
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
            .where(Chunk.tenant_id == tenant_id)         # 격리 강제 — 변형 검색도 동일 경로
            .where(_searchable_condition())              # 출처별 참조 판정 (문서 F2 조건 / FAQ on)
            .order_by(distance)                          # 작은 순 = 유사한 순
            .limit(candidates_per_branch)
        )
        # (await session.execute(stmt)).all() → list[Row]
        rows = (await session.execute(dense_stmt)).all()
        if i == 0:
            dense_results = [(r.id, r.distance) for r in rows]
        per_query_ids.append([r.id for r in rows])
    return per_query_ids, dense_results


def _rank_multi(per_query_ids: list[list[int]]) -> list[int]:
    """쿼리 확장(#5) 순위 — RRF로 union 전체를 정렬한다. 자르지 않는다.

    슬라이스는 리랭크 뒤(호출부)라 max-pool이 union 전체를 채점한다.
    RRF는 max-pool 리랭크 실패 시의 폴백 순서다. 원본 쿼리가 항상 한 리스트로
    들어가므로, 의미 이탈한 변형이 순위를 지배하지 못한다.

    단일 쿼리는 이 함수를 타지 않는다 — dense distance 순이 이미 순위라 그대로 쓴다.
    """
    scores = _rrf_fuse(per_query_ids)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


async def _fetch_chunk_map(
    session: AsyncSession,
    ids: list[int],
) -> dict[int, RetrievedChunk]:
    """id → RetrievedChunk 본문·메타 조회. IN 절 한 번 (개별 SELECT N번 안 함).

    부수효과 없음 (DB 읽기 전용). 순서는 호출부가 정한다 — 여기선 dict만 만든다.
    FAQ 청크는 라벨을 'FAQ'로 — 컨텍스트 라벨=인용 형식 원칙에 따라 모델이 [FAQ]로 인용하게 됨.
    """
    rows = await session.execute(
        select(Chunk, Document.filename, Document.version, Folder.name, Folder.description)
        .outerjoin(Document, Chunk.document_id == Document.id)   # FAQ 청크는 filename/version이 NULL로 옴
        .outerjoin(Folder, Document.folder_id == Folder.id)      # 미분류 문서·FAQ는 폴더가 NULL로 옴
        .where(Chunk.id.in_(ids))
    )
    return {
        c.id: RetrievedChunk(
            chunk_id=c.id,
            document_id=c.document_id,
            text=c.text,
            heading_path=c.heading_path,
            page=c.page,
            filename=filename or 'FAQ',
            version=ver or 1,
            faq_id=c.faq_id,
            is_table=bool((c.meta or {}).get('is_table')),   # F1a: xlsx 표 청크 여부
            folder_name=fname,
            folder_description=fdesc,
        )
        for c, filename, ver, fname, fdesc in rows.all()
    }


def _keep_single_table(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """표 청크는 최상위 1개만 남기고 나머지 표 청크 제외 (F1a '한 시트만').

    파일이 달라도 표는 하나만 남는다 — 채널톡 ALF와 같은 제약이다
    ("ALF는 한 번의 답변에서 하나의 시트만 참조합니다. 여러 엑셀 파일이나 한 파일 내
    여러 시트를 동시에 참조하는 것은 현재 지원되지 않습니다", docs.channel.io 지식 ALF v2).
    산문·FAQ 청크는 무관하게 통과. 시트=1청크(150행 초과는 업로드 거절)라
    "최상위 표 청크 1개"가 곧 "시트 1개"다.

    **버린 자리를 채우지 않는다** — top_n 슬라이스 뒤에 돌기 때문에, 표 청크가 상위에
    여럿이면 최종 후보가 top_n보다 적어진다(21위 이하로 백필하지 않음). 첫 구현부터
    이 동작이고 eval도 이 위에서 측정됐다.
    **'최상위'의 기준은 넘겨받은 순서다** — 리랭크가 꺼졌거나 실패하면 리랭커 점수가
    아니라 dense/RRF 순서로 표 1등이 정해진다.
    """
    kept, table_seen = [], False
    for ch in chunks:
        if ch.is_table:
            if table_seen:
                continue          # 이미 상위 표 청크를 채택 → 다른 표는 버림
            table_seen = True
        kept.append(ch)
    return kept


# ===== 후보 검색 (게이트 미적용) ====================================

async def retrieve_candidates(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    top_n: int = DEFAULT_TOP_N,           # 리랭크 **후** 남길 후보 수 (리랭커 입력 수가 아니다)
    candidates_per_branch: int = 30,      # 각 쿼리의 dense 검색에서 가져올 후보 수
                                          # = 리랭커가 실제로 보는 수 (단일 경로 기준).
                                          # TEI 배치 상한 32 이내라 20→30이 왕복을 늘리지 않는다.
    expanded_queries: list[str] | None = None,   # 쿼리 확장(#5) 변형 — 검색·RRF 융합 전용
) -> RetrievalCandidates:
    """dense 검색 → 순위 → 본문 → 리랭크 → 표 필터. 근거 게이트는 apply_gate가 담당.

    expanded_queries가 있으면 원본+변형을 각각 dense 검색해 union 후보를 만들고,
    쿼리별 cross-encoder 채점의 max-pool로 정렬한다 (#5). 리랭크 꺼짐/실패 시 RRF 순서 폴백.

    부수효과 없음 (DB 읽기 + TEI 호출 2종뿐).
    """
    queries = [query, *(expanded_queries or [])]

    # embed_texts는 이 함수 안에서 직접 부른다 — tests/conftest.py가
    # `rag.retriever.embed_texts`를 monkeypatch하므로, 이 호출을 다른 모듈로 옮기면
    # (헬퍼로 감싸도 그 헬퍼가 retriever.py 밖이면 마찬가지) 패치가 안 먹고
    # 테스트가 실제 TEI 서버를 때린다. 옮기려면 conftest부터 함께 고칠 것.
    q_embs = await embed_texts(queries)   # TEI 호출 1회 — 배치 분할은 embed_texts가 담당

    per_query_ids, dense_results = await _search_dense_per_query(
        session, tenant_id, q_embs, candidates_per_branch,
    )

    # 분기 판정은 여기 한 곳. 아래 리랭크가 이 값을 그대로 받는다.
    # 어느 쪽이든 자르지 않는다 — 슬라이스는 리랭크 뒤에서 한 번만(#38).
    multi = len(per_query_ids) > 1
    top_ids = _rank_multi(per_query_ids) if multi else per_query_ids[0]

    # 빈 결과 → 빈 후보 반환 (no_results 판정은 apply_gate가)
    if not top_ids:
        return RetrievalCandidates(chunks=[], top_dense_distance=999.0)

    chunk_map = await _fetch_chunk_map(session, top_ids)
    result = [chunk_map[cid] for cid in top_ids]      # 순위 순서 그대로 재배열

    # ----- 리랭커 재정렬 (on/off = settings.rerank_enabled) ----------
    # 호출 시점마다 settings를 읽는다 — eval/retrieval.py가 이 속성을 런타임에 켰다 껐다 한다.
    # 표 필터·최종 top_n 전에 수행해 '가장 관련 높은 청크'가 앞·표 필터 기준이 되게 한다.
    # 서버 실패 시 원 순서 유지(graceful) — 이때도 top_n 슬라이스는 보장된다.
    if settings.rerank_enabled and result:
        with otel.span('rerank', 'RERANKER') as sp:
            otel.set_attrs(sp, {otel.RERANK_QUERY: query, otel.RERANK_TOP_K: top_n,
                                'kms.pool_size': len(result), 'kms.mode': 'maxpool' if multi else 'single'})
            # 단일/멀티 공통 — 쿼리 1개의 max-pool은 단일 리랭크와 동일하다(#54, 골든 대조).
            from rag.reranker import rerank_maxpool   # 지연 import (retriever ↔ reranker 순환 방지)
            result, best = await rerank_maxpool(queries, result)
            if best is None:
                # 점수 실패 → 넘겨받은 순서 유지 (멀티=RRF 융합, 단일=dense)
                otel.set_attrs(sp, {'kms.fallback': 'rrf' if multi else 'dense'})
            elif sp.is_recording():   # 채택 점수 — DB에 안 남는 진단 정보 (#5·#7)
                for i, (ch, score) in enumerate(zip(result[:top_n], best[:top_n])):
                    sp.set_attribute(f'reranker.output_documents.{i}.document.id', str(ch.chunk_id))
                    sp.set_attribute(f'reranker.output_documents.{i}.document.score', float(score))

    # top_n이 확정되는 유일한 지점 — 두 경로 공통이다. 리랭크 뒤라서
    # 리랭커가 하위 후보를 끌어올릴 여지를 슬라이스가 미리 깎지 않는다.
    result = result[:top_n]
    result = _keep_single_table(result)   # 버린 자리는 백필하지 않는다 (함수 docstring 참조)

    # 게이트 신호만 챙겨 반환 (판정은 apply_gate). 원본 쿼리의 top-1 거리다.
    top_dense_dist = dense_results[0][1] if dense_results else 999.0
    return RetrievalCandidates(chunks=result, top_dense_distance=top_dense_dist)


# ===== 근거 게이트 ==================================================

def apply_gate(
    candidates: RetrievalCandidates,
    max_dense_distance: float = 0.6,      # 근거 게이트 임계값 (cosine distance)
                                          # 실측(2026-06): 정상 ~0.35, 무관 ~0.6 — 잠정값, Phase 1.5에서 튜닝
) -> tuple[bool, str | None]:
    """근거 게이트 판정만. (no_evidence, reason) 반환.

    DB·LLM 호출 없음 → 같은 후보에 임계값만 바꿔 반복 호출 가능 (1.5.B threshold sweep).

    **운영에서는 거리 판정이 꺼져 있다** — retrieve()가 GATE_DISABLED를 넘기므로
    'low_similarity'는 도달하지 않고 'no_results'만 실질적으로 작동한다.
    유한 임계값을 넘기는 곳은 eval(gate.py·retrieval.py의 sweep)뿐이다.

    reason: 'no_results'     (후보 0건)
          | 'low_similarity' (top-1 dense 거리가 임계값 초과 — eval 전용 경로)
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

    max_dense_distance: float = GATE_DISABLED,  # 거리 게이트 비활성 (no_results만 유지)
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
    with otel.span('retrieve', 'RETRIEVER') as sp:
        candidates = await retrieve_candidates(
            session, tenant_id, query,
            candidates_per_branch=candidates_per_branch,
            expanded_queries=expanded_queries,
        )
        otel.set_attrs(sp, {otel.INPUT_VALUE: query,
                            'kms.expanded_queries': list(expanded_queries or [])})
        otel.set_documents(sp, candidates.chunks)
    no_evidence, reason = apply_gate(candidates, max_dense_distance)
    return RetrievalResult(
        chunks=candidates.chunks[:top_k],
        no_evidence=no_evidence,
        reason=reason,
    )
