"""검색 — dense 후보에 어휘·식별자 후보를 주입하고 리랭크한 뒤, 근거 게이트와 근거 준비까지 한다.

    query (+ 확장 변형)
      -> embed_texts                        쿼리 전체를 배치 1회로 임베딩
      -> os_search.search_dense_per_query   쿼리별 kNN top-N
      -> (단일: distance 순 / 멀티: _rank_multi RRF 융합)
      -> os_search.search_lexical           BM25 후보를 dense 뒤에 주입
      -> os_search.fetch_chunk_map          본문·메타 mget 1회
      -> rerank_maxpool                     cross-encoder 재정렬
      -> [:top_n]                           슬라이스는 항상 리랭크 뒤 == retrieve_candidates()
      -> apply_gate                         근거 게이트 (top-1 거리 임계값)
      -> RetrievalResult                    == retrieve()

리랭커는 두 경로 모두 후보 전체를 본다 — 슬라이스가 리랭크 뒤라 순위를 뒤집을 여지를 안 깎는다.
eval은 retrieve_candidates·apply_gate를 직접 불러 Recall@N·threshold sweep을 본다.

이 모듈은 PG를 읽지 않는다 — `session` 인자는 호출부 호환용으로만 남아 있다. 검색 가능 여부는
색인의 `searchable` 플래그로 판정하며, 그 정합은 쓰기 경로(rag/outbox.py)가 책임진다.
"""

import logging
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag import metrics, os_search, otel
from rag.embeddings import embed_texts
from rag.identifiers import cut_identifier_span, normalize_identifiers
from rag.retrieval_types import DenseHits, GateVerdict, RetrievalCandidates, RetrievalResult, RetrievedChunk
from rag.tokens import estimate_tokens_conservative

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 20  # 평가가 Recall@N을 보려면 최종 top_k보다 넓어야 한다
GATE_DISABLED = float("inf")  # 거리 게이트를 끈 상태 — retrieve()의 운영 기본값(사유는 retrieve() 인자 주석)


# ===== RRF (Reciprocal Rank Fusion) =================================


def _rrf_fuse(
    rank_lists: list[list[int]],
    k: int = 60,
) -> dict[int, float]:
    """N개 순위 리스트를 RRF 점수(sum 1/(k+rank))로 합산 — 점수가 아닌 순위만 써서 리스트 간 스케일 차이에 안정적이다."""
    scores: dict[int, float] = {}
    for ids in rank_lists:
        for rank, cid in enumerate(ids, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


# ===== 단계별 함수 ==================================================
#
# 아래 세 얇은 래퍼는 이름을 바꾸거나 없애지 말 것 — eval/hybrid_weights.py가 이 이름으로 monkeypatch한다.


async def _dense_candidates(tenant_id: str, q_embs: list, candidates_per_branch: int) -> DenseHits:
    return await os_search.search_dense_per_query(tenant_id, q_embs, candidates_per_branch)


async def _lexical_candidates(tenant_id: str, query: str, limit: int) -> list[int]:
    """BM25 상위 limit개의 id — dense가 놓친 어휘 일치(상품코드·고유명)를 리랭커 풀에 주입하는 용도."""
    return await os_search.search_lexical(tenant_id, query, limit)


async def _chunk_map(ids: list[int]) -> dict[int, RetrievedChunk]:
    return await os_search.fetch_chunk_map(ids)


def _rank_multi(per_query_ids: list[list[int]]) -> list[int]:
    """쿼리 확장 순위 — RRF로 union 전체를 정렬하고 자르지 않는다. 리랭크 실패 시의 폴백 순서다."""
    scores = _rrf_fuse(per_query_ids)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def table_identity(ch: RetrievedChunk) -> tuple:
    """어느 문서의 어느 표인가 — 같은 표의 묶음을 접는 키. table_key 필드는 엑셀은 시트명, PDF·DOCX는 문서 안 표 순번."""
    return (ch.document_id, ch.table_key)


# 근거 토큰 예산 — 문맥 창 30,720 − 답변 몫 − 대화 이력 − 근거 뺀 프롬프트에서 여유를 둔 값. 첨부는 이 계산 밖이다.
EVIDENCE_BUDGET_TOKENS = 22000


def apply_identifier_boost(
    chunks: list[RetrievedChunk], scores: list[float], query_idx: list[int], tokens: list[str], boost: float
):
    """리랭커 점수에 식별자 일치 가산(+boost)을 더해 재정렬한다 — 식별자 채널의 순위 정의점.

    질문 토큰이 조각의 `identifiers`에 정확히 있으면 가산하고 `matched_ident`에 적는다(prepare_evidence가 이걸로 오린다).
    리랭크 뒤에만 더하고 리랭커 입력은 바꾸지 않는다 — 리랭커가 제목의 장 번호를 조 번호로 보는 약점은 점수 뒤 보정으로만 넘는다.
    같은 번호가 여러 문서에 있으면 전부 같은 가산을 받는다 — 어느 문서인지는 생성이 파일명으로 가린다.
    """
    tok = set(tokens)
    rows = []
    for ch, score, qi in zip(chunks, scores, query_idx):
        hit = next((t for t in ch.identifiers if t in tok), None)
        if hit is not None:
            ch = replace(ch, matched_ident=hit)
            score = score + boost
        rows.append((ch, score, qi))
    rows.sort(key=lambda r: -r[1])
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


# 표 근거의 두 머리말 — 걸린 묶음을 표 전체 앞에 한 번 더 보여 준다.
TABLE_HIT_LABEL = '[검색에 걸린 행]'
TABLE_WHOLE_LABEL = '[표 전체]'


def table_evidence_text(ch: RetrievedChunk) -> str:
    """표 묶음 → 모델이 받는 글: 걸린 묶음 먼저 + 표 전체. 묶음이 곧 표 전체면 표 전체 하나.

    표 전체만 주면 한 행 조회에서 이웃 행을 읽고, 묶음만 주면 최댓값·개수·전체 나열 질문에서 무너진다.
    """
    if not ch.text.strip() or ch.text.strip() == (ch.table_text or '').strip():
        return ch.table_text
    return f'{TABLE_HIT_LABEL}\n{ch.text}\n\n{TABLE_WHOLE_LABEL}\n{ch.table_text}'


def prepare_evidence(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """검색 후보(순위순) → 모델에 넣을 근거. 운영(retrieve)과 생성 평가(eval.generation)가 함께 쓰는 정의점.

    ① 같은 표의 묶음은 가장 위 것 하나로 접는다(서로 다른 표는 몇 개든 남는다) ② top_k로 자른다
    ③ 표 묶음을 걸린 묶음 + 표 전체로 바꾼다 ④ 식별자로 걸린 산문 조각은 그 식별자 구간만 남긴다
    ⑤ 보수적 추정 토큰이 EVIDENCE_BUDGET_TOKENS를 넘으면 순위 낮은 산문부터, 산문이 없으면 순위 낮은 표를 통째로 뺀다.
       1등은 빼지 않는다. 표를 일부만 넣지 않는다 — 일부만 보면 나머지 행을 지어낸다.
    chunk_id·document_id 등은 걸린 묶음의 것을 그대로 둔다(인용·의미 캐시는 문서 단위).
    검색 지표는 이 함수 앞(retrieve_candidates)을 채점한다 — ①을 앞에서 하면 정답 묶음이 형제에 밀려 미스로 채점된다.
    """
    seen, folded = set(), []
    for ch in chunks:
        if ch.is_table:
            key = table_identity(ch)
            if key in seen:
                continue
            seen.add(key)
        folded.append(ch)
    out = [replace(ch, text=table_evidence_text(ch)) if ch.is_table and ch.table_text else ch for ch in folded[:top_k]]
    # 표 근거(③에서 바뀐 것)는 오리지 않는다 — 조 제목 줄부터 잘라 [검색에 걸린 행] 머리말이 떨어진다.
    out = [
        replace(ch, text=cut_identifier_span(ch.text, ch.headings, ch.matched_ident))
        if ch.matched_ident and ch.headings and not (ch.is_table and ch.table_text)
        else ch
        for ch in out
    ]
    while len(out) > 1 and sum(estimate_tokens_conservative(c.text) for c in out) > EVIDENCE_BUDGET_TOKENS:
        drop = next((i for i in range(len(out) - 1, 0, -1) if not out[i].is_table), len(out) - 1)
        logger.warning('근거 예산 초과 — 순위 %d의 청크(%s)를 뺀다', drop + 1, out[drop].filename)
        metrics.EVIDENCE_BUDGET_TRIM_TOTAL.inc()
        del out[drop]
    return out


# ===== 후보 검색 (게이트 미적용) ====================================


async def retrieve_candidates(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    top_n: int = DEFAULT_TOP_N,  # 리랭크 후 남길 후보 수 (리랭커 입력 수가 아니다)
    candidates_per_branch: int = 30,  # 쿼리별 dense 후보 수(어휘·식별자 주입분은 따로 더해진다)
    expanded_queries: list[str] | None = None,  # 쿼리 확장 변형 — 검색·RRF 융합 전용
) -> RetrievalCandidates:
    """dense 검색 → 주입 → 본문 → 리랭크 → top_n. 근거 게이트는 apply_gate가 담당.

    expanded_queries가 있으면 union 후보를 쿼리별 채점의 max-pool로 정렬한다. 리랭크 꺼짐/실패 시 RRF(단일은 dense) 순서.
    """
    queries = [query, *(expanded_queries or [])]

    # 이 호출을 retriever.py 밖으로 옮기지 말 것 — tests/conftest.py가 `rag.retriever.embed_texts`를 패치한다.
    q_embs = await embed_texts(queries, kind="query")
    query_embedding = q_embs[0].dense  # 캐시가 재사용하는 원본 쿼리 벡터 — Embedding 래퍼는 여기서 벗긴다

    ident_tokens = normalize_identifiers(query) if settings.ident_channel_enabled else []
    dense, top_ids = await _build_pool(tenant_id, query, q_embs, candidates_per_branch, ident_tokens)
    if not top_ids:
        return RetrievalCandidates(chunks=[], top_dense_distance=999.0, query_embedding=query_embedding)

    chunk_map = await _chunk_map(top_ids)
    # `if cid in chunk_map` 필수 — 검색과 본문 조회 사이에 문서가 지워지면 mget에서 빠진다.
    result = [chunk_map[cid] for cid in top_ids if cid in chunk_map]

    # 호출 시점마다 settings를 읽는다 — eval/retrieval.py가 이 속성을 런타임에 켰다 껐다 한다.
    if settings.rerank_enabled and result:
        multi = len(dense.per_query_ids) > 1
        result = await _rerank_pool(queries, result, ident_tokens, top_n, multi)

    # top_n이 확정되는 유일한 지점 — 리랭크 뒤라 하위 후보를 끌어올릴 여지를 미리 깎지 않는다.
    result = result[:top_n]

    # 게이트 신호는 원본 쿼리의 top-1 dense 거리다.
    top_dense_dist = dense.top[0].distance if dense.top else 999.0
    return RetrievalCandidates(chunks=result, top_dense_distance=top_dense_dist, query_embedding=query_embedding)


async def _build_pool(
    tenant_id: str, query: str, q_embs: list, candidates_per_branch: int, ident_tokens: list[str]
) -> tuple[DenseHits, list[int]]:
    """리랭크 전 후보 풀 — dense(멀티면 RRF) 뒤에 어휘·식별자 후보를 '뒤에' 붙인다(리랭크 꺼짐/실패 폴백에서 dense 순위 보존).

    dense가 비면 주입도 안 한다 — 어휘 후보만으로 채우면 '근거 없음' 판정이 조용히 무력화된다. 자르는 건 리랭크 뒤 한 번만.
    """
    dense = await _dense_candidates(tenant_id, q_embs, candidates_per_branch)
    multi = len(dense.per_query_ids) > 1
    top_ids = _rank_multi(dense.per_query_ids) if multi else dense.per_query_ids[0]
    if not top_ids:
        return dense, []

    if settings.hybrid_lexical_enabled:
        lex_ids = await _lexical_candidates(tenant_id, query, candidates_per_branch)
        seen = set(top_ids)
        top_ids = top_ids + [cid for cid in lex_ids if cid not in seen][: settings.hybrid_lexical_inject]

    # 식별자 정확 일치 조각 주입 — 주입만으로는 장·조를 못 가리는 리랭커를 못 넘어 리랭크 뒤 가산이 함께 필요하다.
    if ident_tokens:
        ident_ids = await os_search.search_identifiers(tenant_id, ident_tokens, settings.ident_inject)
        seen = set(top_ids)
        top_ids = top_ids + [cid for cid in ident_ids if cid not in seen][: settings.ident_inject]
    return dense, top_ids


async def _rerank_pool(
    queries: list[str], result: list[RetrievedChunk], ident_tokens: list[str], top_n: int, multi: bool
) -> list[RetrievedChunk]:
    """쿼리별 max-pool 리랭크 + 식별자 가산. 점수 실패면 입력 순서(RRF/dense) 그대로 — 가산도 없다."""
    with otel.span('rerank', 'RERANKER') as sp:
        otel.set_attrs(
            sp,
            {
                otel.RERANK_QUERY: queries[0],
                otel.RERANK_TOP_K: top_n,
                'kms.pool_size': len(result),
                'kms.mode': 'maxpool' if multi else 'single',
            },
        )
        # 단일/멀티 공통 — 쿼리 1개의 max-pool은 단일 리랭크와 같다.
        from rag.reranker import rerank_maxpool  # 지연 import (retriever ↔ reranker 순환 방지)

        result, best, best_qi = await rerank_maxpool(queries, result)
        if best is None:
            otel.set_attrs(sp, {'kms.fallback': 'rrf' if multi else 'dense'})
        elif ident_tokens:
            result, best, best_qi = apply_identifier_boost(result, best, best_qi, ident_tokens, settings.ident_boost)
            otel.set_attrs(
                sp,
                {
                    'kms.ident_tokens': list(ident_tokens),
                    'kms.ident_matched': [str(c.chunk_id) for c in result if c.matched_ident],
                },
            )
        if best is not None and sp.is_recording():  # 채택 점수·기여 쿼리 — DB에 안 남는 진단 정보
            for i, (ch, score, qi) in enumerate(zip(result[:top_n], best[:top_n], best_qi[:top_n])):
                sp.set_attribute(f'reranker.output_documents.{i}.document.id', str(ch.chunk_id))
                sp.set_attribute(f'reranker.output_documents.{i}.document.score', float(score))
                sp.set_attribute(f'reranker.output_documents.{i}.document.query_index', qi)  # 0=원본, 1..=확장
    return result


# ===== 근거 게이트 ==================================================


def apply_gate(
    candidates: RetrievalCandidates,
    max_dense_distance: float = 0.6,  # cosine distance 임계값 — 운영 retrieve()는 값을 넘긴다
) -> GateVerdict:
    """근거 게이트 판정.

    운영은 retrieve()가 GATE_DISABLED를 넘겨 'no_results'만 작동한다 — 유한 임계값은 eval sweep 전용.
    """
    if not candidates.chunks:
        return GateVerdict(True, 'no_results')
    if candidates.top_dense_distance > max_dense_distance:
        return GateVerdict(True, 'low_similarity')
    return GateVerdict(False, None)


# ===== 조합 wrapper (운영 경로) =====================================


async def retrieve(
    session: AsyncSession,
    tenant_id: str,
    query: str,
    top_k: int = 5,
    candidates_per_branch: int = 30,
    expanded_queries: list[str] | None = None,
    # 거리 게이트는 끈다 — 상담원 대상이라 무관 질의가 드물고 정상 질의를 오거부한다.
    # 무근거 방어는 생성 프롬프트(확인 불가)가 맡는다. 외부 공개·잡담 환경이면 0.5~0.6으로 되돌릴 것.
    max_dense_distance: float = GATE_DISABLED,
) -> RetrievalResult:
    """retrieve_candidates(넓게) → apply_gate → prepare_evidence(top_k)."""
    with otel.span('retrieve', 'RETRIEVER') as sp:
        candidates = await retrieve_candidates(
            session,
            tenant_id,
            query,
            candidates_per_branch=candidates_per_branch,
            expanded_queries=expanded_queries,
        )
        otel.set_attrs(sp, {otel.INPUT_VALUE: query, 'kms.expanded_queries': list(expanded_queries or [])})
        otel.set_documents(sp, candidates.chunks)
    verdict = apply_gate(candidates, max_dense_distance)
    return RetrievalResult(
        chunks=prepare_evidence(candidates.chunks, top_k),
        no_evidence=verdict.no_evidence,
        reason=verdict.reason,
        query_embedding=candidates.query_embedding,
    )
