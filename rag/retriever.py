"""dense 검색 + 어휘 채널 주입(하이브리드 #135) + 리랭커 + 근거 게이트.

흐름 — 각 단계가 함수 하나다 (번호 주석 대신 이름으로 좇는다):

    query (+ 확장 변형, #5 Multi-Query)
      -> embed_texts                 쿼리 전체를 배치 1회로 임베딩
      -> os_search.search_dense_per_query   쿼리별 kNN top-N (cosine distance 환산)
      -> (단일: distance 순 그대로 / 멀티: _rank_multi 로 RRF 융합)
      -> os_search.search_lexical           BM25(Nori) 후보를 dense 뒤에 주입 (#135, 플래그)
      -> os_search.fetch_chunk_map          본문·메타 mget 1회 — 엔진이 서빙 정본 (#139)
      -> rerank / rerank_maxpool     cross-encoder 재정렬 (rag.reranker, settings.rerank_enabled)
      -> [:top_n]                    두 경로 공통 — 슬라이스는 항상 리랭크 뒤다
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

검색 저장소는 OpenSearch 하나다(#139 도입 확정). 후보 회수·본문 조회가 전부 엔진에서
끝나고 **이 모듈은 PG를 읽지 않는다** — `session` 인자는 호출부 호환용으로 남아 있을 뿐이다
(service·eval·tests 13곳이 넘긴다. 제거는 호출부 일괄 정리와 함께 별건). 검색 가능 여부
(문서 활성·ready·검색토글·폴더 토글·FAQ 활성)는 색인의 `searchable` 플래그로 판정하며,
그 플래그의 정합은 쓰기 경로(rag/outbox.py)가 책임진다.

LLM 호출 없음. Stage D의 RagService가 이 결과를 받아 답변 생성 또는
"확인 불가" 응답으로 분기.
"""
import logging
from dataclasses import replace

from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from rag import metrics, os_search, otel
from rag.embeddings import embed_texts
from rag.identifiers import cut_identifier_span, normalize_identifiers
from rag.retrieval_types import RetrievalCandidates, RetrievalResult, RetrievedChunk
from rag.tokens import estimate_tokens_conservative

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 20      # 후보 수 — retrieve_candidates 기본값과 retrieve() 호출부의 단일 출처
                        # (평가가 Recall@N을 보려면 최종 top_k보다 넓어야 한다)
GATE_DISABLED = float("inf")   # 거리 게이트를 끈 상태에 붙인 이름 — retrieve()의 운영 기본값.
                               # 사유는 retrieve() 인자 주석 참조. eval은 이 상수를 쓰지 않고
                               # 자기 임계값을 apply_gate에 직접 넘긴다 (threshold sweep).


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
#
# 후보 회수·본문 조회는 rag/os_search.py가 정의점이다. 아래 세 얇은 함수는 이 모듈의
# 단계 이름을 유지하기 위한 것이다 — 테스트·eval이 이 이름으로 monkeypatch·호출한다.
# `session`은 받지 않는다: 검색은 DB를 읽지 않는다(모듈 docstring).


async def _dense_candidates(tenant_id: str, q_embs: list, candidates_per_branch: int):
    """쿼리별 dense top-N. (쿼리별 id 리스트, 원본 쿼리의 (id, distance)) — 계약은 opensearch 참조."""
    return await os_search.search_dense_per_query(tenant_id, q_embs, candidates_per_branch)


async def _lexical_candidates(tenant_id: str, query: str, limit: int) -> list[int]:
    """어휘 채널 (#135) — BM25(Nori) 상위 limit개의 id. dense가 놓친 어휘 일치(상품코드·고유명)를
    리랭커 풀에 주입하는 용도. 최종 순위는 리랭커가 정하므로 무관 후보는 걸러진다."""
    return await os_search.search_lexical(tenant_id, query, limit)


async def _chunk_map(ids: list[int]) -> dict[int, RetrievedChunk]:
    """id → RetrievedChunk. 엔진 _source에서 만든다 — 왕복 1회, PG를 되묻지 않는다(실무 표준)."""
    return await os_search.fetch_chunk_map(ids)


def _rank_multi(per_query_ids: list[list[int]]) -> list[int]:
    """쿼리 확장(#5) 순위 — RRF로 union 전체를 정렬한다. 자르지 않는다.

    슬라이스는 리랭크 뒤(호출부)라 max-pool이 union 전체를 채점한다.
    RRF는 max-pool 리랭크 실패 시의 폴백 순서다. 원본 쿼리가 항상 한 리스트로
    들어가므로, 의미 이탈한 변형이 순위를 지배하지 못한다.

    단일 쿼리는 이 함수를 타지 않는다 — dense distance 순이 이미 순위라 그대로 쓴다.
    """
    scores = _rrf_fuse(per_query_ids)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _table_key(ch: RetrievedChunk) -> tuple:
    """표 묶음이 속한 표 — (문서 id, table_key). table_key는 엑셀은 시트명, PDF·DOCX는 문서 안 표 순번이다(#257)."""
    return (ch.document_id, ch.table_key)


# 근거 토큰 예산 — 문맥 창 30,720 − 답변 몫 3,000 − 대화 이력 최대 2,000 − 근거 1칸 뺀 프롬프트 p95 3,045 ≈ 22,675
# (#222 실측: 골드 658문항을 운영 검색으로 조립해 Qwen3 /tokenize로 셈). 여유를 두고 22,000. 첨부는 이 계산 밖이다.
# 표 하나는 TABLE_MAX_TOKENS(6,000, rag/chunking)를 넘지 않으므로(표 근거 = 걸린 묶음 약 1,000 + 표 전체, #268) 1등 근거는 언제나 들어간다. #257로 표가 여러 개 들어올 수
# 있게 되어(6,000 × 4 > 22,000) 실제로 닿을 수 있다 — 넘치면 prepare_evidence ④가 뺀다.
EVIDENCE_BUDGET_TOKENS = 22000


def apply_identifier_boost(chunks: list[RetrievedChunk], scores: list[float], query_idx: list[int],
                           tokens: list[str], boost: float):
    """리랭커 점수에 식별자 일치 가산(+boost)을 더해 재정렬한다 — 식별자 채널(#238)의 순위 정의점.

    질문 토큰이 조각의 `identifiers`(제목 경로 + 묶음 헤딩, rag/identifiers)에 **정확히** 있으면 가산하고
    `matched_ident`에 그 토큰을 적는다 — prepare_evidence가 이 표시로 묶음에서 그 구간만 오린다.
    가산은 리랭크 뒤에만 더한다: 리랭커가 제목의 장 번호를 조 번호로 보는 약점(실측 "제3장"→"제9장" 0.90→0.09)을
    리랭커 입력으로는 못 고치고, 점수 뒤 보정으로만 넘을 수 있었다. 0.3의 근거는 config.ident_boost 주석.
    같은 번호가 여러 문서에 있으면 전부 같은 가산을 받는다 — 어느 문서인지는 생성이 파일명으로 가린다(결정 16).
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


# 표 근거의 두 머리말(#268 결정 4) — 걸린 묶음을 표 전체 앞에 한 번 더 보여 준다.
TABLE_HIT_LABEL = '[검색에 걸린 행]'
TABLE_WHOLE_LABEL = '[표 전체]'


def table_evidence_text(ch: RetrievedChunk) -> str:
    """표 묶음 → 모델이 받는 글: **걸린 묶음 먼저 + 표 전체**(#268 결정 4). 묶음이 곧 표 전체면(작은 시트) 표 전체 하나.

    표 전체만 주면(#257) 한 행을 찾는 질문에서 14B가 이름이 비슷한 이웃 행을 읽었다 — '닭봉 51호' 단가에 59호 값(90행 단가표),
    '네롤리 앱솔루트' 성분에 옆 행 성분. 묶음만 주면(#257 전) 최댓값·개수·전체 나열에서 무너진다. 둘을 함께 주면 한 행 조회는
    걸린 행에서, 표 전체 질문은 표 전체에서 답한다. 실측(2026-10-20, 14B, 표 문항 50 × 2회, 같은 색인에서 근거 모양만 바꿈):
    한 행 조회 30문항 묶음만 48/60 · 표 전체만 48/60 · **둘 다 51/60**, 표 전체 질문 20문항 3/40 · 19/40 · **20/40**.
    대가: 근거가 평균 약 2,000자 길어진다. 알려진 경향: 집계 질문 일부에서 걸린 행 위주로 답해 항목을 빠뜨렸다(A/S 접수처 4곳 중 2곳).
    """
    if not ch.text.strip() or ch.text.strip() == (ch.table_text or '').strip():
        return ch.table_text
    return f'{TABLE_HIT_LABEL}\n{ch.text}\n\n{TABLE_WHOLE_LABEL}\n{ch.table_text}'


def prepare_evidence(chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    """검색 후보(순위순) → 모델에 넣을 근거. **운영(retrieve)과 생성 평가(eval.generation)가 함께 쓰는 정의점**(#222).

    ① 같은 표의 묶음은 가장 위 것 하나로 접는다 — 근거 칸을 같은 표가 여럿 차지하지 않게(접힌 자리는 아래 순위가 채운다).
       표는 엑셀 시트(#222)와 PDF·DOCX 큰 표(#257)다 — 키는 _table_key. 서로 다른 표는 몇 개든 남는다(#257 결정 1: 옛 '한
       시트만' 필터를 없앴다 — PDF는 한 문서에 표가 여럿인 게 정상이고, 엑셀만 막을 근거가 없었다)
    ② top_k로 자른다
    ③ 남은 표 묶음을 **걸린 묶음 + 표 전체**로 바꾼다(table_evidence_text) — 모델은 언제나 표 전체를 보고("부분 표" 위험 차단,
       #257), 걸린 행은 앞에 따로 본다(긴 표에서 이웃 행을 읽는 오답 차단, #268 결정 4)
    ⑤ 식별자 채널로 걸린 묶음 조각(matched_ident)은 그 식별자 구간만 남긴다(#238) — 구간 정의는 rag/identifiers.
       표 전체에는 기록된 헤딩 줄이 없어 그대로 통과한다(cut_identifier_span은 경계를 못 찾으면 글을 바꾸지 않는다)
    ④ 근거 추정 토큰(보수적 ÷1.1 — 예산이 실제 토크나이저 기준이라 덜 세면 안 된다)이 EVIDENCE_BUDGET_TOKENS를 넘으면
       순위가 낮은 산문 청크부터 빼고, 산문이 없으면 순위가 낮은 표를 **통째로** 뺀다(경고 로그·지표). 1등은 빼지 않는다 —
       표 하나는 TABLE_MAX_TOKENS 안이라 늘 들어간다. 표를 묶음으로 되돌려 일부만 넣지 않는다(#257 결정 4 — 일부만 보면
       나머지 행을 지어낸다)
    chunk_id·document_id 등은 걸린 묶음의 것을 그대로 둔다 — 인용 번호·의미 캐시는 문서 단위라 바뀌지 않는다.
    **#257 이전에 색인된 표는 재색인해야 한다** — 옛 이름(sheet_text)은 읽지 않는다(대비 코드 없음, #257 리뷰 결정). 재색인 전
    엑셀은 table_key가 없어 한 파일의 시트가 한 칸으로 접히고 table_text가 없어 걸린 묶음만 들어간다.
    검색 지표는 이 함수 앞(retrieve_candidates)을 채점한다 — ①을 앞에서 하면 정답 묶음이 형제에 밀려 미스로 채점된다.
    """
    seen, folded = set(), []
    for ch in chunks:
        if ch.is_table:
            key = _table_key(ch)
            if key in seen:
                continue
            seen.add(key)
        folded.append(ch)
    out = [replace(ch, text=table_evidence_text(ch)) if ch.is_table and ch.table_text else ch for ch in folded[:top_k]]
    # ⑤ 식별자로 걸린 묶음 조각은 그 구간만(#238 결정 13·18) — 색인·순위·라벨은 그대로, 넘기는 글만 바뀐다.
    #    묶음을 통째로 주면 모델이 옆 조를 읽는다(실측: 실질문 23건 18→21 정답, 회귀 0). 조 하나짜리 조각은 불변.
    #    표 근거(③에서 바뀐 것)는 오리지 않는다 — 걸린 묶음 머리의 조 제목 줄부터 잘라 [검색에 걸린 행] 머리말이 떨어졌다(#268).
    out = [replace(ch, text=cut_identifier_span(ch.text, ch.headings, ch.matched_ident))
           if ch.matched_ident and ch.headings and not (ch.is_table and ch.table_text) else ch for ch in out]
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
    top_n: int = DEFAULT_TOP_N,           # 리랭크 **후** 남길 후보 수 (리랭커 입력 수가 아니다)
    candidates_per_branch: int = 30,      # 각 쿼리의 dense 검색에서 가져올 후보 수
                                          # = 리랭커가 실제로 보는 수 (단일 경로 기준).
                                          # TEI 배치 상한 32 이내라 20→30이 왕복을 늘리지 않는다.
    expanded_queries: list[str] | None = None,   # 쿼리 확장(#5) 변형 — 검색·RRF 융합 전용
) -> RetrievalCandidates:
    """dense 검색 → 순위 → 본문 → 리랭크 → 표 필터. 근거 게이트는 apply_gate가 담당.

    expanded_queries가 있으면 원본+변형을 각각 dense 검색해 union 후보를 만들고,
    쿼리별 cross-encoder 채점의 max-pool로 정렬한다 (#5). 리랭크 꺼짐/실패 시 RRF 순서 폴백.

    부수효과 없음 (엔진 읽기 + 임베딩·리랭크 호출 2종뿐 — 공급자는 설정, #289). `session`은 쓰지 않는다(모듈 docstring).
    """
    queries = [query, *(expanded_queries or [])]

    # embed_texts는 이 함수 안에서 직접 부른다 — tests/conftest.py가
    # `rag.retriever.embed_texts`를 monkeypatch하므로, 이 호출을 다른 모듈로 옮기면
    # (헬퍼로 감싸도 그 헬퍼가 retriever.py 밖이면 마찬가지) 패치가 안 먹고
    # 테스트가 실제 임베딩 서버를 때린다. 옮기려면 conftest부터 함께 고칠 것.
    q_embs = await embed_texts(queries, kind="query")   # 임베딩 호출 1회 — 배치 분할은 embed_texts가 담당. 질의 용도(#289)
    query_embedding = q_embs[0].dense     # 원본 쿼리 벡터 — 캐시가 재사용한다 (#50).
                                          # Embedding 래퍼를 여기서 벗긴다: 도착지(cache.py·
                                          # AnswerCacheRow.query_embedding)가 list[float]를 받으므로
                                          # 하위 모듈이 rag.embeddings.Embedding을 알 필요가 없다.

    per_query_ids, dense_results = await _dense_candidates(
        tenant_id, q_embs, candidates_per_branch,
    )

    # 분기 판정은 여기 한 곳. 아래 리랭크가 이 값을 그대로 받는다.
    # 어느 쪽이든 자르지 않는다 — 슬라이스는 리랭크 뒤에서 한 번만(#38).
    multi = len(per_query_ids) > 1
    top_ids = _rank_multi(per_query_ids) if multi else per_query_ids[0]

    # 빈 결과 → 빈 후보 반환 (no_results 판정은 apply_gate가).
    # dense가 비면 어휘 주입도 안 한다 — 게이트 신호(top-1 거리)가 dense 기준이라,
    # 어휘 후보만으로 채우면 '근거 없음' 판정이 조용히 무력화된다.
    if not top_ids:
        return RetrievalCandidates(chunks=[], top_dense_distance=999.0,
                                   query_embedding=query_embedding)

    # 하이브리드 어휘 주입 (#135) — 원본 쿼리 기준 BM25 상위 중 dense 미포함분을 뒤에 붙인다.
    # '뒤에'가 계약이다: 리랭크 꺼짐/실패 폴백에서 dense 순위가 깨지지 않아야 한다
    # (리랭크 on에선 cross-encoder가 쌍 독립 채점이라 순서 무관). 슬라이스는 여전히 리랭크 뒤.
    if settings.hybrid_lexical_enabled:
        lex_ids = await _lexical_candidates(tenant_id, query, candidates_per_branch)
        seen = set(top_ids)
        top_ids = top_ids + [cid for cid in lex_ids if cid not in seen][:settings.hybrid_lexical_inject]

    # 식별자 채널 주입(#238) — 질문에 사전 식별자('11조')가 있을 때만. 정확 일치 조각 중 dense·BM25가 놓친 것을
    # 뒤에 붙인다(순서 계약은 BM25 주입과 같다). 주입만으로는 부족하고(585문항 실패 15→6, 실질문 1위 7/23)
    # 아래 리랭크 뒤 가산이 있어야 "제11장" 조각을 넘는다 — 리랭커가 장·조를 못 가리기 때문(실측 0.90→0.09).
    ident_tokens = normalize_identifiers(query) if settings.ident_channel_enabled else []
    if ident_tokens:
        ident_ids = await os_search.search_identifiers(tenant_id, ident_tokens, settings.ident_inject)
        seen = set(top_ids)
        top_ids = top_ids + [cid for cid in ident_ids if cid not in seen][:settings.ident_inject]

    chunk_map = await _chunk_map(top_ids)
    # 순위 순서 그대로 재배열. `if cid in chunk_map`은 필수다 — 검색과 본문 조회가 다른
    # 호출이라 그 사이에 문서가 지워지면 mget에서 빠지고, 가드가 없으면 KeyError로 검색
    # 전체가 죽는다.
    result = [chunk_map[cid] for cid in top_ids if cid in chunk_map]

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
            result, best, best_qi = await rerank_maxpool(queries, result)
            if best is None:
                # 점수 실패 → 넘겨받은 순서 유지 (멀티=RRF 융합, 단일=dense)
                otel.set_attrs(sp, {'kms.fallback': 'rrf' if multi else 'dense'})
            elif ident_tokens:
                # 식별자 가산(#238 결정 12·17) — 리랭커 **뒤**, 리랭커 입력은 불변. 점수 실패 시엔 가산도 없다(주입만 남는다).
                result, best, best_qi = apply_identifier_boost(result, best, best_qi, ident_tokens, settings.ident_boost)
                otel.set_attrs(sp, {'kms.ident_tokens': list(ident_tokens),
                                    'kms.ident_matched': [str(c.chunk_id) for c in result if c.matched_ident]})
            if best is not None and sp.is_recording():   # 채택 점수·기여 쿼리 — DB에 안 남는 진단 정보 (#5·#7·#54)
                for i, (ch, score, qi) in enumerate(zip(result[:top_n], best[:top_n], best_qi[:top_n])):
                    sp.set_attribute(f'reranker.output_documents.{i}.document.id', str(ch.chunk_id))
                    sp.set_attribute(f'reranker.output_documents.{i}.document.score', float(score))
                    sp.set_attribute(f'reranker.output_documents.{i}.document.query_index', qi)   # 0=원본, 1..=확장

    # top_n이 확정되는 유일한 지점 — 두 경로 공통이다. 리랭크 뒤라서
    # 리랭커가 하위 후보를 끌어올릴 여지를 슬라이스가 미리 깎지 않는다.
    result = result[:top_n]

    # 게이트 신호만 챙겨 반환 (판정은 apply_gate). 원본 쿼리의 top-1 거리다.
    top_dense_dist = dense_results[0][1] if dense_results else 999.0
    return RetrievalCandidates(chunks=result, top_dense_distance=top_dense_dist,
                               query_embedding=query_embedding)


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
        chunks=prepare_evidence(candidates.chunks, top_k),   # 접기·자르기·시트 확장·예산(#222)
        no_evidence=no_evidence,
        reason=reason,
        # 게이트 판정과 무관하게 항상 옮긴다 — no_evidence면 유일한 소비처(service의
        # cache.get_semantic 호출)가 애초에 안 불리므로 분기를 두는 게 군더더기다.
        query_embedding=candidates.query_embedding,
    )
