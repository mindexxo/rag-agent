"""하이브리드 가중치 검증(#248) — 캡처한 값만으로 검색 조립을 다시 계산하는 **순수 계산부**. 실행 중 외부 연결(DB·OpenSearch·TEI)을
열지 않는다(import 체인에는 eval.retrieval의 database 모듈이 딸려 온다 — score_one 정의점을 한 곳에 두려는 대가).

왜 재생인가(설계 결정 5): 리랭커(cross-encoder)는 질문-조각 쌍을 독립으로 채점하므로, 질문마다 후보 상위집합의 점수를
한 번 받아 두면 풀 구성을 바꾼 조건을 다시 검색하지 않고 계산할 수 있다. 조건끼리 같은 점수를 쓰니 TEI의 호출마다의
흔들림(1.4e-4)이 조건 차이에 섞이지 않는다. 캡처·CLI는 eval/hybrid_weights.py.

**재생이 운영과 같다는 보장**: `assemble(BASELINE)`은 운영 rag/search/retriever.retrieve_candidates의 단일 질의 경로를 그대로 따른다
— 벡터 30 → BM25 30 중 벡터에 없는 것 15개 뒤에 → 식별자 일치 뒤에 → 리랭커 점수 안정 정렬 → 식별자 +가산 → 상위 20
(#257로 '한 시트만' 필터는 없어졌다). 재구현은 풀 조립(build_pool)뿐이고 나머지는 운영 함수(apply_identifier_boost·prepare_evidence)를
그대로 부른다. 그래도 재구현이 어긋날 수 있으므로 eval/hybrid_weights의 verify가 **캡처값을 운영 함수에 끼워 실제로 돌린
결과와 전 문항 완전 일치**를 확인한 뒤에만 측정한다(#128 1차 측정은 조립이 틀려 기준선과 수학적으로 같은 풀이 됐다).
tests/test_hybrid_replay가 같은 비교를 합성 데이터로 하고, 일부러 망가뜨린 재생이 그 비교에서 실패하는지도 본다.

레코드(캡처 1줄, dict): set·qid·tenant·type·query·gold_ids·token(키워드 축만)·dense30[[id,거리]]·dense100·bm25_30[id]·
bm25_100[[id,점수]]·ident_tokens·ident[id]·rerank{id:점수}. 조각은 {chunk_id: RetrievedChunk} 한 벌을 따로 둔다.
"""
import random
from collections import defaultdict
from dataclasses import dataclass

from eval.retrieval import score_one
from eval.axis_common import evidence_has
from rag.search.retrieval_types import RetrievedChunk
from rag.search.retriever import table_identity, apply_identifier_boost, prepare_evidence

TOP_N = 20      # 운영 retrieve_candidates 기본 — rag.search.retriever.DEFAULT_TOP_N
TOP_K = 5       # 모델이 받는 근거 수 — eval.generation.TOP_K


class PoolCoverageError(RuntimeError):
    """조건이 캡처에 없는 후보를 요구했다 — 조용히 0점 처리하지 않고 멈춘다(그 조건은 재캡처가 필요하다)."""


@dataclass(frozen=True)
class Condition:
    """조건 = 풀을 어떻게 채우고 리랭크 뒤 점수를 어떻게 섞나. 기본값이 운영(config.hybrid_lexical_inject=15·ident_inject=10·
    ident_boost=0.3, 혼합 없음)과 같다.

    리랭크 뒤 혼합(blend, 3단계 — 2단계 진단에서 풀 누락이 0이라 가중치가 닿을 수 있는 자리는 여기뿐이었다):
      - "add":    리랭커 점수 + w × BM25 정규화 점수 — #238 식별자 가산과 같은 자리·같은 모양(리랭커 확신은 크게 못 뒤집는다)
      - "convex": (1 − w) × 리랭커 점수 + w × BM25 정규화 점수 — 다른 팀의 '키워드 : 의미' 비중을 그대로 옮긴 꼴
      BM25 정규화 = 그 질문의 BM25 점수 / 그 질문 BM25 1위 점수(0~1). BM25 상위 100 밖 후보는 0.
      w는 질문이 '키워드형'(숫자 포함 또는 15자 이하 — 다른 팀 규칙 그대로)이면 w_key, 아니면 w_desc.
    식별자 가산은 혼합 뒤에 그대로 더한다(운영 순서: 리랭크 → 가산)."""
    name: str
    dense_n: int = 30
    lex_n: int = 30
    lex_inject: int = 15
    ident_inject: int = 10
    ident_boost: float = 0.3
    blend: str | None = None
    w_key: float = 0.0
    w_desc: float = 0.0


BASELINE = Condition("baseline")
NO_LEXICAL = Condition("키워드주입_0", lex_inject=0)   # 하이브리드 기여도 진단용 — 후보 아님(현행에서 키워드 채널만 끈 것)
# 3단계 후보(결정 11) — 결과를 보기 전에 고정한 목록이다(여러 조건 중 우연히 이긴 것을 고르는 착시를 막는다).
CANDIDATES = (
    Condition("add_0.1", blend="add", w_key=0.1, w_desc=0.1),
    Condition("add_0.2", blend="add", w_key=0.2, w_desc=0.2),
    Condition("add_0.3", blend="add", w_key=0.3, w_desc=0.3),
    Condition("add_동적_0.3_0.1", blend="add", w_key=0.3, w_desc=0.1),
    Condition("convex_동적_70_40", blend="convex", w_key=0.7, w_desc=0.4),    # 다른 팀 규칙: 키워드형 70:30, 서술형 40:60
)


def is_keyword_query(query: str) -> bool:
    """다른 팀의 동적 가중 기준 — 숫자가 있거나 15자 이하면 키워드형."""
    return any(ch.isdigit() for ch in query) or len(query.strip()) <= 15


def _blend(rec: dict, pool: list[RetrievedChunk], scores: list[float], cond: Condition) -> list[float]:
    if not cond.blend:
        return scores
    bm25 = {cid: sc for cid, sc in rec["bm25_100"]}
    top = max(bm25.values(), default=0.0) or 1.0
    w = cond.w_key if is_keyword_query(rec["query"]) else cond.w_desc
    norm = [bm25.get(c.chunk_id, 0.0) / top for c in pool]
    if cond.blend == "add":
        return [s + w * n for s, n in zip(scores, norm)]
    if cond.blend == "convex":
        return [(1 - w) * s + w * n for s, n in zip(scores, norm)]
    raise ValueError(f"알 수 없는 혼합 방식: {cond.blend}")


def _dense_ids(rec: dict, n: int) -> list[int]:
    if n == 30:
        return [cid for cid, _ in rec["dense30"]]   # 운영 크기는 따로 받은 것(설계 2 — 근사 탐색에서 100의 앞 30 ≠ 30)
    if n > len(rec["dense100"]) and len(rec["dense100"]) == 100:
        raise PoolCoverageError(f"벡터 {n}개는 캡처(100) 밖")
    return [cid for cid, _ in rec["dense100"][:n]]


def _bm25_ids(rec: dict, n: int) -> list[int]:
    if n == 30:
        return list(rec["bm25_30"])
    if n > len(rec["bm25_100"]) and len(rec["bm25_100"]) == 100:
        raise PoolCoverageError(f"BM25 {n}개는 캡처(100) 밖")
    return [cid for cid, _ in rec["bm25_100"][:n]]


def build_pool(rec: dict, cond: Condition = BASELINE) -> list[int]:
    """리랭커 풀(순서 있음) — rag/search/retriever.retrieve_candidates의 '하이브리드 어휘 주입(#135)'·'식별자 채널 주입(#238)' 블록과
    같은 조립. 벡터가 비면 빈 풀(어휘 주입도 안 함)."""
    top = _dense_ids(rec, cond.dense_n)
    if not top:
        return []
    seen = set(top)
    top = top + [cid for cid in _bm25_ids(rec, cond.lex_n) if cid not in seen][:cond.lex_inject]
    if rec["ident_tokens"]:
        seen = set(top)
        top = top + [cid for cid in rec["ident"] if cid not in seen][:cond.ident_inject]
    return top


def assemble(rec: dict, chunks: dict[int, RetrievedChunk], cond: Condition = BASELINE,
             top_n: int = TOP_N) -> list[RetrievedChunk]:
    """조건으로 최종 후보 top_n — 운영 retrieve_candidates(리랭크 on, 단일 질의)의 결과와 같아야 한다."""
    pool = [chunks[cid] for cid in build_pool(rec, cond) if cid in chunks]   # 본문 조회에서 빠진 id는 운영처럼 조용히 빠진다
    try:
        scores = [rec["rerank"][str(c.chunk_id)] for c in pool]
    except KeyError as e:
        raise PoolCoverageError(f"{rec['qid']}: 리랭커 점수 없는 후보 {e}") from None
    scores = _blend(rec, pool, scores, cond)
    order = sorted(range(len(pool)), key=lambda i: -scores[i])       # rag.search.reranker.rerank_maxpool과 같은 안정 정렬
    ranked, ranked_scores = [pool[i] for i in order], [scores[i] for i in order]
    if rec["ident_tokens"]:
        ranked, _, _ = apply_identifier_boost(ranked, ranked_scores, [0] * len(ranked), rec["ident_tokens"],
                                              cond.ident_boost)
    return ranked[:top_n]


def evaluate(rec: dict, final: list[RetrievedChunk], chunks: dict[int, RetrievedChunk]) -> dict:
    """문항 하나의 지표. in_evidence = 모델이 실제로 받는 근거에 정답이 있나 — 키워드 축은 근거 **글**에 토큰이 있나(evidence_has),
    나머지는 근거 조각 id에 정답이 있거나 **정답과 같은 표의 조각이 근거에 있나**. 표는 같은 표 조각을 가장 위 하나로
    접고 표 전체로 바꿔 넘기므로(prepare_evidence ①③, #257부터 PDF·DOCX 큰 표도) 정답 행이 든 형제 조각은 id로는 빠져도 모델은 그 행을 본다
    (첫 진단 실측: 최종 2~3위인데 '근거 밖'으로 세진 문장형 문항이 전부 이 꼴, 2026-10-18 — 건수는 eval/report_hybrid_weights). retrieval_article은 조 번호라 표가
    거의 없어 id 기준을 그대로 쓴다 — 이 함수의 셈법이 그보다 정확하다."""
    evidence = prepare_evidence(final, TOP_K)
    if rec["set"] == "keyword":
        in_ev = evidence_has(evidence, rec["token"])
    else:
        gold = set(rec["gold_ids"])
        tables = {table_identity(c) for c in evidence if c.is_table}
        in_ev = float(any(c.chunk_id in gold for c in evidence)
                      or any(g in chunks and chunks[g].is_table and table_identity(chunks[g]) in tables for g in gold))
    return {**score_one([c.chunk_id for c in final], rec["gold_ids"]), "in_evidence": in_ev}


# ----- 진단 (2단계, 가중치 없이) -----------------------------------------------

def diagnose(rec: dict, chunks: dict[int, RetrievedChunk]) -> dict:
    """현행 조건에서 문항이 어디서 놓치나 — 배타적 셋 중 하나.
    pool_miss: 정답이 리랭커 풀에 아예 없음 → 풀 구성을 바꾸는 가중치(가중 RRF·주입 수)만 구할 수 있다
    rerank_miss: 풀에는 있는데 근거 상위 5 밖 → 리랭크 뒤 점수에 섞는 가중치만 구할 수 있다
    hit: 근거 상위 5 안
    덧붙여 정답이 벡터 30·BM25 30/50/100 안에 있었는지(주입 상방 계산용)."""
    gold = set(rec["gold_ids"])
    ev = evaluate(rec, assemble(rec, chunks), chunks)
    in_pool = bool(gold & set(build_pool(rec)))
    status = "hit" if ev["in_evidence"] else ("rerank_miss" if in_pool else "pool_miss")
    no_inject = evaluate(rec, assemble(rec, chunks, NO_LEXICAL), chunks)["in_evidence"]

    def rank(ids):
        return next((i + 1 for i, c in enumerate(ids) if c in gold), None)

    d, b = rank([c for c, _ in rec["dense100"]]), rank([c for c, _ in rec["bm25_100"]])
    return {"status": status, "hit_at_1": ev["hit_at_1"],
            "in_dense30": bool(gold & {c for c, _ in rec["dense30"]}),
            **{f"in_bm25_{n}": bool(gold & set(_bm25_ids(rec, n))) for n in (30, 50, 100)},
            # 하이브리드 기여도 — 키워드 주입을 끄면 근거에서 빠지나(+1)/새로 들어오나(−1). 리랭커가 있어도 '후보 찾기'는 키워드가 한다
            "lexical_saves": int(ev["in_evidence"] > no_inject) - int(ev["in_evidence"] < no_inject),
            # 채널 단독 세기(리랭커 없이) — BM25가 강한 구간이 어디인가
            "dense_rank": d, "bm25_rank": b}


def summarize_diagnosis(rows: list[dict], key: str = "type") -> dict:
    """유형별 진단 집계. 주입 상방(N) = 정답이 현행 풀 밖인데 BM25 상위 N 안에 있는 문항 수 — 풀을 넓히는 가중치가 구할 수
    있는 상한이다(#128과 같은 정의, 그때 문장형 450문항에서 0). lexical_saves = 키워드 주입을 끄면 근거에서 빠지는 문항 수
    (하이브리드가 지금 하는 일), dense_top1·bm25_top1 = 리랭커 없이 채널 혼자 정답을 1위·5위 안에 두는 비율."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
        groups["(전체)"].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        n = len(rs)
        out[k] = {"n": n, **{s: sum(r["status"] == s for r in rs) for s in ("hit", "rerank_miss", "pool_miss")},
                  "top1_miss": sum(not r["hit_at_1"] for r in rs),
                  **{f"upside_bm25_{n_}": sum(r["status"] == "pool_miss" and r[f"in_bm25_{n_}"] for r in rs)
                     for n_ in (30, 50, 100)},
                  "lexical_saves": sum(r["lexical_saves"] > 0 for r in rs),
                  "lexical_hurts": sum(r["lexical_saves"] < 0 for r in rs),
                  **{f"{ch}_top{k_}": round(sum((r[f"{ch}_rank"] or 999) <= k_ for r in rs) / n, 3)
                     for ch in ("dense", "bm25") for k_ in (1, 5)}}
    return out


# ----- 통계 (3단계) ----------------------------------------------------------

def paired_bootstrap(base: list[float], cand: list[float], n: int = 10_000, seed: int = 248) -> dict:
    """같은 문항끼리 짝지은 차이(후보 − 기준)의 평균과 95% 신뢰구간(재표본 n회, 퍼센타일), 승·패·무 문항 수(설계 7)."""
    if len(base) != len(cand):
        raise ValueError("짝지을 수 없다 — 문항 수가 다르다")
    if not base:
        return {"n": 0, "mean_diff": 0.0, "ci95": (0.0, 0.0), "wins": 0, "losses": 0, "ties": 0}
    diffs = [c - b for b, c in zip(base, cand)]
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(diffs[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return {"n": m, "mean_diff": sum(diffs) / m, "ci95": (means[int(0.025 * n)], means[int(0.975 * n) - 1]),
            "wins": sum(d > 0 for d in diffs), "losses": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs)}


def compare(recs: list[dict], chunks: dict[int, RetrievedChunk], conds=CANDIDATES, key: str = "set",
            metrics=("in_evidence", "hit_at_1"), n_boot: int = 10_000) -> dict:
    """후보 조건 × 묶음(key) × 지표 → 현행 대비 짝 부트스트랩. 문항별 결과(이긴·진 문항 id)도 함께 — 원문과 나란히 볼 대상."""
    def scores_of(cond):
        return {r["qid"]: evaluate(r, assemble(r, chunks, cond), chunks) for r in recs}

    base = scores_of(BASELINE)
    groups: dict[str, list[str]] = defaultdict(list)
    for r in recs:
        groups[r[key]].append(r["qid"])
        groups["(전체)"].append(r["qid"])
    out = {}
    for cond in conds:
        got = scores_of(cond)
        out[cond.name] = {}
        for g, qids in sorted(groups.items()):
            out[cond.name][g] = {m: {**paired_bootstrap([base[q][m] for q in qids], [got[q][m] for q in qids], n=n_boot),
                                     "won": [q for q in qids if got[q][m] > base[q][m]],
                                     "lost": [q for q in qids if got[q][m] < base[q][m]]}
                                 for m in metrics}
    return out
