"""하이브리드(BM25 주입) 어블레이션 v2 (#128) — dense 단독 대비 어휘 채널 후보 주입 실측.

구 어블레이션(eval/report_retrieval_ablation_v1.md)의 재측정판이다. 그때와 다른 것:
현행 코퍼스(corpus_v2 + 하드 문서)·현행 gold(v2+hard95)·'파일명>헤딩' 프리픽스 도입 후.
가중 RRF(0.8/0.2)+상위30 컷 조립은 1차 실행에서 **결함으로 판명**(BM25_INJECT 주석 참조)
— 하이브리드의 실제 가설은 "dense가 놓친 정답의 후보 주입"이라 union으로 잰다.
스크립트를 저장소에 두는 이유: 구판 스크립트가 scratchpad에서 유실돼 재작성했던 교훈.

변형 (전부 리랭커 on = 운영 동일 조건, +off 진단):
  baseline      dense30 → 리랭크 → top20              (eval/retrieval_v2.py와 동일 경로)
  hyb_union     dense30 + bm25 전용 상위 15 주입 → 리랭크(풀 ≤45) → top20
  bm25only      bm25_30 → 리랭크 → top20               (어휘 채널 건강 진단용)
  *_norr        위 순서에서 리랭크 생략                  (TEI 무료 — '리랭커가 채널 차이 흡수' 재확인)

BM25 = 순수 Python Okapi(k1=1.5, b=0.75), 토크나이저 = 문자 bigram(pg_bigm 프리뷰 —
구 어블레이션과 동일). 입력 텍스트 = 임베딩과 동일한 build_index_text(파일명>헤딩+본문).
후보 코퍼스 = dense와 같은 _searchable_condition 풀 (변인 격리).

채점은 retrieval_v2와 동일 계약: gold 정본만, resolve_gold는 테넌트 전체 배치(부분 배치는
gold_ids 분포를 바꾼다 — resolve_gold docstring), score_one(ks=(5,20)), 슬라이스는
리랭크·융합 뒤 한 번, _keep_single_table 동일 위치. baseline 파일은 덮어쓰지 않고
eval/results/hybrid_ablation_v2.jsonl 에 변형별 행을 저장한다.

실행: python -m eval._hybrid_ablation   (의존: DB + TEI 임베딩·리랭커. LLM 불필요)
"""
import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from sqlalchemy import select, text as sa_text

from database import AsyncSessionLocal
from eval.generation import row_tenant
from eval.retrieval import resolve_gold, score_one
from eval.retrieval_v2 import DIFFICULTY, GOLD, METRICS, TYPES
from rag.embeddings import embed_texts
from rag.index_text import build_index_text
from rag.models import Chunk, Document, Faq, Folder
from rag.reranker import rerank
from rag.retriever import (_fetch_chunk_map, _keep_single_table, _search_dense_per_query,
                           _searchable_condition)

OUT = Path(__file__).resolve().parent / "results" / "hybrid_ablation_v2.jsonl"
POOL = 30           # 채널별 후보 수 = candidates_per_branch 기본값과 동일
TOP_N = 20
# 가중 RRF+상위30 컷 변형은 v2 1차 실행에서 조립 결함으로 판명 — dense 우세 가중에선
# 어휘 전용 후보의 최고 RRF 점수(w_b/(k+1))가 dense 꼴찌 점수(w_d/(k+POOL))보다 낮아
# 후보 풀이 baseline과 동일해진다(실측: 소수점 3자리까지 일치). 하이브리드의 실제 가설은
# "dense가 놓친 정답의 후보 주입"이므로 union이 옳은 조립이다: 최종 순위는 리랭커가 정한다.
BM25_INJECT = 15    # dense30에 없는 bm25 상위 주입 수 — 리랭커 풀 최대 45


# ── BM25 (Okapi, 문자 bigram) ────────────────────────────────────────────────

def _bigrams(text: str) -> list[str]:
    s = "".join(text.split()).lower()
    return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


class Bm25:
    """테넌트 코퍼스 하나에 대한 Okapi BM25. 코퍼스가 작아(테넌트당 ~100청크) 전수 채점."""

    def __init__(self, docs: dict[int, str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf: dict[int, dict[str, int]] = {}
        self.dl: dict[int, int] = {}
        df: dict[str, int] = defaultdict(int)
        for cid, text in docs.items():
            toks = _bigrams(text)
            tf: dict[str, int] = defaultdict(int)
            for t in toks:
                tf[t] += 1
            self.tf[cid] = tf
            self.dl[cid] = len(toks)
            for t in tf:
                df[t] += 1
        n = len(docs)
        self.avgdl = (sum(self.dl.values()) / n) if n else 1.0
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def top(self, query: str, n: int) -> list[int]:
        q = _bigrams(query)
        scores: dict[int, float] = defaultdict(float)
        for t in q:
            idf = self.idf.get(t)
            if idf is None:
                continue
            for cid, tf in self.tf.items():
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[cid] / self.avgdl)
                scores[cid] += idf * f * (self.k1 + 1) / denom
        return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])[:n]]


async def _trgm_top(session, tenant: str, query: str, n: int) -> list[int]:
    """pg_trgm 채널 — 운영 후보 그대로(기설치 확장) 실제 DB similarity()로 잰다.

    대상 텍스트는 chunks.text 원문 — 운영에서 GIN 인덱스를 걸 컬럼과 동일하게.
    (임베딩형 프리픽스 텍스트는 DB에 없다 — 그걸 쓰려면 컬럼 추가라 별도 판단.)
    searchable 필터는 dense와 동일해야 변인이 격리된다.
    """
    stmt = sa_text("""
        SELECT c.id, similarity(c.text, :q) AS sim
        FROM chunks c
        LEFT JOIN documents d ON c.document_id = d.id
        LEFT JOIN folders f ON d.folder_id = f.id
        LEFT JOIN faqs q2 ON c.faq_id = q2.id
        WHERE c.tenant_id = :tenant
          AND ((c.document_id IS NOT NULL AND d.is_active AND d.status = 'ready'
                AND d.is_searchable AND (d.folder_id IS NULL OR f.is_searchable))
               OR (c.faq_id IS NOT NULL AND q2.is_active))
        ORDER BY sim DESC
        LIMIT :n""")
    rows = (await session.execute(stmt, {"q": query, "tenant": tenant, "n": n})).all()
    return [r.id for r in rows if r.sim and r.sim > 0]


async def _load_corpus(session, tenant: str) -> dict[int, str]:
    """BM25 코퍼스 — dense와 동일한 searchable 풀. 텍스트는 임베딩 입력과 동일 조립."""
    stmt = (
        select(Chunk.id, Chunk.text, Chunk.heading_path, Document.filename)
        .outerjoin(Document, Chunk.document_id == Document.id)
        .outerjoin(Folder, Document.folder_id == Folder.id)
        .outerjoin(Faq, Chunk.faq_id == Faq.id)
        .where(Chunk.tenant_id == tenant)
        .where(_searchable_condition())
    )
    rows = (await session.execute(stmt)).all()
    out = {}
    for r in rows:
        if r.filename is None:      # FAQ 청크 — 인제스션과 동일하게 프리픽스 없음
            out[r.id] = r.text
        else:
            out[r.id] = build_index_text(r.text, r.filename, list(r.heading_path or []))
    return out


# ── 융합·슬라이스 (운영 retrieve_candidates와 같은 순서 규약) ─────────────────

async def _final_ids(session, query: str, cand_ids: list[int], use_rerank: bool) -> list[int]:
    """후보 id → (리랭크) → top20 → 표 필터 — 슬라이스는 재정렬 뒤 한 번 (#38 규약)."""
    if not cand_ids:
        return []
    chunk_map = await _fetch_chunk_map(session, cand_ids)
    chunks = [chunk_map[cid] for cid in cand_ids if cid in chunk_map]
    if use_rerank:
        chunks = await rerank(query, chunks)
    chunks = _keep_single_table(chunks[:TOP_N])
    return [c.chunk_id for c in chunks]


def _agg(rows_v: list[dict]) -> dict:
    if not rows_v:
        return {}
    out = {m: sum(r["scores"][m] for r in rows_v) / len(rows_v) for m in METRICS}
    by_diff = defaultdict(list)
    by_type = defaultdict(list)
    for r in rows_v:
        by_diff[DIFFICULTY.get(r["type"], "medium")].append(r["scores"])
        by_type[r["type"]].append(r["scores"])
    out["by_difficulty"] = {g: {m: sum(s[m] for s in ss) / len(ss) for m in ("recall_at_5", "hit_at_1")}
                            for g, ss in by_diff.items()}
    out["by_type"] = {t: {m: sum(s[m] for s in ss) / len(ss) for m in ("recall_at_5", "hit_at_1")}
                      for t, ss in by_type.items()}
    return out


async def main() -> None:
    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g["type"] in TYPES]
    by_tenant = defaultdict(list)
    for g in target:
        by_tenant[row_tenant(g)].append(g)

    variants = ["baseline", "hyb_union", "hyb_trgm", "bm25only",
                "baseline_norr", "hyb_union_norr", "bm25only_norr"]
    rows: dict[str, list[dict]] = {v: [] for v in variants}
    skipped = 0
    inject_wins: list[str] = []   # 주입 상방 — 정답이 dense30 밖 & bm25_30 안
    trgm_wins: list[str] = []     # 동일 상방 — trgm 채널

    async with AsyncSessionLocal() as session:
        for tenant, items in by_tenant.items():
            corpus = await _load_corpus(session, tenant)
            bm25 = Bm25(corpus)
            resolved = await resolve_gold(session, tenant, items)
            print(f"[{tenant}] 청크 {len(corpus)} · 문항 {len(items)}", flush=True)
            for g in items:
                gold_ids = resolved.chunk_ids.get(g["id"]) or []
                if not gold_ids:
                    skipped += 1
                    continue
                q = g["query"]
                q_embs = await embed_texts([q])
                per_query_ids, _ = await _search_dense_per_query(session, tenant, q_embs, POOL)
                dense_ids = per_query_ids[0]
                bm25_ids = bm25.top(q, POOL)

                trgm_ids = await _trgm_top(session, tenant, q, POOL)
                injected = [c for c in bm25_ids if c not in set(dense_ids)][:BM25_INJECT]
                trgm_injected = [c for c in trgm_ids if c not in set(dense_ids)][:BM25_INJECT]
                if not (set(gold_ids) & set(dense_ids)) and (set(gold_ids) & set(bm25_ids)):
                    inject_wins.append(g["id"])
                if not (set(gold_ids) & set(dense_ids)) and (set(gold_ids) & set(trgm_ids)):
                    trgm_wins.append(g["id"])
                cand = {
                    "baseline": dense_ids,
                    "bm25only": bm25_ids,
                    # union: dense 뒤에 주입 — norr(리랭크 없음)에서 dense 순위가 안 깨지게.
                    # 리랭크 on에선 순서 무관(cross-encoder가 쌍 독립 채점).
                    "hyb_union": dense_ids + injected,
                    "hyb_trgm": dense_ids + trgm_injected,   # 전환 후보 채널(#128 사용자 결정)
                }
                for name, ids in cand.items():
                    got = await _final_ids(session, q, ids, use_rerank=True)
                    rows[name].append({"id": g["id"], "type": g["type"], "tenant": tenant,
                                       "scores": score_one(got, gold_ids)})
                for name in ("baseline", "hyb_union", "bm25only"):
                    got = await _final_ids(session, q, cand[name], use_rerank=False)
                    rows[f"{name}_norr"].append({"id": g["id"], "type": g["type"], "tenant": tenant,
                                                 "scores": score_one(got, gold_ids)})

    if not any(rows.values()):
        raise SystemExit("결과 0행 — DB/코퍼스 상태 확인 (파일 미저장)")

    out_rows = [{"variant": v, **r} for v in variants for r in rows[v]]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))

    print(f"\n측정 {datetime.now():%Y-%m-%d %H:%M} · 대상 {len(rows['baseline'])}문항 (skip {skipped})")
    print(f"{'variant':<15}{'R@5':>7}{'Hit@1':>8}{'MRR':>7}{'R@5 hard':>10}{'Hit@1 hard':>12}")
    for v in variants:
        a = _agg(rows[v])
        hard = a["by_difficulty"].get("hard_new", {})
        print(f"{v:<15}{a['recall_at_5']:>7.3f}{a['hit_at_1']:>8.3f}{a['mrr']:>7.3f}"
              f"{hard.get('recall_at_5', 0):>10.3f}{hard.get('hit_at_1', 0):>12.3f}")

    base = _agg(rows["baseline"])["by_type"]
    print("\n[type별 Hit@1 — baseline 대비 Δ]")
    for v in ("hyb_union", "hyb_trgm", "bm25only"):
        a = _agg(rows[v])["by_type"]
        deltas = {t: round(a[t]["hit_at_1"] - base[t]["hit_at_1"], 3) for t in sorted(base)}
        print(f"  {v}: { {t: d for t, d in deltas.items() if abs(d) >= 0.005} or '전 타입 ±0.005 미만' }")
    print(f"\n주입 상방(정답이 dense30 밖 & bm25 안): {len(inject_wins)}건 {inject_wins}")
    print(f"주입 상방(정답이 dense30 밖 & trgm 안): {len(trgm_wins)}건 {trgm_wins}")
    print(f"행 저장: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
