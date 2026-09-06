"""하이브리드(BM25 주입) 어블레이션 v2 (#128) — dense 단독 대비 어휘 채널 후보 주입 실측.

구 어블레이션(eval/report_retrieval_ablation_v1.md)의 재측정판이다. 그때와 다른 것:
현행 코퍼스(corpus_v2 + 하드 문서)·현행 gold(v2+hard95)·'파일명>헤딩' 프리픽스 도입 후.
가중 RRF(0.8/0.2)+상위30 컷 조립은 1차 실행에서 **결함으로 판명**(BM25_INJECT 주석 참조)
— 하이브리드의 실제 가설은 "dense가 놓친 정답의 후보 주입"이라 union으로 잰다.
스크립트를 저장소에 두는 이유: 구판 스크립트가 scratchpad에서 유실돼 재작성했던 교훈.

변형 (#133 토크나이저 A/B — 전부 리랭커 on = 운영 동일 조건):
  baseline      dense30 → 리랭크 → top20              (eval/retrieval_v2.py와 동일 경로)
  hyb_bigram    dense30 + BM25(어절 내 bigram) 주입 15 → 리랭크 → top20
  hyb_kiwi      dense30 + BM25(kiwi 형태소, 내용어만) 주입 15 → 리랭크 → top20
  + 채널 단독 회수력(top30 리콜·주입 상방)은 리랭크 없이 별도 계측
kiwi(kiwipiepy)는 eval 전용 선택 의존 — 미설치면 kiwi 채널만 스킵.

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

from sqlalchemy import select

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
    """어절 내 bigram (#133 개선 — 구판은 공백 제거 후 절단이라 "반품 기간"→"품기" 같은
    어절 경계를 넘는 가짜 조각을 만들었다). 1글자 어절은 그대로 토큰."""
    out = []
    for w in text.lower().split():
        if len(w) < 2:
            out.append(w)
        else:
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out or [text.lower()]


_kiwi = None


def _kiwi_tokens(text: str) -> list[str]:
    """kiwi 형태소 토큰 (#133 토크나이저 A/B). 조사·어미·구두점 제거, 내용어만.

    지연 import — kiwipiepy는 eval 전용 선택 의존(requirements 미등재). 미설치면
    호출부(main)가 kiwi 채널을 스킵한다.
    """
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    drop = ('J', 'E', 'SF', 'SP', 'SS', 'SE', 'SO', 'SW')   # 조사·어미·구두점류
    return [t.form.lower() for t in _kiwi.tokenize(text)
            if not t.tag.startswith(drop)] or [text.lower()]


class Bm25:
    """테넌트 코퍼스 하나에 대한 Okapi BM25. 코퍼스가 작아(테넌트당 ~100청크) 전수 채점."""

    def __init__(self, docs: dict[int, str], k1: float = 1.5, b: float = 0.75,
                 tokenize=_bigrams):
        self.k1, self.b = k1, b
        self.tokenize = tokenize
        self.tf: dict[int, dict[str, int]] = {}
        self.dl: dict[int, int] = {}
        df: dict[str, int] = defaultdict(int)
        for cid, text in docs.items():
            toks = self.tokenize(text)
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
        q = self.tokenize(query)
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

    variants = ["baseline", "hyb_bigram", "hyb_kiwi"]
    rows: dict[str, list[dict]] = {v: [] for v in variants}
    skipped = 0
    # 토크나이저 A/B (#133): 채널 단독 회수력 + 주입 상방을 리랭크 없이 계측
    chan_stats = {c: {"recall30": 0, "uplift": 0, "uplift_ids": []}
                  for c in ("bigram", "kiwi")}
    try:
        _kiwi_tokens("초기화")
        kiwi_ok = True
    except ImportError:
        kiwi_ok = False
        print("kiwipiepy 미설치 — kiwi 채널 스킵")

    async with AsyncSessionLocal() as session:
        for tenant, items in by_tenant.items():
            corpus = await _load_corpus(session, tenant)
            channels = {"bigram": Bm25(corpus, tokenize=_bigrams)}
            if kiwi_ok:
                channels["kiwi"] = Bm25(corpus, tokenize=_kiwi_tokens)
            resolved = await resolve_gold(session, tenant, items)
            print(f"[{tenant}] 청크 {len(corpus)} · 문항 {len(items)}", flush=True)
            for g in items:
                gold_ids = set(resolved.chunk_ids.get(g["id"]) or [])
                if not gold_ids:
                    skipped += 1
                    continue
                q = g["query"]
                q_embs = await embed_texts([q])
                per_query_ids, _ = await _search_dense_per_query(session, tenant, q_embs, POOL)
                dense_ids = per_query_ids[0]
                dense_hit = bool(gold_ids & set(dense_ids))

                cand = {"baseline": dense_ids}
                for cname, bm in channels.items():
                    ch_ids = bm.top(q, POOL)
                    st = chan_stats[cname]
                    st["recall30"] += bool(gold_ids & set(ch_ids))
                    if not dense_hit and (gold_ids & set(ch_ids)):
                        st["uplift"] += 1
                        st["uplift_ids"].append(g["id"])
                    injected = [c for c in ch_ids if c not in set(dense_ids)][:BM25_INJECT]
                    cand[f"hyb_{cname}"] = dense_ids + injected

                for name in variants:
                    if name not in cand:
                        continue
                    got = await _final_ids(session, q, cand[name], use_rerank=True)
                    rows[name].append({"id": g["id"], "type": g["type"], "tenant": tenant,
                                       "scores": score_one(got, gold_ids)})

    if not any(rows.values()):
        raise SystemExit("결과 0행 — DB/코퍼스 상태 확인 (파일 미저장)")

    out_rows = [{"variant": v, **r} for v in variants for r in rows[v]]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))

    print(f"\n측정 {datetime.now():%Y-%m-%d %H:%M} · 대상 {len(rows['baseline'])}문항 (skip {skipped})")
    print(f"{'variant':<13}{'R@5':>7}{'Hit@1':>8}{'MRR':>7}{'R@5 hard':>10}{'Hit@1 hard':>12}")
    for v in variants:
        if not rows[v]:
            continue
        a = _agg(rows[v])
        hard = a["by_difficulty"].get("hard_new", {})
        print(f"{v:<13}{a['recall_at_5']:>7.3f}{a['hit_at_1']:>8.3f}{a['mrr']:>7.3f}"
              f"{hard.get('recall_at_5', 0):>10.3f}{hard.get('hit_at_1', 0):>12.3f}")

    n = len(rows["baseline"])
    print("\n[토크나이저 채널 단독 비교 — 리랭크 무관 회수력]")
    for cname, st in chan_stats.items():
        if cname == "kiwi" and not kiwi_ok:
            continue
        print(f"  {cname:<7} 단독 리콜(top30): {st['recall30']}/{n} ({st['recall30']/n:.1%})"
              f" · 주입 상방(dense 실패 구제): {st['uplift']}건 {st['uplift_ids']}")
    print(f"행 저장: {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
