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

`--os` (#139, OpenSearch A/B): 검색 계층만 OpenSearch로 바꾼 변형을 같은 표에 얹는다.
같은 청크·같은 벡터(재계산 금지)·같은 리랭커·같은 채점이므로 **검색 계층만이 변인**이다.
  os_knn          OS kNN 30 → 리랭크 → top20            (dense 채널 대체)
  hyb_os_bigram   dense30 + OS BM25(bigram) 주입 15      (BM25 구현만 다름 — 토큰 동일)
  hyb_os_nori     dense30 + OS BM25(Nori) 주입 15        (토크나이저만 다름)
  os_hybrid       OS hybrid(kNN+BM25 Nori) + 정규화 융합 30
읽는 법 — 각 대조가 한 변인만 분리한다:
  os_knn        vs baseline       → kNN 구현 차이 (pgvector HNSW vs OS lucene HNSW)
  hyb_os_bigram vs hyb_bigram     → BM25 구현 차이 (Lucene vs 앱 계산). 토큰 동일
  hyb_os_nori   vs hyb_os_bigram  → 토크나이저 차이 (Nori vs bigram). BM25 엔진 동일
  os_hybrid     vs os_knn         → 엔진 융합의 상방
게이트 3(kNN 동형성)을 먼저 찍는다 — os_knn 후보가 dense 후보와 거의 같아야 한다(같은 벡터·
같은 코사인·같은 m/ef_construction·같은 색인 집합). 크게 벌어지면 품질 차이가 아니라 배선
오류다. 매핑·색인·잔차의 근거는 eval/_os_backend.py, 색인·나머지 게이트는 eval/_os_index.py.

BM25 = 순수 Python Okapi(k1=1.5, b=0.75), 토크나이저 = 문자 bigram(pg_bigm 프리뷰 —
구 어블레이션과 동일). 입력 텍스트 = 임베딩과 동일한 build_index_text(파일명>헤딩+본문).
후보 코퍼스 = dense와 같은 _searchable_condition 풀 (변인 격리).

채점은 retrieval_v2와 동일 계약: gold 정본만, resolve_gold는 테넌트 전체 배치(부분 배치는
gold_ids 분포를 바꾼다 — resolve_gold docstring), score_one(ks=(5,20)), 슬라이스는
리랭크·융합 뒤 한 번, _keep_single_table 동일 위치. baseline 파일은 덮어쓰지 않고
eval/results/hybrid_ablation_v2.jsonl 에 변형별 행을 저장한다.

실행: python -m eval._hybrid_ablation           (의존: DB + TEI 임베딩·리랭커. LLM 불필요)
      python -m eval._hybrid_ablation --os      (+ OpenSearch — eval._os_index 선행 필요)
"""
import argparse
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
from rag.lexical import bigrams as _bigrams   # 정의점은 운영 코드(#135) — 측정과 운영이 같은 함수
from rag.models import Chunk, Document, Faq, Folder
from rag.reranker import rerank
from rag.retriever import (_fetch_chunk_map, _keep_single_table, _search_dense_per_query,
                           _searchable_condition)

OUT = Path(__file__).resolve().parent / "results" / "hybrid_ablation_v2.jsonl"
# --os는 별도 파일에 쓴다 — 기존 눈금 파일(#128 리포트의 근거)을 덮지 않는다.
OUT_OS = Path(__file__).resolve().parent / "results" / "os_ablation_v1.jsonl"
POOL = 30           # 채널별 후보 수 = candidates_per_branch 기본값과 동일
TOP_N = 20
# 가중 RRF+상위30 컷 변형은 v2 1차 실행에서 조립 결함으로 판명 — dense 우세 가중에선
# 어휘 전용 후보의 최고 RRF 점수(w_b/(k+1))가 dense 꼴찌 점수(w_d/(k+POOL))보다 낮아
# 후보 풀이 baseline과 동일해진다(실측: 소수점 3자리까지 일치). 하이브리드의 실제 가설은
# "dense가 놓친 정답의 후보 주입"이므로 union이 옳은 조립이다: 최종 순위는 리랭커가 정한다.
BM25_INJECT = 15    # dense30에 없는 bm25 상위 주입 수 — 리랭커 풀 최대 45


# ── BM25 (Okapi, 문자 bigram) ────────────────────────────────────────────────


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


class OsBm25Channel:
    """OpenSearch BM25 어휘 채널 (#139) — Bm25와 같은 `.top(query, n)` 계약.

    같은 인터페이스로 맞춘 이유: 아래 측정 루프가 채널 종류를 몰라도 되게 해서, 앱 BM25와
    엔진 BM25가 **같은 위치·같은 주입 규칙**으로 비교되도록 하는 것이다.

    pretok=True면 질의를 rag.lexical.bigrams()로 잘라 공백으로 이어 넘긴다 — 필드
    analyzer가 whitespace라 색인 토큰과 바이트 단위로 같아진다(_os_backend.LEX_BIGRAM_FIELD 주석).
    """

    def __init__(self, os_client, tenant: str, field: str, *, pretok: bool):
        self.c, self.tenant, self.field, self.pretok = os_client, tenant, field, pretok

    def top(self, query: str, n: int) -> list[int]:
        import eval._os_backend as B
        q = " ".join(_bigrams(query)) if self.pretok else query
        return B.bm25_ids(self.c, self.tenant, self.field, q, n)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--os", action="store_true",
                    help="OpenSearch 변형 추가 (#139). eval._os_index 선행 색인 필요")
    args = ap.parse_args()

    gold = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    target = [g for g in gold if g["type"] in TYPES]
    by_tenant = defaultdict(list)
    for g in target:
        by_tenant[row_tenant(g)].append(g)

    variants = ["baseline", "hyb_bigram", "hyb_kiwi"]
    os_client = None
    if args.os:
        import eval._os_backend as B
        os_client = B.client()
        if not os_client.indices.exists(B.INDEX):
            raise SystemExit(f"인덱스 {B.INDEX} 없음 — python -m eval._os_index --recreate 먼저")
        B.ensure_pipeline(os_client)
        variants += ["os_knn", "hyb_os_bigram", "hyb_os_nori", "os_hybrid"]
        print(f"OpenSearch {os_client.info()['version']['number']} · 인덱스 {B.INDEX}")
    rows: dict[str, list[dict]] = {v: [] for v in variants}
    skipped = 0
    # 게이트 3 — kNN 동형성. os_knn 후보가 dense 후보와 거의 같아야 한다(같은 벡터·같은 코사인·
    # 같은 HNSW 파라미터·같은 색인 집합). 벌어지면 품질 차이가 아니라 배선 오류다.
    knn_gate = {"n": 0, "overlap": 0, "top1_same": 0}
    # 토크나이저 A/B (#133): 채널 단독 회수력 + 주입 상방을 리랭크 없이 계측
    chan_names = ["bigram", "kiwi"] + (["os_bigram", "os_nori"] if args.os else [])
    chan_stats = {c: {"recall30": 0, "uplift": 0, "uplift_ids": []} for c in chan_names}
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
            if os_client is not None:
                channels["os_bigram"] = OsBm25Channel(
                    os_client, tenant, B.LEX_BIGRAM_FIELD, pretok=True)
                channels["os_nori"] = OsBm25Channel(
                    os_client, tenant, B.NORI_FIELD, pretok=False)
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
                if os_client is not None:
                    vec = list(q_embs[0].dense)
                    os_knn = B.knn_ids(os_client, tenant, vec, POOL)
                    cand["os_knn"] = os_knn
                    cand["os_hybrid"] = B.hybrid_ids(
                        os_client, tenant, vec, q, B.NORI_FIELD, POOL)
                    knn_gate["n"] += 1
                    knn_gate["overlap"] += len(set(dense_ids) & set(os_knn)) / max(len(dense_ids), 1)
                    knn_gate["top1_same"] += bool(dense_ids and os_knn
                                                  and dense_ids[0] == os_knn[0])
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

    if knn_gate["n"]:
        ov = knn_gate["overlap"] / knn_gate["n"]
        t1 = knn_gate["top1_same"] / knn_gate["n"]
        # 임계 0.95: 같은 벡터·같은 코사인·같은 m/ef_construction·같은 색인 집합이므로 후보가
        # 거의 같아야 한다. 남는 차이는 HNSW 탐색의 근사성(ef_search 기본값이 두 엔진에서
        # 다르다)뿐이다. 이보다 낮으면 space_type·정규화·필터 중 하나가 어긋난 것이다.
        verdict = "통과" if ov >= 0.95 else "실패 — 배선을 먼저 고칠 것"
        print(f"\n[게이트 3 — kNN 동형성] 후보 겹침 {ov:.3f} · top1 일치 {t1:.3f}"
              f" ({knn_gate['n']}문항) → {verdict}")
        if ov < 0.95:
            # **결과 파일을 쓰지 않고 멈춘다.** 콘솔 경고만 찍고 jsonl을 남기면, 나중에 그
            # 파일만 집어다 분석하는 사람이 배선 오류를 품질 차이로 오독한다. 게이트 통과
            # 전에는 수치를 남기지도 않는 것이 AGENTS.md 측정 규율이다(_os_index도 동일).
            raise SystemExit(
                "게이트 3 실패 — os_knn 후보가 dense 후보와 다르다. 결과를 저장하지 않았다.\n"
                "  space_type(cosinesimil)·벡터 정규화·테넌트 필터·ef_search를 확인할 것.")

    out_path = OUT_OS if args.os else OUT
    out_rows = [{"variant": v, **r} for v in variants for r in rows[v]]
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows))

    print(f"\n측정 {datetime.now():%Y-%m-%d %H:%M} · 대상 {len(rows['baseline'])}문항 (skip {skipped})")
    print(f"{'variant':<15}{'R@5':>7}{'Hit@1':>8}{'MRR':>7}{'R@5 hard':>10}{'Hit@1 hard':>12}")
    for v in variants:
        if not rows[v]:
            continue
        a = _agg(rows[v])
        hard = a["by_difficulty"].get("hard_new", {})
        print(f"{v:<15}{a['recall_at_5']:>7.3f}{a['hit_at_1']:>8.3f}{a['mrr']:>7.3f}"
              f"{hard.get('recall_at_5', 0):>10.3f}{hard.get('hit_at_1', 0):>12.3f}")

    n = len(rows["baseline"])
    print("\n[채널 단독 비교 — 리랭크 무관 회수력]")
    for cname, st in chan_stats.items():
        if cname == "kiwi" and not kiwi_ok:
            continue
        print(f"  {cname:<7} 단독 리콜(top30): {st['recall30']}/{n} ({st['recall30']/n:.1%})"
              f" · 주입 상방(dense 실패 구제): {st['uplift']}건 {st['uplift_ids']}")
    print(f"행 저장: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
