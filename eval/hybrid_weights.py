"""하이브리드 가중치 검증 — 캡처·동치 확인·진단 CLI. 캡처는 eval/hybrid_capture, 계산은 eval/hybrid_replay(순수부)에 있다.

질문: 키워드(BM25 Nori)·벡터 결합에 **가중치를 둘지 말지**. 기준선은 현행(가중치 없음: 벡터 30 + 키워드 주입 15 → 리랭커).
리랭커가 순서를 정하는 구조라 가중치가 품질을 바꿀 수 있는 자리는 둘뿐이다 — ① 리랭커 풀에 누가 들어가나 ② 리랭크 뒤
점수에 섞기. 그래서 먼저 현행에서 놓친 문항이 어느 쪽인지 진단하고(diagnose), 0이 아닌 쪽에 해당하는 후보만 비교한다.

질문셋 셋: v2(문장형 골드, retrieval_v2와 같은 문항·정답) / keyword(상담원식 짧은 질의, eval/retrieval_keyword) /
article(조 번호, eval/retrieval_article). 셋 다 단일 질의(멀티쿼리 확장 없음).

절차 (각 단계 결과는 eval/results/hybrid_<stamp>/ — gitignore, 실문서 원문이 들어간다):
  capture  질문마다 한 번 — 임베딩 벡터, 벡터 30·100(거리), BM25 30·100(점수), 식별자 후보, 후보 상위집합의 리랭커 점수,
           그 조각들. 벡터·BM25 30은 운영 크기를 **따로** 받는다(100의 앞 30을 운영 결과로 가정하지 않는다).
  verify   ① 운영 retrieve_candidates의 단계 함수(임베딩·벡터·BM25·식별자·본문 조회·리랭커)를 그 레코드의 캡처값으로 끼워
           실제로 돌린 결과 = 재생(assemble) — **전 문항 완전 일치여야 측정 진행**(조립 검사, 색인 변동과 무관).
           ② 끼우지 않은 실제 운영(표본)과 재생 비교 — 근접 동점(리랭커 흔들림)·색인 변동으로 설명되지 않는 차이 0건이어야 한다.
  diagnose 현행 조건에서 문항별로 hit / rerank_miss(풀엔 있는데 근거 5 밖) / pool_miss(풀에 없음), 주입 상방(BM25 30·50·100).
  compare  후보 조건(리랭크 뒤 BM25 점수 혼합 — hybrid_replay.CANDIDATES)을 현행과 짝 부트스트랩. 이긴·진 문항 id 포함.

필요: OpenSearch + TEI(임베딩·리랭커) + DB(v2 정답 해석). LLM 불필요.
사용: python -m eval.hybrid_weights capture [--sets v2,keyword,article] [--limit N]
     python -m eval.hybrid_weights verify   --capture eval/results/hybrid_<stamp> [--live 100]
     python -m eval.hybrid_weights diagnose --capture eval/results/hybrid_<stamp>
     python -m eval.hybrid_weights compare  --capture eval/results/hybrid_<stamp>
"""

import argparse
import asyncio
import contextlib
import contextvars
import json
import random
from collections.abc import Iterator
from pathlib import Path

import rag.search.reranker
import rag.search.retriever
from eval import hybrid_replay as hr
from eval.hybrid_capture import CONCURRENCY, SETS, capture, load_capture
from rag.search import os_search
from rag.search.embeddings import Embedding, Kind
from rag.search.os_client import close_client
from rag.search.retrieval_types import DenseHit, DenseHits, RetrievedChunk

LIVE_TIE_EPS = (
    5e-3  # 실제 운영과 재생이 다를 때 허용하는 근접 동점 — 리랭커 점수 차가 이보다 작은 두 후보의 자리바꿈만(설계 8)
)


# ----- 동치 확인 ------------------------------------------------------------

_CURRENT: contextvars.ContextVar[tuple[dict, dict, list[float]]] = contextvars.ContextVar("hybrid_current")


@contextlib.contextmanager
def _patched() -> Iterator[None]:
    """운영 retrieve_candidates의 **단계 함수만** 지금 레코드(_CURRENT)의 캡처값으로 바꾼다 — 조립·정렬·가산·자르기는 운영 코드가
    실제로 돈다. 그래서 ① 확인은 '재생 조립 = 운영 조립'만 보고, 공유 색인이 캡처 뒤에 바뀌어도(다른 세션의 업로드 — BM25 IDF는
    인덱스 전체라 남의 테넌트 문서에도 점수가 움직인다) 흔들리지 않는다. 레코드별 값은 ContextVar라 gather에도
    섞이지 않는다 — 질문 문장을 키로 쓰면 같은 문장이 두 셋에 있는 문항의 점수가 섞인다.
    재생이 요구하지 않은 후보를 운영이 요구하면 KeyError — 풀이 캡처 상위집합 밖이라는 뜻이다."""
    names = {
        (rag.search.retriever, "embed_texts"),
        (rag.search.retriever, "_dense_candidates"),
        (rag.search.retriever, "_lexical_candidates"),
        (rag.search.os_search, "search_identifiers"),
        (rag.search.retriever, "_chunk_map"),
        (rag.search.reranker, "rerank_scores"),
    }
    orig = {(m, n): getattr(m, n) for m, n in names}

    async def _embed(texts: list[str], kind: Kind = "document") -> list[Embedding]:
        return [Embedding(dense=_CURRENT.get()[2]) for _ in texts]

    async def _dense(tenant: str, q_embs: list, n: int) -> DenseHits:
        rec = _CURRENT.get()[0]
        assert n == 30, n
        return DenseHits([[c for c, _ in rec["dense30"]]], [DenseHit(*x) for x in rec["dense30"]])

    async def _lexical(tenant: str, query: str, limit: int) -> list[int]:
        assert limit == 30, limit
        return list(_CURRENT.get()[0]["bm25_30"])

    async def _ident(tenant: str, tokens: list[str], limit: int) -> list[int]:
        return list(_CURRENT.get()[0]["ident"])[:limit]

    async def _chunk_map(ids: list[int]) -> dict[int, RetrievedChunk]:
        chunks = _CURRENT.get()[1]
        return {c: chunks[c] for c in ids if c in chunks}

    async def _scores(query: str, chunks: list[RetrievedChunk]) -> list[float]:
        rerank = _CURRENT.get()[0]["rerank"]
        return [rerank[str(c.chunk_id)] for c in chunks]

    repl = {
        "embed_texts": _embed,
        "_dense_candidates": _dense,
        "_lexical_candidates": _lexical,
        "search_identifiers": _ident,
        "_chunk_map": _chunk_map,
        "rerank_scores": _scores,
    }
    for m, n in names:
        assert hasattr(m, n), f"운영 이음새 {m.__name__}.{n}가 없다 — 이름이 바뀌었으면 재생·확인을 함께 고칠 것"
        setattr(m, n, repl[n])
    try:
        yield
    finally:
        for (m, n), f in orig.items():
            setattr(m, n, f)


async def _production_ids(rec: dict) -> list[int]:
    # 세션은 쓰이지 않는다 — 세션을 쓰는 단계(_chunk_map)는 _patched가 캡처값으로 바꿔 둔다
    cands = await rag.search.retriever.retrieve_candidates(None, rec["tenant"], rec["query"], top_n=hr.TOP_N)  # type: ignore[arg-type]
    return [c.chunk_id for c in cands.chunks]


def _first_diff(a: list[int], b: list[int]) -> int | None:
    return next(
        (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
        None if len(a) == len(b) else min(len(a), len(b)),
    )


async def _verify_patched(
    recs: list[dict], chunks: dict[int, RetrievedChunk], vecs: dict[str, list[float]], sem: asyncio.Semaphore
) -> tuple[dict[str, list[int]], list[tuple[str, int | None]]]:
    """① 캡처값을 끼운 운영 조립 vs 재생 → (문항별 재생 순서, 불일치 (qid, 첫 차이 위치))."""

    async def _patched_one(r: dict) -> tuple[dict, list[int]]:
        async with sem:
            _CURRENT.set((r, chunks, vecs[r["qid"]]))
            return r, await _production_ids(r)

    with _patched():
        got = await asyncio.gather(*(_patched_one(r) for r in recs))
    replayed = {r["qid"]: [c.chunk_id for c in hr.assemble(r, chunks)] for r in recs}
    bad = [(r["qid"], _first_diff(replayed[r["qid"]], ids)) for r, ids in got if replayed[r["qid"]] != ids]
    print(f"① 캡처값을 끼운 운영 조립 = 재생: {len(recs) - len(bad)}/{len(recs)} 일치")
    for qid, pos in bad[:20]:
        print(f"   불일치 {qid} — 첫 차이 {pos}위")
    return replayed, bad


def _near_tie(r: dict, mine: list[int], ids: list[int]) -> bool:
    """근접 동점 = 어긋난 **모든** 자리에서 두 후보의 리랭커 점수 차 < LIVE_TIE_EPS(첫 자리만 보면 뒤의 실질 차이를 놓친다).
    상위 20 경계에서 점수가 0에 가까운 후보끼리 들고 나는 것도 여기 든다."""
    diffs = [(a, b) for a, b in zip(mine, ids, strict=False) if a != b]
    return len(mine) == len(ids) and all(
        str(a) in r["rerank"]
        and str(b) in r["rerank"]
        and abs(r["rerank"][str(a)] - r["rerank"][str(b)]) < LIVE_TIE_EPS
        for a, b in diffs
    )


async def _verify_live(
    recs: list[dict], vecs: dict[str, list[float]], replayed: dict[str, list[int]], live_n: int, sem: asyncio.Semaphore
) -> dict:
    """② 아무것도 끼우지 않은 실제 운영(표본) vs 재생. 차이는 셋 중 하나로 가른다 — 근접 동점(리랭커 흔들림), 색인 변동(캡처 뒤
    벡터·BM25 30이 달라짐 — 공유 색인), 설명 안 됨. 합격은 '설명 안 됨 0'이다: 리랭커 흔들림만으로 근접 동점 자리바꿈이 표본의
    10%대다 — 완전 일치 비율을 기준으로 두면 TEI 잡음을 결함으로 오판한다."""
    sample = random.Random(248).sample(recs, min(live_n, len(recs)))

    async def _live_one(r: dict) -> tuple[dict, list[int], bool]:
        async with sem:
            emb = [Embedding(dense=vecs[r["qid"]])]
            d30 = (await os_search.search_dense_per_query(r["tenant"], emb, 30)).top
            b30 = await os_search.search_lexical(r["tenant"], r["query"], 30)
            drift = [c for c, _ in d30] != [c for c, _ in r["dense30"]] or b30 != r["bm25_30"]
            return r, await _production_ids(r), drift

    live = await asyncio.gather(*(_live_one(r) for r in sample))
    exact = near_tie = drifted = 0
    unexplained = []
    for r, ids, drift in live:
        mine = replayed[r["qid"]]
        if mine == ids:
            exact += 1
        elif drift:
            drifted += 1
        elif _near_tie(r, mine, ids):
            near_tie += 1
        else:
            unexplained.append((r["qid"], _first_diff(mine, ids)))
    print(
        f"② 실제 운영(표본 {len(live)}) = 재생: 완전 일치 {exact}, 근접 동점 뒤바뀜 {near_tie}, 색인 변동 {drifted}, "
        f"설명 안 됨 {len(unexplained)}"
    )
    for u in unexplained[:20]:
        print(f"   설명 안 됨 {u}")
    return {"n": len(live), "exact": exact, "near_tie": near_tie, "index_drift": drifted, "unexplained": unexplained}


async def verify(path: Path, live_n: int) -> bool:
    _, recs, chunks, vecs = load_capture(path, need_vectors=True)
    sem = asyncio.Semaphore(CONCURRENCY)
    replayed, bad = await _verify_patched(recs, chunks, vecs, sem)
    live = await _verify_live(recs, vecs, replayed, live_n, sem)
    ok = not bad and not live["unexplained"]
    print("동치 확인 통과" if ok else "동치 확인 실패 — 측정하지 말 것")
    (path / "verify.json").write_text(
        json.dumps(
            {"patched": {"n": len(recs), "match": len(recs) - len(bad), "mismatch": bad}, "live": live, "ok": ok},
            ensure_ascii=False,
            indent=1,
        )
    )
    return ok


# ----- 진단 ---------------------------------------------------------------


def run_diagnose(path: Path) -> Path:
    meta, recs, chunks, _ = load_capture(path)
    rows = [
        {
            **hr.diagnose(r, chunks),
            "set": r["set"],
            "type": f"{r['set']}:{r['type']}",
            "qid": r["qid"],
            "tenant": r["tenant"],
        }
        for r in recs
    ]
    summary = {"by_type": hr.summarize_diagnosis(rows, "type"), "by_set": hr.summarize_diagnosis(rows, "set")}
    out = path / "diagnosis.json"
    out.write_text(json.dumps({"meta": meta, "summary": summary, "rows": rows}, ensure_ascii=False, indent=1))
    print(
        f"{'유형':28} {'n':>4} {'적중':>5} {'리랭커누락':>8} {'풀누락':>6} {'1위아님':>7}  상방(BM25 30/50/100)  "
        f"주입이살림/망침  벡터1위·5안  BM25 1위·5안"
    )
    for part in ("by_set", "by_type"):
        for k, v in summary[part].items():
            print(
                f"{k:28} {v['n']:>4} {v['hit']:>5} {v['rerank_miss']:>8} {v['pool_miss']:>6} {v['top1_miss']:>7}  "
                f"{v['upside_bm25_30']}/{v['upside_bm25_50']}/{v['upside_bm25_100']:<14}  {v['lexical_saves']:>4}/{v['lexical_hurts']:<6}"
                f"  {v['dense_top1']:.3f}·{v['dense_top5']:.3f}  {v['bm25_top1']:.3f}·{v['bm25_top5']:.3f}"
            )
        print()
    print(f"→ {out}")
    return out


def run_compare(path: Path) -> Path:
    """3단계 — 후보 조건(hybrid_replay.CANDIDATES, 결과 보기 전 고정)을 현행과 짝 비교. 주 지표 in_evidence, 보조 hit_at_1."""
    meta, recs, chunks, _ = load_capture(path)
    res = {
        "by_set": hr.compare(recs, chunks, key="set"),
        "by_type": hr.compare(recs, chunks, key="type", metrics=("in_evidence",), n_boot=2000),
    }
    out = path / "compare.json"
    out.write_text(json.dumps({"meta": meta, **res}, ensure_ascii=False, indent=1))
    print(f"{'조건':22} {'묶음':10} {'지표':12} {'차이(평균)':>10}  {'95% 신뢰구간':^22} {'이김':>4} {'짐':>4}")
    for cond, groups in res["by_set"].items():
        for g, ms in groups.items():
            for m, v in ms.items():
                lo, hi = v["ci95"]
                print(
                    f"{cond:22} {g:10} {m:12} {v['mean_diff']:+10.4f}  [{lo:+.4f}, {hi:+.4f}] {v['wins']:>4} {v['losses']:>4}"
                )
        print()
    print(f"→ {out}")
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--sets", default=",".join(SETS))
    c.add_argument("--limit", type=int, help="셋마다 앞 N문항만(스모크·시간 측정)")
    v = sub.add_parser("verify")
    v.add_argument("--capture", required=True, type=Path)
    v.add_argument("--live", type=int, default=100)
    d = sub.add_parser("diagnose")
    d.add_argument("--capture", required=True, type=Path)
    cp = sub.add_parser("compare")
    cp.add_argument("--capture", required=True, type=Path)
    args = ap.parse_args()
    if args.cmd == "capture":
        await capture(args.sets.split(","), args.limit)
    elif args.cmd == "verify":
        if not await verify(args.capture, args.live):
            raise SystemExit(1)
    elif args.cmd == "diagnose":
        run_diagnose(args.capture)
    else:
        run_compare(args.capture)


async def _run() -> None:
    try:
        await main()
    finally:
        await close_client()


if __name__ == "__main__":
    asyncio.run(_run())
