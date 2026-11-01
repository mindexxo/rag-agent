"""하이브리드 가중치 검증의 캡처 단계 — 질문셋을 모아 질문마다 검색 채널 값·리랭커 점수를 떠서 캡처 디렉터리에 쓰고 읽는다."""

import asyncio
import dataclasses
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import rag.search.reranker
from config import settings
from eval import hybrid_replay as hr
from eval.axis_common import commit, load_chunks
from rag.identifiers import normalize_identifiers
from rag.search import os_search
from rag.search.embeddings import Embedding, embed_texts
from rag.search.os_client import NORI_FIELD, client
from rag.search.retrieval_types import RetrievedChunk

RESULT_DIR = Path("eval/results")
CONCURRENCY = 6  # TEI 공유 장비 — retrieval_article과 같은 안전선
DEPTH = 100  # 진단용 깊이(주입 상방 N=30·50·100)
POOL_M = 50  # 리랭커 점수를 받는 상위집합 = 벡터·BM25 각 상위 POOL_M ∪ 운영 풀 — 후보 조건이 이 안에서 움직이게
SETS = ("v2", "keyword", "article")


def settings_snapshot() -> dict[str, Any]:
    """재생이 전제하는 운영 설정. 캡처 때와 지금이 다르면 재생·확인을 거부한다(BASELINE이 이 값을 가정한다)."""
    return {
        k: getattr(settings, k)
        for k in (
            "rerank_enabled",
            "hybrid_lexical_enabled",
            "hybrid_lexical_inject",
            "ident_channel_enabled",
            "ident_inject",
            "ident_boost",
        )
    }


# ----- 질문셋 -------------------------------------------------------------


async def load_v2() -> list[dict[str, Any]]:
    """retrieval_v2와 같은 문항·정답(정답 = 스니펫 포함 조각 전부) — 같은 공용부(resolve_by_tenant·scorable)로 뽑는다."""
    from database import (
        AsyncSessionLocal,  # DB 세션을 여는 건 v2 캡처뿐이다(모듈 import 자체는 hybrid_replay 경유로 이미 됨)
    )
    from eval.generation import load_gold, row_tenant
    from eval.retrieval import resolve_by_tenant, scorable
    from eval.retrieval_v2 import TYPES

    items = []
    async with AsyncSessionLocal() as session:
        target = [g for g in load_gold() if g["type"] in TYPES]
        async for tenant, rows, resolved in resolve_by_tenant(session, target, row_tenant):
            for g, gold in scorable(rows, resolved)[0]:
                items.append(
                    {
                        "set": "v2",
                        "qid": g["id"],
                        "tenant": tenant,
                        "type": g["type"],
                        "query": g["query"],
                        "gold_ids": gold,
                        "token": None,
                    }
                )
    return items


async def load_keyword() -> list[dict[str, Any]]:
    from eval.retrieval_keyword import generate_items

    items, _ = await generate_items()
    return [{k: it[k] for k in ("set", "qid", "tenant", "type", "query", "gold_ids", "token")} for it in items]


async def load_article() -> list[dict[str, Any]]:
    from eval import retrieval_article as ra

    items = []
    for tenant in ra.TENANTS:
        chunks = await load_chunks(tenant)
        gold, _ = ra.build_gold(chunks)
        for i, it in enumerate(ra.build_items(chunks, gold, tenant)):
            items.append(
                {
                    "set": "article",
                    "qid": f"ar_{tenant}_{i:04d}",
                    "tenant": tenant,
                    "type": "article",
                    "query": it["query"],
                    "gold_ids": it["gold_ids"],
                    "token": None,
                }
            )
    return items


LOADERS = {"v2": load_v2, "keyword": load_keyword, "article": load_article}


# ----- 캡처 ---------------------------------------------------------------


async def _bm25_scored(tenant: str, query: str, n: int) -> list[list]:
    """BM25 상위 n의 (id, 점수). 운영 search_lexical은 id만 돌려주므로 같은 절(lex_clause·tenant_filter·NORI_FIELD)로 한 번 더
    묻는다 — 운영 질의 바디의 정의점: rag/search/os_search.search_lexical. 어긋나면 verify ①이 잡는다(BM25 30은 운영 함수로 받는다)."""
    resp = await client().search(
        index=settings.opensearch_index,
        body={
            "size": n,
            "_source": ["chunk_id"],
            "query": {
                "bool": {"filter": [os_search.tenant_filter(tenant)], "must": [os_search.lex_clause(NORI_FIELD, query)]}
            },
        },
    )
    return [[int(h["_source"]["chunk_id"]), float(h["_score"])] for h in resp["hits"]["hits"]]


async def capture_one(it: dict, vec: list[float]) -> tuple[dict, dict[int, RetrievedChunk]]:
    tenant, query, emb = it["tenant"], it["query"], [Embedding(dense=vec)]
    dense30 = (await os_search.search_dense_per_query(tenant, emb, 30)).top  # 운영 크기 — 운영 함수 그대로
    dense100 = (await os_search.search_dense_per_query(tenant, emb, DEPTH)).top
    bm25_30 = await os_search.search_lexical(tenant, query, 30)
    bm25_100 = await _bm25_scored(tenant, query, DEPTH)
    tokens = normalize_identifiers(query) if settings.ident_channel_enabled else []
    ident = await os_search.search_identifiers(tenant, tokens, settings.ident_inject) if tokens else []
    rec = {
        **it,
        "dense30": [list(x) for x in dense30],
        "dense100": [list(x) for x in dense100],
        "bm25_30": bm25_30,
        "bm25_100": bm25_100,
        "ident_tokens": tokens,
        "ident": ident,
    }
    superset = list(
        dict.fromkeys(hr.build_pool(rec) + [c for c, _ in dense100[:POOL_M]] + [c for c, _ in bm25_100[:POOL_M]])
    )
    # 정답 조각도 함께 받아 둔다(점수는 안 매김) — 시트 접기 판정(hybrid_replay.evaluate)이 정답 조각의 시트를 봐야 해서
    chunk_map = await os_search.fetch_chunk_map(list(dict.fromkeys(superset + list(it["gold_ids"]))))
    pool = [chunk_map[c] for c in superset if c in chunk_map]
    scores = await rag.search.reranker.rerank_scores(query, pool) if pool else []
    if scores is None:
        raise RuntimeError(f"{it['qid']}: 리랭커 실패 — 부분 성공은 쓰지 않는다(rerank_maxpool과 같은 규율)")
    rec["rerank"] = {str(c.chunk_id): float(s) for c, s in zip(pool, scores, strict=False)}
    return rec, chunk_map


async def _load_items(sets: list[str], limit: int | None) -> list[dict[str, Any]]:
    """셋마다 문항을 모은다(셋마다 앞 limit개). 문항 id가 겹치면 거부한다."""
    items = []
    for s in sets:
        got = (await LOADERS[s]())[: limit or None]
        items += got
        print(f"{s}: 문항 {len(got)}", flush=True)
    dup = [q for q, n in Counter(it["qid"] for it in items).items() if n > 1]
    if dup:  # 벡터·확인·비교가 전부 qid를 키로 쓴다 — 겹치면 조용히 덮어쓴다
        raise SystemExit(f"문항 id 중복: {dup[:5]}")
    return items


async def _capture_all(items: list[dict], vecs: list[list[float]]) -> list[tuple[dict, dict[int, RetrievedChunk]]]:
    sem = asyncio.Semaphore(CONCURRENCY)
    done = 0

    async def _one(it: dict, vec: list[float]) -> tuple[dict, dict[int, RetrievedChunk]]:
        nonlocal done
        async with sem:
            r = await capture_one(it, vec)
        done += 1
        if done % 200 == 0:
            print(f"  캡처 {done}/{len(items)}", flush=True)
        return r

    return list(await asyncio.gather(*(_one(it, v) for it, v in zip(items, vecs, strict=False))))


def _write_capture(
    out: Path, sets: list[str], items: list[dict], vecs: list[list[float]], results: list[tuple[dict, dict]]
) -> int:
    """meta·레코드·조각·질문 벡터를 out에 쓴다 → 조각 수."""
    chunks: dict[int, RetrievedChunk] = {}
    for _, cm in results:
        chunks.update(cm)
    (out / "meta.json").write_text(
        json.dumps(
            {
                "commit": commit(),
                "at": out.name,
                "settings": settings_snapshot(),
                "index": settings.opensearch_index,
                "depth": DEPTH,
                "pool_m": POOL_M,
                "sets": {s: sum(it["set"] == s for it in items) for s in sets},
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    (out / "records.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r, _ in results))
    (out / "chunks.jsonl").write_text(
        "\n".join(json.dumps(dataclasses.asdict(c), ensure_ascii=False) for c in chunks.values())
    )
    (out / "vectors.jsonl").write_text(
        "\n".join(json.dumps({"qid": it["qid"], "vec": v}) for it, v in zip(items, vecs, strict=False))
    )
    return len(chunks)


async def capture(sets: list[str], limit: int | None) -> Path:
    out = RESULT_DIR / f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out.mkdir(parents=True, exist_ok=True)
    items = await _load_items(sets, limit)
    vecs = []
    for i in range(0, len(items), 256):  # embed_texts가 32씩 나눠 보낸다
        vecs += [e.dense for e in await embed_texts([it["query"] for it in items[i : i + 256]], kind="query")]
    results = await _capture_all(items, vecs)
    n_chunks = _write_capture(out, sets, items, vecs, results)
    print(f"→ {out} (문항 {len(results)}, 조각 {n_chunks})")
    return out


type Capture = tuple[dict, list[dict], dict[int, RetrievedChunk], dict[str, list[float]]]  # meta·레코드·조각·질문 벡터


def load_capture(path: Path, *, need_vectors: bool = False) -> Capture:
    meta = json.loads((path / "meta.json").read_text())
    if meta["settings"] != settings_snapshot():
        raise SystemExit(
            f"설정이 캡처 때와 다르다 — 재생이 전제하는 운영 조건이 아니다: {meta['settings']} vs {settings_snapshot()}"
        )
    recs = [json.loads(ln) for ln in (path / "records.jsonl").read_text().splitlines() if ln]
    chunks = {
        d["chunk_id"]: RetrievedChunk(**d)
        for d in (json.loads(ln) for ln in (path / "chunks.jsonl").read_text().splitlines() if ln)
    }
    vecs = (
        {d["qid"]: d["vec"] for d in (json.loads(ln) for ln in (path / "vectors.jsonl").read_text().splitlines() if ln)}
        if need_vectors
        else {}
    )
    return meta, recs, chunks, vecs
