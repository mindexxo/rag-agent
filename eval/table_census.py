"""표 전수 조사(#232) — 색인 조각을 테넌트별로 읽어 문서마다 표 조각·큰 표·작은 표·표 행 수를 세고 코퍼스 × 형식으로 묶는다.

표 조각 판정은 eval.axis_common.is_table_chunk(큰 표 표식 `is_table` 또는 마크다운 구분선)다. 큰 표는 `table_key`로 세고(묶음이
여럿이라도 표 하나), 작은 표는 구분선 수로 센다. 행 수는 마크다운 행에서 헤더·구분선을 뺀 것 — 큰 표는 묶음마다 헤더가
반복되므로(rag.chunking.batch_table_rows) 묶음 단위로 뺀다. 재인제스트 전 구판 색인은 `table_key`가 없어(#257) 큰 표가
문서당 하나로 뭉쳐 보인다 — 그런 수치가 나오면 재색인부터.

용법: python -m eval.table_census [--tenants summers homeplus ...] [--out eval/results/table_census.jsonl]
문서별 행(파일명 포함)은 --out 파일에만 쓴다 — 실문서 파일명이라 저장소에 올리지 않는다(eval/results는 gitignore).
화면에는 코퍼스(inticube/그 외) × 형식 집계만 낸다.
"""

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

from config import settings
from eval.axis_common import TABLE_SEP, commit, is_table_chunk
from eval.generation import V2_TENANTS
from rag.search.os_client import client
from rag.search.os_search import tenant_filter

_ROW = re.compile(r"^\|.*\|\s*$", re.M)


async def _chunks(tenant: str) -> list[dict]:
    """조각 전부 — axis_common.load_chunks는 is_table·table_key를 안 읽어 여기서 따로 읽는다."""
    os_ = client()
    body = {
        "size": 1000,
        "_source": ["filename", "text", "is_table", "table_key"],
        "query": tenant_filter(tenant),
        "sort": [{"_doc": "asc"}],
    }
    resp = await os_.search(index=settings.opensearch_index, body=body, scroll="2m")
    out = []
    while resp["hits"]["hits"]:
        out += [h["_source"] for h in resp["hits"]["hits"]]
        resp = await os_.scroll(scroll_id=resp["_scroll_id"], scroll="2m")
    return [s for s in out if s.get("filename")]  # FAQ 제외


def census(tenant: str, chunks: list[dict]) -> list[dict]:
    """문서별 집계 행. 순수 함수 — 테스트가 조각 dict만 먹인다."""
    docs = defaultdict(lambda: {"chunks": 0, "table_chunks": 0, "big": set(), "small": 0, "rows": 0})
    for s in chunks:
        d = docs[s["filename"]]
        d["chunks"] += 1
        if not is_table_chunk(SimpleNamespace(is_table=s.get("is_table"), text=s.get("text"))):
            continue
        text = s.get("text") or ""
        seps = len(TABLE_SEP.findall(text))
        d["table_chunks"] += 1
        d["rows"] += len(_ROW.findall(text)) - 2 * seps
        if s.get("is_table"):
            d["big"].add(s.get("table_key"))
        else:
            d["small"] += seps
    return [
        {
            "tenant": tenant,
            "file": fn,
            "ext": fn.rsplit(".", 1)[-1].lower(),
            "chunks": d["chunks"],
            "table_chunks": d["table_chunks"],
            "big": len(d["big"]),
            "small": d["small"],
            "rows": d["rows"],
        }
        for fn, d in sorted(docs.items())
    ]


def summary(rows: list[dict]) -> str:
    agg = defaultdict(Counter)
    for o in rows:
        a = agg[("inticube" if o["tenant"] == "inticube" else "모의", o["ext"])]
        a["docs"] += 1
        a["with_table"] += o["table_chunks"] > 0
        for k in ("big", "small", "table_chunks", "chunks", "rows"):
            a[k] += o[k]
    out = [
        "| 코퍼스 | 형식 | 문서 | 표 있는 문서 | 큰 표 | 작은 표 | 표 조각 | 조각 | 표 행 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for (corpus, ext), a in sorted(agg.items()):
        out.append(
            f"| {corpus} | {ext} | {a['docs']} | {a['with_table']} | {a['big']} | {a['small']} | {a['table_chunks']} | "
            f"{a['chunks']} | {a['rows']} |"
        )
    return "\n".join(out)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tenants", nargs="*", default=list(V2_TENANTS))
    ap.add_argument("--out", default="eval/results/table_census.jsonl")
    args = ap.parse_args()
    rows = []
    for t in args.tenants:
        rows += census(t, await _chunks(t))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    print(f"커밋 {commit()} · 인덱스 {settings.opensearch_index} · 테넌트 {' '.join(args.tenants)}\n")
    print(summary(rows))
    print(f"\n문서별 → {out}")


if __name__ == "__main__":

    async def _run():
        try:
            await main()
        finally:
            from rag.search import os_client

            await os_client.close_client()

    asyncio.run(_run())
