import asyncio
import json
import re
import time
from collections import defaultdict
from dataclasses import field, dataclass
from pathlib import Path

from sqlalchemy import select

from config import settings
from database import AsyncSessionLocal
from rag.embeddings import embed_query
from rag.models import Document, Chunk
from rag.retriever import apply_gate, retrieve_candidates

# 게이트 임계값 sweep 후보 (max_dense_distance)
SWEEP = [0.4, 0.5, 0.6, 0.7]

TENANT = "demo"
GOLD = Path("eval/gold_set_v1.jsonl")
FINGERPRINT = Path("eval/corpus_fingerprint_v1.json")
RESULT_DIR = Path("eval/results")
# 검색 지표 적용 subset (§4.5 — 일반 품질 본체)
RETRIEVAL_TYPES = {"single_fact", "paraphrase", "rare_lexical", "multi_doc"}


@dataclass
class Resolved:
    """gold의 안정 키를 현재 DB id로 변환한 결과 묶음."""
    doc_ids: dict[str, int] = field(default_factory=dict)
    chunk_ids: dict[str, list[int]] = field(default_factory=dict)
    stale: list[str] = field(default_factory=list)


# 안정 키 -> 현재 DB id
async def resolve_gold(session, tenant_id, gold_rows) -> Resolved:
    """gold set의 안정 키(filename/heading_path/snippet)를 현재 DB id로 resolve.

    gold는 DB id가 아닌 내용 기준 안정 키로 정답을 적어둔다 (재인제스천마다 id가
    바뀌므로 — §4.1). 하지만 Recall/Hit/MRR 계산은 검색 결과의 DB id와 비교해야
    하므로, 평가 본 계산 전에 안정 키 -> 현재 id 변환이 선행돼야 한다.

    resolve 실패(문서 개정·청킹 변경으로 라벨이 낡음)는 조용히 0점 처리하지 않고
    stale에 모아 경보로 올린다 — "검색 실패"와 "라벨 노후"를 구분하기 위함.
    """
    r = Resolved()

    # 1) filename -> active/ready doc id  (한 번에 조회)
    # expected_docs 원소는 str 또는 [str,...] (대체 출처 그룹) — 평탄화해 조회
    filenames = {fn for row in gold_rows for d in row["expected_docs"]
                 for fn in ([d] if isinstance(d, str) else d)}
    doc_rows = (await session.execute(
        select(Document.filename, Document.id)
        .where(Document.tenant_id == tenant_id)
        .where(Document.is_active.is_(True))
        .where(Document.status == "ready")
        .where(Document.filename.in_(filenames))
    )).all()
    r.doc_ids = {fn: did for fn, did in doc_rows}

    # 2) expected_chunks -> chunk id (heading_path 일치 + text 에 snippet 포함)
    for row in gold_rows:
        cids = []
        for ec in row.get("expected_chunks", []):
            doc_id = r.doc_ids.get(ec["filename"])
            if doc_id is None:
                r.stale.append(f'{row["id"]}: doc 못찾음 {ec["filename"]}')
                continue
            chunk_rows = (await session.execute(
                select(Chunk.id, Chunk.heading_path, Chunk.text)
                .where(Chunk.document_id == doc_id)
            )).all()
            # snippet 포함이 정답 판정의 최종 검증자 (§4.1). heading_path는
            # 완전일치를 요구하지 않고, 여러 청크가 snippet을 포함할 때 우선순위로만 사용.
            # 매칭은 공백 무시 — PDF/DOCX 추출 텍스트는 줄바꿈·공백 위치가 원본과 다르다 (v2).
            snip = re.sub(r"[\s|]+", "", ec["snippet"])     # 검증기와 동일 정규화 (공백+파이프)
            hits = [(cid, hp) for cid, hp, text in chunk_rows
                    if snip in re.sub(r"[\s|]+", "", text)]
            want_hp = ec.get("heading_path")     # v2 스키마엔 없음 (PDF/DOCX 헤딩 미보존)
            match = next((cid for cid, hp in hits if want_hp and hp == want_hp),
                         hits[0][0] if hits else None)
            if match is None:
                r.stale.append(f'{row["id"]}: chunk 못찾음 {ec["snippet"][:20]}')
            else:
                cids.append(match)
        r.chunk_ids[row["id"]] = cids
    return r


# 한 문항 검색 채점
def score_one(got_ids: list[int], gold_ids: list[int], ks=(5, 20)) -> dict | None:
    """검색 결과 순위(got)와 정답 id(gold)로 Recall@k·Hit@1·MRR 계산.

    got_ids: 검색기가 가져온 id, RRF 순 (1등이 [0]).
    gold_ids: resolve된 정답 id. 비어 있으면 채점 불가 → None (집계 제외).
    """
    if not gold_ids:
        return None
    gold = set(gold_ids)

    # Recall@k: 정답이 상위 k개 안에 하나라도 있으면 1, 없으면 0
    scores = {f"recall_at_{k}": float(bool(gold & set(got_ids[:k]))) for k in ks}

    # Hit@1: 1등(got_ids[0])이 정답인가
    scores["hit_at_1"] = float(bool(got_ids and got_ids[0] in gold))

    # MRR: 정답이 처음 등장하는 순위의 역수 (1등=1.0, 3등=0.33, 못 찾으면 0)
    rank = next((i for i, cid in enumerate(got_ids, 1) if cid in gold), None)
    scores["mrr"] = 1.0 / rank if rank else 0.0

    return scores



# dense-only 진단 — 리랭커·게이트 제외, dense 순위만 (임베딩 순수 비교)
async def retrieve_dense_only(session, tenant_id, query, backend=None, top_n=20) -> list[int]:
    """dense cosine 순위만 반환 (chunk_id 리스트).
    backend 인자는 과거 KURE 비교(Q2 — 동률 종결)용 잔재 — 호출부 호환으로만 유지, 무시됨."""
    q = (await embed_query(query)).dense
    dist = Chunk.dense.cosine_distance(q).label("d")
    stmt = (
        select(Chunk.id.label("cid"), dist)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.tenant_id == tenant_id)
        .where(Document.is_active.is_(True))
        .where(Document.status == "ready")
        .order_by(dist)
        .limit(top_n)
    )
    return [r.cid for r in (await session.execute(stmt)).all()]


async def run_dense_only(session, backend, gold_rows, resolved):
    """dense-only 로 검색 subset 점수 산출 (게이트 없음)."""
    rows = []
    for row in gold_rows:
        if row["type"] not in RETRIEVAL_TYPES:
            continue
        got = await retrieve_dense_only(session, TENANT, row["query"], backend, top_n=20)
        gold_ids = resolved.chunk_ids.get(row["id"]) or []
        if gold_ids:
            score = score_one(got, gold_ids)
        else:
            # dense-only 는 chunk 단위만 — doc 폴백은 chunk_id→doc 매핑이 없어 생략
            continue
        if score is None:
            continue
        rows.append({"id": row["id"], "type": row["type"], "scores": score})
    return rows


# 게이트 임계값별 오거부 판정
def gate_sweep(cands) -> dict[float, str]:
    """임계값별 게이트 판정. cands 재사용 → DB 재조회 없음 (B0 분리 덕).

    반환: {threshold: 'reject'|'pass'}.
    근거 있는 문항(has_evidence=True)에서 'reject'면 오거부 — 집계에서 카운트.
    """
    out = {}
    for th in SWEEP:
        no_ev, _ = apply_gate(cands, max_dense_distance=th)
        out[th] = "reject" if no_ev else "pass"
    return out


# 평가 config: 이름 -> (후보검색 함수, 리랭커 모델 or None)
CONFIG_REGISTRY = {
    "bge":           (retrieve_candidates,      None),                          # 운영 기본
    "bge+rerank":    (retrieve_candidates,      "BAAI/bge-reranker-v2-m3"),     # 1.5.C 1라운드
    "bge+rerank-ko": (retrieve_candidates,      "dragonkue/bge-reranker-v2-m3-ko"),  # 2라운드
}
# 이번 실행에서 돌릴 config (KURE/리랭커 모델 재로딩 줄이려 필요한 것만)
ACTIVE = ["bge"]


async def run_config(session, name, gold_rows, resolved, fingerprint):
    """한 config(검색+선택적 리랭크)로 subset 평가 → (rows, 평균 latency ms)."""
    candidates_fn, rerank_model = CONFIG_REGISTRY[name]
    rows = []
    elapsed = []
    for row in gold_rows:
        if row["type"] not in RETRIEVAL_TYPES:              # 검색 지표 subset만
            continue
        t0 = time.perf_counter()
        # 이중 리랭크 방지: retrieve_candidates 내부 리랭크(settings.rerank_enabled)를 끄고
        # config의 rerank_model만 유일한 리랭크 제어점으로 둔다. bge=리랭크0, bge+rerank=리랭크1.
        prev_rerank = settings.rerank_enabled
        settings.rerank_enabled = False
        try:
            cands = await candidates_fn(session, TENANT, row["query"], top_n=20)
        finally:
            settings.rerank_enabled = prev_rerank
        if rerank_model:                                    # 후보 순서만 재정렬 (R@20 불변)
            from rag.reranker import rerank                  # 사용 시점에만 import (모듈은 직접 구현 예정)
            cands.chunks = await rerank(row["query"], cands.chunks, rerank_model)
        elapsed.append((time.perf_counter() - t0) * 1000)
        got = [c.chunk_id for c in cands.chunks]

        # chunk 정답 우선, 없으면 doc 단위 폴백 (§5.2)
        gold_ids = resolved.chunk_ids.get(row["id"]) or []
        if gold_ids:
            score = score_one(got, gold_ids)
        else:
            got_docs = [c.document_id for c in cands.chunks]
            doc_gold = [resolved.doc_ids[fn] for fn in row["expected_docs"]
                        if fn in resolved.doc_ids]
            score = score_one(got_docs, doc_gold)

        if score is None:                                   # resolve 실패 → 제외
            continue

        rows.append({
            "id": row["id"],
            "type": row["type"],
            "scores": score,
            "gate": gate_sweep(cands),                      # 게이트는 top_dense_distance 기준 (리랭크 무관)
            "has_evidence": row["has_evidence"],
            "gold_version": "v1",
            "corpus_fingerprint": fingerprint,
        })
    mean_ms = sum(elapsed) / len(elapsed) if elapsed else 0.0
    return rows, mean_ms


# ----- 메인: 백엔드별 평가 실행 + 결과 저장 + 비교 -------------------
async def main():
    gold_rows = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    fingerprint = [d["sha256"] for d in json.loads(FINGERPRINT.read_text())["documents"]]
    RESULT_DIR.mkdir(exist_ok=True)

    async with AsyncSessionLocal() as session:
        resolved = await resolve_gold(session, TENANT, gold_rows)
        if resolved.stale:                                  # 라벨 노후 먼저 경보
            print(f"⚠ 라벨 노후 {len(resolved.stale)}건 (집계 제외):")
            for s in resolved.stale:
                print("  -", s)

        all_rows = {}
        latency = {}
        for name in ACTIVE:
            print(f"\n=== config: {name} ===")
            rows, mean_ms = await run_config(session, name, gold_rows, resolved, fingerprint)
            out = RESULT_DIR / f"retrieval_{name}.jsonl"
            out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
            summarize(rows)
            print(f"→ {out} ({len(rows)} rows), 평균 검색 latency {mean_ms:.0f} ms")
            all_rows[name] = rows
            latency[name] = mean_ms

    print("\n[config 비교 — Recall@5 / Hit@1]")
    compare(all_rows)
    print("\n[평균 검색 latency (ms/query)]")
    for name in ACTIVE:
        print(f"  {name:<16} {latency[name]:.0f}")


def compare(all_rows):
    """백엔드별 type 평균 Recall@5/Hit@1 나란히 비교."""
    backends = list(all_rows)
    header = f"{'type':<14}" + "".join(f"{b:>18}" for b in backends)
    print(header)
    types = ["single_fact", "paraphrase", "rare_lexical", "multi_doc"]
    for t in types:
        cells = []
        for b in backends:
            ss = [r["scores"] for r in all_rows[b] if r["type"] == t]
            if ss:
                r5 = sum(s["recall_at_5"] for s in ss) / len(ss)
                h1 = sum(s["hit_at_1"] for s in ss) / len(ss)
                cells.append(f"{r5:>8.2f}/{h1:<8.2f}")
            else:
                cells.append(f"{'-':>17}")
        print(f"{t:<14}" + "".join(f"{c:>18}" for c in cells))


def summarize(rows):
    """type별 평균 지표 + threshold별 오거부 건수 출력."""
    by_type = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r["scores"])

    print("\n[type별 검색 지표]")
    print(f"{'type':<14}{'n':>4}{'R@5':>7}{'R@20':>7}{'Hit@1':>7}{'MRR':>7}")
    for t, ss in by_type.items():
        n = len(ss)
        avg = lambda key: sum(s[key] for s in ss) / n
        print(f"{t:<14}{n:>4}{avg('recall_at_5'):>7.2f}{avg('recall_at_20'):>7.2f}"
              f"{avg('hit_at_1'):>7.2f}{avg('mrr'):>7.2f}")

    print("\n[threshold별 오거부 (근거 있는데 거부)]")
    ev = [r for r in rows if r["has_evidence"]]
    for th in SWEEP:
        cnt = sum(1 for r in ev if r["gate"][th] == "reject")
        print(f"  th={th}: {cnt}/{len(ev)}")


if __name__ == "__main__":
    asyncio.run(main())
