"""시맨틱 캐시 히트 정확성 평가 — 캐시가 '같은 질문은 재사용, 다른 질문은 안 섞는지'.

캐시 히트 조건은 이중 가드: 임베딩 유사도 ≥ threshold(0.95) AND 검색 doc집합 동일.
두 실패 방향:
- 오탐(false hit): 임베딩은 비슷하나 의미가 다른 질문("환불 되는"vs"안되는")이 옛 답 재생 → 오답.
- 과잉거절(false miss): 같은 질문의 다른 표현(paraphrase)인데 캐시 미스 → 캐시 무용.

방식: doc_ids는 실제 코퍼스 테넌트(summers)에서 검색해 뽑되, 캐시 저장/조회는 버려도 되는
가짜 테넌트(EVAL_TENANT)에 격리 → summers 실캐시 오염 없음. 쌍마다 set(q1)→조회(q2)→정리.
프롬프트 무관·LLM 생성 없음(임베딩+DB만). 임계값·doc집합 가드 변경 시 회귀 감지.

실행: python -m eval.cache_eval
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete

from database import AsyncSessionLocal
from rag.cache import AnswerCache
from rag.models import AnswerCache as AnswerCacheRow
from rag.retriever import retrieve
from rag.service import _source_doc_ids

GOLD = Path(__file__).resolve().parent / "cache_set_v1.jsonl"
EVAL_TENANT = "__cache_eval__"     # 캐시 저장 격리용 (summers 실캐시 불건드림)


async def _docids(session, tenant: str, query: str) -> list[int]:
    r = await retrieve(session, tenant, query)
    return _source_doc_ids(r.chunks)


async def _clear(session) -> None:
    await session.execute(delete(AnswerCacheRow).where(AnswerCacheRow.tenant_id == EVAL_TENANT))
    await session.commit()


async def compute() -> dict:
    """캐시 히트 정확성 채점 → 요약. 반환: {accuracy, n, false_hit, false_miss, by_kind, misses}."""
    pairs = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    cache = AnswerCache()
    rows = []

    async with AsyncSessionLocal() as session:
        await _clear(session)
        for p in pairs:
            # 1. q1의 doc집합으로 캐시에 심는다 (답변은 더미 — 히트 여부만 관심)
            ids1 = await _docids(session, p["tenant"], p["q1"])
            await cache.set(session, EVAL_TENANT, p["q1"], "더미 답변", [], ids1)
            await session.commit()
            # 2. q2로 조회 — 히트하나?
            ids2 = await _docids(session, p["tenant"], p["q2"])
            hit = await cache.get_semantic(session, EVAL_TENANT, p["q2"], ids2) is not None
            await _clear(session)     # 다음 쌍 오염 방지
            rows.append({**p, "hit": hit, "ok": hit == p["should_hit"]})

    by_kind = defaultdict(lambda: [0, 0])
    for r in rows:
        by_kind[r["kind"]][0] += r["ok"]
        by_kind[r["kind"]][1] += 1

    false_hit = [r for r in rows if not r["should_hit"] and r["hit"]]     # 다른데 재사용
    false_miss = [r for r in rows if r["should_hit"] and not r["hit"]]    # 같은데 미스
    neg_n = sum(1 for r in rows if not r["should_hit"])
    pos_n = sum(1 for r in rows if r["should_hit"])

    return {
        "accuracy": sum(r["ok"] for r in rows) / len(rows) if rows else 0.0,
        "n": len(rows),
        "false_hit": len(false_hit), "false_hit_n": neg_n,
        "false_miss": len(false_miss), "false_miss_n": pos_n,
        "by_kind": {k: v for k, v in by_kind.items()},
        "misses": [r for r in rows if not r["ok"]],
    }


async def main() -> None:
    r = await compute()
    print(f"[시맨틱 캐시 히트 정확성]  {r['n']}쌍 (negative=재사용안함정답 / positive=재사용정답)\n")
    print(f"{'kind':<12}{'정확도':>14}")
    for k, (ok, tot) in sorted(r["by_kind"].items()):
        print(f"{k:<12}{f'{ok}/{tot} ({ok / tot:.0%})':>14}")
    print(f"\n전체 정확도: {r['accuracy']:.0%}")
    print(f"  오탐(다른질문 재사용)  : {r['false_hit']}/{r['false_hit_n']}  ← 오답 재생 위험")
    print(f"  과잉거절(같은질문 미스): {r['false_miss']}/{r['false_miss_n']}  ← 캐시 효용 손실")

    if r["misses"]:
        print("\n[오판정]")
        for m in r["misses"]:
            verdict = "재사용함" if m["hit"] else "미스"
            print(f"  [{m['kind']}] {m['q1']!r} ↔ {m['q2']!r} → {verdict}  ({m['note']})")


if __name__ == "__main__":
    asyncio.run(main())
