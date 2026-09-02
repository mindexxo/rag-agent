"""시맨틱 캐시 히트 정확성 평가 — 캐시가 '같은 질문은 재사용, 다른 질문은 안 섞는지'.

캐시 히트 조건은 이중 가드: 임베딩 유사도 ≥ threshold(0.95) AND 검색 doc집합 동일.
두 실패 방향:
- 오탐(false hit): 임베딩은 비슷하나 의미가 다른 질문("환불 되는"vs"안되는")이 옛 답 재생 → 오답.
- 과잉거절(false miss): 같은 질문의 다른 표현(paraphrase)인데 캐시 미스 → 캐시 효용 손실.

셋 구성(#113 확장, 40쌍): negative 22 = negation(6)·numeric(4)·condition(4)·temporal(4)·
exception(4) / positive 18 = para_surface(9)·para_deep(9). kind는 문형 분류이지 현행 판정
분류가 아니다 — para_deep에는 지금도 히트하는 쌍과 임계 미달 쌍이 섞여 있고, 그 간극이
임계·가드 개선의 측정 대상이다. **positive 쌍은 전부 "q1·q2의 top-5 doc집합이 실측 동일"을
확인하고 넣었다** — doc집합이 다르면 임계를 고쳐도 히트가 불가능해 문항 오류가 되기 때문
(#113 실측: 구어체 쌍 25개 중 18개가 doc집합 상이 — 이런 쌍은 여기 없고, doc집합 가드
완화(#113 개선 4)의 근거 데이터로만 남겼다). 셋을 고치면 아래 건수 주장과 composition을
같이 갱신하라 — tests/test_docs_freshness.py가 기계 검증한다.

방식: doc_ids는 각 행의 tenant(6개 모의 테넌트, 로컬 DB에 인제스트 필수)에서 검색해 뽑되,
캐시 저장/조회는 버려도 되는 가짜 테넌트(EVAL_TENANT)에 격리 → 실테넌트 캐시 오염 없음.
쌍마다 save_answer(q1)→조회(q2)→정리. 쌍마다 지우므로 DB에 항상 row 1개 — get_semantic의
LIMIT 1이 교란 요인이 아니게 하는 전제이니 "한 번에 다 심는" 최적화를 하지 마라.
LLM 생성 없음. 의존: DB + TEI 임베딩 + TEI 리랭커(retrieve가 rerank_enabled 기본 True로 탐).

q1·q2 임베딩은 여기서 한 번씩 만들어 save_answer/get_semantic에 넘기고(#50 계약과 동일)
쌍별 유사도도 같은 벡터로 계산한다 — 오판정이 임계 탓인지 doc집합 탓인지(failed_guard)를
기록하기 위함. TEI는 호출마다 비결정적(실측 1.4e-4)이라 운영 경로의 판정과 등가다.

실행: python -m eval.cache_eval
"""
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from sqlalchemy import delete

from database import AsyncSessionLocal
from rag import cache
from rag.embeddings import embed_query
from rag.models import AnswerCache as AnswerCacheRow
from rag.retriever import retrieve
from rag.service import _source_doc_ids
from config import settings

GOLD = Path(__file__).resolve().parent / "cache_set_v1.jsonl"
EVAL_TENANT = "__cache_eval__"     # 캐시 저장 격리용 (실테넌트 캐시 불건드림)

# 셋 구성이 바뀌면 눈금이 바뀐다 — run_all이 version_key로 써서 직전 값과의 비교를 차단.
# 셋을 고칠 때마다 이 문자열을 갱신하라 (#113, 검색축 gold_composition과 같은 메커니즘).
COMPOSITION = "v1-40p"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)


async def _docids(session, tenant: str, query: str) -> list[int]:
    r = await retrieve(session, tenant, query)
    return _source_doc_ids(r.chunks)


async def _clear(session) -> None:
    await session.execute(delete(AnswerCacheRow).where(AnswerCacheRow.tenant_id == EVAL_TENANT))
    await session.commit()


def _failed_guard(similarity: float, docset_equal: bool) -> str | None:
    """미스일 때 어느 가드가 막았는가 — 오판정 분해용. 히트면 None."""
    below = similarity < settings.semantic_cache_threshold
    if below and not docset_equal:
        return "both"
    if below:
        return "threshold"
    if not docset_equal:
        return "docset"
    return None


async def compute() -> dict:
    """캐시 히트 정확성 채점 → 요약.

    반환: {accuracy, n, false_hit, false_miss, by_kind, misses, composition}.
    rows(misses 포함)에는 similarity·failed_guard가 실린다 — 오판정이 임계 탓인지
    doc집합 탓인지 목록에서 바로 읽기 위함.
    """
    pairs = [json.loads(l) for l in GOLD.read_text().splitlines() if l.strip()]
    rows = []

    async with AsyncSessionLocal() as session:
        await _clear(session)
        for p in pairs:
            # 0. 임베딩은 여기서 만들어 재사용 (#50 계약) — 유사도 기록 + TEI 호출 절감
            v1 = (await embed_query(p["q1"])).dense
            v2 = (await embed_query(p["q2"])).dense
            similarity = _cosine(v1, v2)
            # 1. q1의 doc집합으로 캐시에 심는다 (답변은 더미 — 히트 여부만 관심)
            ids1 = await _docids(session, p["tenant"], p["q1"])
            await cache.save_answer(session, EVAL_TENANT, p["q1"], "더미 답변", [], ids1,
                                    query_embedding=v1)
            await session.commit()
            # 2. q2로 조회 — 히트하나?
            ids2 = await _docids(session, p["tenant"], p["q2"])
            hit = await cache.get_semantic(session, EVAL_TENANT, p["q2"], ids2,
                                           query_embedding=v2) is not None
            await _clear(session)     # 다음 쌍 오염 방지
            rows.append({**p, "hit": hit, "ok": hit == p["should_hit"],
                         "similarity": round(similarity, 4),
                         "failed_guard": _failed_guard(similarity, set(ids1) == set(ids2))})

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
        "composition": COMPOSITION,
    }


async def main() -> None:
    r = await compute()
    print(f"[시맨틱 캐시 히트 정확성]  {r['n']}쌍 · 구성 {r['composition']} "
          f"(negative=재사용안함정답 / positive=재사용정답)\n")
    print(f"{'kind':<14}{'정확도':>14}")
    for k, (ok, tot) in sorted(r["by_kind"].items()):
        print(f"{k:<14}{f'{ok}/{tot} ({ok / tot:.0%})':>14}")
    print(f"\n전체 정확도: {r['accuracy']:.0%}")
    print(f"  오탐(다른질문 재사용)  : {r['false_hit']}/{r['false_hit_n']}  ← 오답 재생 위험")
    print(f"  과잉거절(같은질문 미스): {r['false_miss']}/{r['false_miss_n']}  ← 캐시 효용 손실")

    if r["misses"]:
        print("\n[오판정]  (막은 가드: threshold=유사도 미달 · docset=근거집합 상이)")
        for m in r["misses"]:
            verdict = "재사용함" if m["hit"] else f"미스({m['failed_guard']})"
            print(f"  [{m['kind']}] sim={m['similarity']:.4f} "
                  f"{m['q1']!r} ↔ {m['q2']!r} → {verdict}  ({m['note']})")


if __name__ == "__main__":
    asyncio.run(main())
