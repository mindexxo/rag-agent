"""검색 결과 순서 골든 — 리팩터링 전/후를 바이트로 대조한다 (#38에서 만듦).

검색은 **순서가 곧 결과**라 프롬프트처럼 해시로 동일성을 증명할 수 없다. 그렇다고
eval을 매번 돌릴 수도 없다(TEI 필요, 수십 분). 그 사이를 메우는 도구다 —
결정적 가짜 임베딩·가짜 리랭커로 **실제 DB**를 태워 chunk_id 순서만 뽑는다.

TEI 없이 돌아가고 30초면 끝난다. 검색 로직을 건드리는 축마다 재사용할 것.
청킹을 건드릴 때는 `_golden_chunks.py`(청크 자체를 대조 — 더 직접적)를 쓴다.

사용:
    python -m eval._golden_order > before.json     # 변경 전
    python -m eval._golden_order > after.json      # 변경 후
    diff before.json after.json                    # 무변경이면 출력 없음
"""
import asyncio
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import settings


def _fake_dense(text: str) -> list[float]:
    """tests/conftest.py의 fake_vector와 **완전히 동일한 구현**.
    직접 만든 변형은 NaN이 섞여 pgvector가 거부했다 — 검증된 것을 그대로 쓴다."""
    out: list[float] = []
    block = 0
    while len(out) < 1024:
        seed = hashlib.sha256(f'{block}:{text}'.encode()).digest()
        out.extend((b / 255.0) - 0.5 for b in seed)
        block += 1
    return out[:1024]


QUERIES = [
    '교환은 몇 번까지 가능해?',
    '반품 기간',
    '배송비 얼마야',
    '케어라벨 표기대로 세탁했는데 옷이 줄었어요',
    '적립금 사용 조건',
    '회원 등급별 혜택',
    'RF-01',
    '아무 질의',
]
# 멀티쿼리(#5) 경로도 덮는다 — 단일 경로만 보면 분기 절반이 미검증
EXPANDED = {
    '교환은 몇 번까지 가능해?': ['교환 횟수 제한', '재교환 가능 여부'],
    '반품 기간': ['반품 신청 기한', '환불 접수 기간'],
}


async def main():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag import embeddings as emb_mod
    from rag import retriever as rt
    from rag.models import Document

    # 임베딩을 결정적 가짜로 교체 (conftest와 동일 목적, TEI 불필요)
    async def _fake_embed_texts(texts):
        return [emb_mod.Embedding(dense=_fake_dense(t)) for t in texts]

    rt.embed_texts = _fake_embed_texts

    # 리랭커도 결정적 가짜로 — TEI 없이 리랭크 경로(단일 rerank / 멀티 max-pool)까지 덮는다.
    # 리랭크를 꺼버리면 이번 리팩터링이 가장 많이 건드리는 분기가 골든에서 빠진다.
    # rag.reranker의 이름을 패치하므로, max-pool이 reranker.py로 이관돼도 그대로 먹는다.
    import rag.reranker as rr

    async def _fake_scores(query: str, chunks: list):
        # query·청크쌍마다 결정적 점수. 순위가 dense 순서와 확실히 달라지게 섞는다.
        return [
            int.from_bytes(
                hashlib.sha256(f'{query}|{c.chunk_id}'.encode()).digest()[:4], 'big'
            ) / 2**32
            for c in chunks
        ]

    rr.rerank_scores = _fake_scores

    engine = create_async_engine(settings.database_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    out = {}
    async with Session() as session:
        tenants = (await session.execute(
            select(Document.tenant_id).distinct().order_by(Document.tenant_id)
        )).scalars().all()
        for tenant in tenants:
            for q in QUERIES:
                for label, exp in (('single', None), ('multi', EXPANDED.get(q))):
                    if label == 'multi' and not exp:
                        continue
                    for rerank_on in (False, True):
                        settings.rerank_enabled = rerank_on   # 호출 시점마다 읽히는지도 함께 검증
                        cands = await rt.retrieve_candidates(
                            session, tenant, q, top_n=20, expanded_queries=exp
                        )
                        key = f'{tenant}|{q}|{label}|rerank={rerank_on}'
                        out[key] = {
                            'ids': [c.chunk_id for c in cands.chunks],
                            'dist': round(cands.top_dense_distance, 9),
                        }
    await engine.dispose()
    print(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True))


asyncio.run(main())
