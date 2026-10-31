"""검색 결과 순서 골든 — 리팩토링 전/후의 후보 순서(chunk_id)를 대조한다.

임베딩·리랭커를 결정적 가짜로 바꾸고 **개발계 DB·OpenSearch 색인**을 그대로 태운다(TEI 불필요, 수십 초).
질의 벡터가 가짜라 순위의 의미는 없고, 같은 색인에서 같은 순서가 나오는지만 본다.
dense·어휘·식별자 채널과 멀티쿼리·리랭크 on/off를 함께 덮는다. 청킹은 `_golden_chunks.py`로 본다.

색인은 공유 자원이다 — 전후 실행 사이에 누가 업로드하면 순서가 달라진다. diff가 나오면 main에서 한 번 더
떠서 색인 변화인지 먼저 가른다.
검색 채널 설정(하이브리드·식별자 채널·가산)은 `.env`에서 읽으므로 전후를 같은 환경에서 뜬다.

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
    """tests/conftest.py의 fake_vector와 같은 구현."""
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
    '제3조 내용 알려줘',
    '아무 질의',
]
# 멀티쿼리 경로도 덮는다 — 단일 경로만 보면 분기 절반이 미검증
EXPANDED = {
    '교환은 몇 번까지 가능해?': ['교환 횟수 제한', '재교환 가능 여부'],
    '반품 기간': ['반품 신청 기한', '환불 접수 기간'],
}


async def main():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from rag.models import Document
    from rag.search import embeddings as emb_mod
    from rag.search import retriever as rt

    # 임베딩을 결정적 가짜로 교체 (conftest와 동일 목적, TEI 불필요)
    async def _fake_embed_texts(texts, kind="document"):
        return [emb_mod.Embedding(dense=_fake_dense(t)) for t in texts]

    rt.embed_texts = _fake_embed_texts

    # 리랭커도 결정적 가짜로 — TEI 없이 리랭크 경로(단일 rerank / 멀티 max-pool)까지 덮는다.
    # 리랭크를 꺼버리면 이번 리팩터링이 가장 많이 건드리는 분기가 골든에서 빠진다.
    # rag.search.reranker의 이름을 패치하므로, max-pool이 reranker.py로 이관돼도 그대로 먹는다.
    import rag.search.reranker as rr

    async def _fake_scores(query: str, chunks: list):
        # query·청크쌍마다 결정적 점수. 순위가 dense 순서와 확실히 달라지게 섞는다.
        return [
            int.from_bytes(hashlib.sha256(f'{query}|{c.chunk_id}'.encode()).digest()[:4], 'big') / 2**32 for c in chunks
        ]

    rr.rerank_scores = _fake_scores

    engine = create_async_engine(settings.database_url)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    out = {}
    async with Session() as session:
        tenants = (
            (await session.execute(select(Document.tenant_id).distinct().order_by(Document.tenant_id))).scalars().all()
        )
        for tenant in tenants:
            for q in QUERIES:
                for label, exp in (('single', None), ('multi', EXPANDED.get(q))):
                    if label == 'multi' and not exp:
                        continue
                    for rerank_on in (False, True):
                        settings.rerank_enabled = rerank_on  # 호출 시점마다 읽히는지도 함께 검증
                        cands = await rt.retrieve_candidates(session, tenant, q, top_n=20, expanded_queries=exp)
                        key = f'{tenant}|{q}|{label}|rerank={rerank_on}'
                        out[key] = {
                            'ids': [c.chunk_id for c in cands.chunks],
                            'ident': [c.matched_ident for c in cands.chunks],
                            'dist': round(cands.top_dense_distance, 9),
                        }
    await engine.dispose()
    print(json.dumps(out, ensure_ascii=False, indent=1, sort_keys=True))


asyncio.run(main())
