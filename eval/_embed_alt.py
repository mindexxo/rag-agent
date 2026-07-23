"""eval_kure_dense 테이블 생성 + demo 코퍼스를 KURE-v1 dense 로 적재 (일회용).

운영 chunks 는 안 건드린다. KURE 벡터는 eval_kure_dense 에만 저장.
재실행 시 기존 행을 비우고 다시 채운다 (청크 id 가 재인제스천으로 바뀔 수 있으므로).

실행: python -m eval._embed_alt
"""
import asyncio

from sqlalchemy import delete, select, text

from database import AsyncSessionLocal
from rag.models import Chunk, Document
from eval._kure import EvalKureDense, embed_dense_kure

TENANT = "demo"

DDL = """
CREATE TABLE IF NOT EXISTS eval_kure_dense (
    chunk_id BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dense    VECTOR(1024)
);
CREATE INDEX IF NOT EXISTS idx_eval_kure_dense_hnsw
    ON eval_kure_dense USING hnsw (dense vector_cosine_ops);
"""


async def main():
    # 1) 테이블/인덱스 생성 (자체 트랜잭션)
    async with AsyncSessionLocal() as session:
        async with session.begin():
            for stmt in DDL.strip().split(";"):
                if stmt.strip():
                    await session.execute(text(stmt))

    # 2) 활성/ready 청크의 id + 본문 읽기 (별도 세션)
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(Chunk.id, Chunk.text)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.tenant_id == TENANT)
            .where(Document.is_active.is_(True))
            .where(Document.status == "ready")
        )).all()
    ids = [r.id for r in rows]
    texts = [r.text for r in rows]
    print(f"청크 {len(ids)}개 KURE 인코딩...")

    # 3) 인코딩 (DB 무관)
    vecs = embed_dense_kure(texts)

    # 4) 적재 (별도 트랜잭션)
    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(delete(EvalKureDense))   # 재실행 대비 초기화
            for cid, v in zip(ids, vecs):
                session.add(EvalKureDense(chunk_id=cid, dense=v))
    print(f"적재 완료: {len(ids)}행 -> eval_kure_dense")


if __name__ == "__main__":
    asyncio.run(main())
