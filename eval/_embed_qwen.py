"""chunks_dense_qwen 생성 + 전 청크를 Qwen3-Embedding-4B(1024차원) 으로 적재 (일회용).

임베딩 모델 A/B용. 운영 `chunks.dense`는 안 건드린다 — 별 테이블에 보관해두고,
측정할 쪽만 `chunks.dense`로 스왑한다(하네스·retriever 코드 변경 0):

    -- 1) 현 BGE 벡터 보관 (최초 1회)
    CREATE TABLE chunks_dense_bge AS SELECT id, dense FROM chunks;
    -- 2) 이 스크립트로 Qwen 벡터 적재
    -- 3) 측정 대상 스왑
    UPDATE chunks c SET dense = q.dense FROM chunks_dense_qwen q WHERE c.id = q.id;  -- Qwen 측정
    UPDATE chunks c SET dense = b.dense FROM chunks_dense_bge  b WHERE c.id = b.id;  -- 원복

**입력 텍스트를 인제스션과 동일하게 조립하는 것이 이 비교의 전제다.**
BGE 벡터는 `build_index_text(text, filename, heading_path)`로 만들어졌고(검증: 저장 벡터와
코사인 1.000000), FAQ 청크는 프리픽스 없이 본문만 쓴다(rag/reranker._rerank_text와 같은 규칙).
folder는 리랭커 전용이라 임베딩엔 넣지 않는다(rag/index_text 참조).

실행 (사내망 필요):
    EMBED_BASE_URL=http://10.1.32.15:38891 EMBED_DIMENSIONS=1024 python3 -m eval._embed_qwen
"""
import asyncio
import sys

from sqlalchemy import select, text

from config import settings
from database import AsyncSessionLocal
from rag.embeddings import embed_texts
from rag.index_text import build_index_text
from rag.models import Chunk, Document

# 차원별로 테이블을 따로 둔다 (1024 = MRL 절단, 2000 = vector+HNSW 상한).
TABLE = sys.argv[1] if len(sys.argv) > 1 else "chunks_dense_qwen"

DDL = """
CREATE TABLE IF NOT EXISTS {tbl} (
    id    BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    dense VECTOR({dim})
);
"""


async def main() -> None:
    if not settings.embed_dimensions:
        raise SystemExit("EMBED_DIMENSIONS=1024 로 실행하세요 (Qwen3 원 차원 2560 → 컬럼과 불일치)")
    print(f"임베딩 서버: {settings.embed_base_url}  차원: {settings.embed_dimensions}")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            for stmt in DDL.format(tbl=TABLE, dim=settings.embed_dimensions).strip().split(";"):
                if stmt.strip():
                    await session.execute(text(stmt))

    # 전 테넌트·전 청크. FAQ 청크는 document가 없으므로 outer join (retriever와 같은 다형성).
    # 검색 가능 여부(is_active/ready)로 걸러내지 않는다 — 상태가 바뀌어도 양쪽 테이블 모수가
    # 어긋나지 않게 하려는 것이고, 실제 필터는 조회 시점에 retriever가 건다.
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(Chunk.id, Chunk.text, Chunk.heading_path, Chunk.faq_id, Document.filename)
            .outerjoin(Document, Chunk.document_id == Document.id)
            .order_by(Chunk.id)
        )).all()

    ids, inputs = [], []
    for cid, ctext, heading, faq_id, filename in rows:
        ids.append(cid)
        inputs.append(ctext if faq_id else build_index_text(ctext, filename, heading))
    prefixed = sum(1 for i, r in zip(inputs, rows) if i != r.text)
    print(f"청크 {len(ids)}개 인코딩... (프리픽스 적용 {prefixed}개 / FAQ·미적용 {len(ids) - prefixed}개)")

    vecs = [e.dense for e in await embed_texts(inputs)]
    bad = [len(v) for v in vecs if len(v) != settings.embed_dimensions]
    if bad:
        raise SystemExit(f"차원 불일치 {len(bad)}건 (예: {bad[0]}) — 서버가 dimensions를 무시했습니다")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await session.execute(text(f"TRUNCATE {TABLE}"))   # 재실행 대비
            await session.execute(
                text(f"INSERT INTO {TABLE} (id, dense) VALUES (:id, :dense)"),
                [{"id": cid, "dense": str(v)} for cid, v in zip(ids, vecs)],
            )
    print(f"적재 완료: {len(ids)}행 -> {TABLE}")


if __name__ == "__main__":
    asyncio.run(main())
