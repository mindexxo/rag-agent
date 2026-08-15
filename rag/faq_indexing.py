"""FAQ 항목 인덱싱 (F3).

항목 하나를 청크 하나로 임베딩해 chunks(검색 인덱스)에 upsert한다.
문서 인제스션과 달리 파싱·청킹이 없어 워커 큐를 태우지 않는다 (임베딩 1회뿐).
임베딩은 async(TEI AsyncClient) — 호출부(라우터)가 직접 await한다.
"""
from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from rag.models import Chunk, Faq


def build_faq_chunk_text(question: str, variants: list[str], answer: str) -> str:
    """질문+유사질문+답변을 한 청크로 — 질문 표현들이 dense 매칭을 견인한다."""
    lines = [f"Q: {question}"]
    clean = [v.strip() for v in variants if v.strip()]
    if clean:
        lines.append(f"(유사 질문: {', '.join(clean)})")
    lines.append(f"A: {answer}")
    return "\n".join(lines)


async def reindex_faq(session: AsyncSession, faq: Faq, embedding) -> None:
    """기존 청크 삭제 후 새 임베딩으로 재삽입 — 항목 단위 재인덱싱.

    add만 하고 commit은 호출자가 담당한다.
    """
    await session.execute(delete(Chunk).where(Chunk.faq_id == faq.id))
    session.add(Chunk(
        faq_id=faq.id,
        document_id=None,
        tenant_id=faq.tenant_id,
        chunk_index=0,
        text=build_faq_chunk_text(faq.question, faq.variants or [], faq.answer),
        heading_path=[faq.question],
        page=None,
        dense=embedding.dense,
    ))
