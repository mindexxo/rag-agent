"""FAQ 항목 인덱싱 (F3).

항목 하나를 청크 하나로 임베딩해 chunks(검색 인덱스)에 upsert한다.
문서 인제스션과 달리 파싱·청킹이 없어 워커 큐를 태우지 않는다 (임베딩 1회뿐).
임베딩은 async(TEI AsyncClient) — 호출부(라우터)가 직접 await한다.
"""
from sqlalchemy import ARRAY, Text, cast, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from rag import lexical
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
    text = build_faq_chunk_text(faq.question, faq.variants or [], faq.answer)
    # 어휘 채널(#135) — FAQ는 프리픽스 없이 원문 그대로 (임베딩 입력과 동일한 비대칭,
    # 어블레이션 코퍼스 조립도 같다). 청크와 같은 트랜잭션이라 별도 정합 관리 없음.
    toks = lexical.bigrams(text)
    session.add(Chunk(
        faq_id=faq.id,
        document_id=None,
        tenant_id=faq.tenant_id,
        chunk_index=0,
        text=text,
        heading_path=[faq.question],
        page=None,
        dense=embedding.dense,
        lex_tsv=func.array_to_tsvector(cast(lexical.tsvector_lexemes(toks), ARRAY(Text))),
        lex_len=len(toks),
    ))
