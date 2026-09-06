"""어휘 채널(#135) 백필 — 기존 청크의 lex_tsv·lex_len을 채운다.

reindex_documents와 달리 **청킹·임베딩을 다시 하지 않는다**(TEI 불필요) — 청크 본문은
그대로 두고 어휘 컬럼만 UPDATE한다. 토큰화 입력은 인제스션과 동일 조립(문서=
'파일명>헤딩' 프리픽스, FAQ=원문 — 정의점 rag/lexical.py·rag/documents.py 참조).

기본은 lex_tsv IS NULL인 청크만(증분). --recompute는 전 청크 재계산 — 토크나이저를
바꿨을 때(#133: 교체=재색인 1회가 이것) 쓴다. 멱등: 몇 번 돌려도 같은 결과.

실행: python -m eval.backfill_lexical --tenant <id> | --all [--recompute] [--dry-run]
"""
import argparse
import asyncio

from sqlalchemy import ARRAY, Text, cast, func, select, update

from database import AsyncSessionLocal
from rag import lexical
from rag.index_text import build_index_text
from rag.models import Chunk, Document

BATCH = 200   # 커밋 단위 — 실패 시 그 배치만 롤백, 재실행이 이어받는다(멱등)


async def backfill(tenant: str | None, recompute: bool, dry_run: bool) -> None:
    async with AsyncSessionLocal() as session:
        stmt = (
            select(Chunk.id, Chunk.text, Chunk.heading_path, Document.filename)
            .outerjoin(Document, Chunk.document_id == Document.id)
        )
        if tenant:
            stmt = stmt.where(Chunk.tenant_id == tenant)
        if not recompute:
            stmt = stmt.where(Chunk.lex_tsv.is_(None))
        rows = (await session.execute(stmt)).all()
        print(f"대상 {len(rows)}청크 (tenant={tenant or '전체'}, "
              f"{'전량 재계산' if recompute else 'NULL만'})")
        if dry_run:
            for r in rows[:5]:
                text = r.text if r.filename is None else build_index_text(
                    r.text, r.filename, list(r.heading_path or []))
                toks = lexical.bigrams(text)
                print(f"  chunk {r.id}: lex_len={len(toks)} 토큰 예시 {toks[:8]}")
            print("dry-run — 변경 없음")
            return

        done = 0
        for r in rows:
            text = r.text if r.filename is None else build_index_text(
                r.text, r.filename, list(r.heading_path or []))
            toks = lexical.bigrams(text)
            await session.execute(
                update(Chunk).where(Chunk.id == r.id).values(
                    lex_tsv=func.array_to_tsvector(
                        cast(lexical.tsvector_lexemes(toks), ARRAY(Text))),
                    lex_len=len(toks),
                ))
            done += 1
            if done % BATCH == 0:
                await session.commit()
                print(f"  …{done}/{len(rows)}", flush=True)
        await session.commit()
        print(f"완료 {done}청크")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--tenant", help="특정 테넌트만")
    g.add_argument("--all", action="store_true", help="전 테넌트")
    p.add_argument("--recompute", action="store_true",
                   help="lex_tsv가 이미 있어도 재계산 (토크나이저 교체 시)")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    asyncio.run(backfill(a.tenant, a.recompute, a.dry_run))


if __name__ == "__main__":
    main()
