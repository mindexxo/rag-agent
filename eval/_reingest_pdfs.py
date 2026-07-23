"""PDF 7종 재인제스천 (일회용, do_cell_matching=False 적용 후 표 텍스트 갱신).

기존 PDF 문서 행 삭제(→ chunks cascade) 후 docs/generated/ 에서 재적재.
파일 내용(sha)은 불변이라 fingerprint·gold 안정 키는 그대로 유효.

실행: python -m eval._reingest_pdfs
"""
import asyncio
from pathlib import Path

from sqlalchemy import delete, select

from database import AsyncSessionLocal
from rag.ingestion import ingest_file
from rag.models import Document

TENANT = "demo"
PDFS = [
    "kms_01_비밀번호재설정.pdf",
    "kms_02_환불정책.pdf",
    "kms_04_교환반품정책.pdf",
    "kms_05_주문취소.pdf",
    "kms_07_멤버십혜택.pdf",
    "kms_09_포인트적립금.pdf",
    "kms_11_공지_202606.pdf",
]
SRC = Path("docs/generated")


async def main():
    # 1) 기존 PDF 문서 삭제 (chunks 는 FK cascade)
    async with AsyncSessionLocal() as session:
        async with session.begin():
            ids = (await session.execute(
                select(Document.id)
                .where(Document.tenant_id == TENANT)
                .where(Document.filename.in_(PDFS))
            )).scalars().all()
            if ids:
                await session.execute(delete(Document).where(Document.id.in_(ids)))
        print(f"삭제: {len(ids)} 문서")

    # 2) 재인제스천
    for fn in PDFS:
        doc_id = await ingest_file(SRC / fn, TENANT)
        print(f"재적재: {fn} -> doc {doc_id}")


if __name__ == "__main__":
    asyncio.run(main())
