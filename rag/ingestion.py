"""문서 인제스션 CLI.

사용법:
    python -m rag.ingestion <file_path> <tenant_id>

흐름: 파일 -> 청킹 -> 임베딩 -> DB 적재.
documents 1개 row + chunks N개 row insert (한 트랜잭션 내).

Phase 1엔 dedupe/version 정책 미구현. 같은 파일 두 번 돌리면 UNIQUE 위반.
정식 정책은 Stage E.3에서 추가.
"""
import argparse
import asyncio
import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path


from database import AsyncSessionLocal
from rag.chunking import chunk_file
from rag.embeddings import embed_texts
from rag.models import Chunk, Document

_MIME_OVERRIDES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


def _detect_mime(file_path: Path) -> str:
    """파일 확장자로 mime 추론."""
    suffix = file_path.suffix.lower()
    if suffix in _MIME_OVERRIDES:
        return _MIME_OVERRIDES[suffix]
    mime, _ = mimetypes.guess_type(str(file_path))
    if not mime:
        raise ValueError(f"mime 추론 실패: {file_path}")
    return mime

def _sha256(file_path: Path) -> str:
    """파일 내용 해시. dedupe 정책(Stage E.3)에서 활용."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(64 * 1024), b""):
            h.update(block)
    return h.hexdigest()

async def ingest_file(file_path: str | Path, tenant_id: str) -> int:
    """파일 한 개를 적재. 반환: 생성된 document.id."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(file_path)

    mime = _detect_mime(path)
    sha = _sha256(path)

    chunks = chunk_file(file_path)
    embeddings = await embed_texts([c.text for c in chunks])

    async with AsyncSessionLocal() as session:
        async with session.begin():
            doc = Document(
                tenant_id=tenant_id,
                filename=path.name,
                mime=mime,
                blob_path=str(path.resolve()),
                sha256=sha,
                version=1,
                is_active=True,
                status="ready",
                char_count=sum(len(c.text) for c in chunks),
                indexed_at=datetime.now(timezone.utc).replace(tzinfo=None),
            )
            session.add(doc)
            await session.flush()

            for chunk, embedding in zip(chunks, embeddings):
                session.add(Chunk(
                    document_id=doc.id,
                    tenant_id=tenant_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    page=chunk.page,
                    heading_path=chunk.heading_path,
                    dense=embedding.dense,
                    # sparse=SparseVector(embedding.sparse, 250002),   # [dense-only, F99]
                ))
        return doc.id


def main():
    parser = argparse.ArgumentParser(description="KMS 문서 인제스션")
    parser.add_argument("file_path", help="파일 경로 (PDF/DOCX)")
    parser.add_argument("tenant_id", help="테넌트 ID")
    args = parser.parse_args()

    asyncio.run(ingest_file(args.file_path, args.tenant_id))


if __name__ == "__main__":
    main()