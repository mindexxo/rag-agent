"""실행 단계들이 함께 쓰는 결과 경로와 DB·색인 조회."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.models import Document

ROOT = Path('eval/results/doc_fidelity')


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows))


async def ready_documents(tenant: str) -> Sequence[Document]:
    from sqlalchemy import select

    from database import AsyncSessionLocal
    from rag.models import Document

    async with AsyncSessionLocal() as s:
        return (
            (
                await s.execute(
                    select(Document)
                    .where(Document.tenant_id == tenant, Document.status == 'ready', Document.is_active.is_(True))
                    .order_by(Document.id)
                )
            )
            .scalars()
            .all()
        )


async def index_chunks(document_id: int) -> list[dict]:
    from config import settings
    from rag.search import os_client

    resp = await os_client.client().search(
        index=settings.opensearch_index,
        body={
            'size': 2000,
            '_source': ['text', 'heading_path', 'page'],
            'query': {'term': {'document_id': document_id}},
            'sort': [{'chunk_id': 'asc'}],
        },
    )
    return [h['_source'] for h in resp['hits']['hits']]
