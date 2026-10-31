"""MD 업로드 → 색인 왕복(#267) — MD가 docling 변환 프로세스를 타고 표 블록·결정적 실패 규칙까지 PDF·DOCX와 같은 길을 가는지.

단위(tests/test_chunking_md.py)는 chunk_file까지만 본다. 여기서는 업로드 → outbox → 변환 프로세스 → 엔진 색인 → 문서 상태를 운영과
같은 경로로 확인한다(DB·OpenSearch 필요, 임베딩은 fake).
"""

import pytest

from database import AsyncSessionLocal
from rag.models import Document
from tests.conftest import indexed_chunk_texts, ingest

pytest.importorskip('docling')

HEADER = '| 코드 | 품목 | 비고 |'


async def _upload(client, filename: str, text: str) -> int:
    res = await client.post('/kms/documents', files={'file': (filename, text.encode(), 'text/markdown')})
    assert res.status_code == 200, res.text
    return res.json()['document_id']


@pytest.mark.asyncio
async def test_큰_MD_표는_묶음마다_열_이름을_달고_칸_내용이_그대로_색인된다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed
):
    rows = '\n'.join(f'| MD-{i:03d} | 협업\\|모델 {i} | 비고 {i}<br>추가 |' for i in range(120))
    doc_id = await _upload(client, '표문서.md', f'# 표 문서\n\n## 1. 목록\n\n{HEADER}\n| --- | --- | --- |\n{rows}\n')
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    texts = await indexed_chunk_texts(doc_id)
    batches = [t for t in texts if '| MD-' in t]
    assert len(batches) >= 2 and all(HEADER in t for t in batches)
    assert '| MD-077 | 협업\\|모델 77 | 비고 77<br>추가 |' in '\n'.join(texts)
    async with AsyncSessionLocal() as s:
        assert (await s.get(Document, doc_id)).status == 'ready'


@pytest.mark.asyncio
async def test_상한을_넘는_MD_표는_재시도_없이_failed_사유를_남긴다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed
):
    # #257의 TABLE_MAX_TOKENS가 MD에도 걸린다 — 정규식 시절에는 글로 색인됐다(#267 알려진 동작 변화)
    rows = '\n'.join(f'| BIG-{i:04d} | {"긴 설명 " * 12} | 값 {i} |' for i in range(400))
    doc_id = await _upload(client, '큰표.md', f'# 큰 표\n\n{HEADER}\n| --- | --- | --- |\n{rows}\n')
    assert await ingest(doc_id) == {'done': 0, 'failed': 1}
    async with AsyncSessionLocal() as s:
        doc = await s.get(Document, doc_id)
    assert doc.status == 'failed' and '허용 크기' in doc.status_reason
