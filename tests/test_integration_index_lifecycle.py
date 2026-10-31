"""색인 수명 경합 — 색인이 끝나기 전에 삭제되거나, 이미 색인된 문서의 폴더 참조가 바뀌는 경우.

워커 대신 `ingest`(그 문서의 대기열 행만 drain)로 같은 경로를 태운다. 대기열 행은 처리 순서와 무관하게
같은 결과에 수렴해야 한다 — 삭제된 문서가 ready로 되살아나거나 청크가 남으면 검색에 다시 뜬다.
"""

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import SearchIndexOutbox
from rag.retriever import retrieve_candidates
from tests.conftest import indexed_chunk_texts, ingest
from tests.helpers_documents import MD, get_doc, make_folder, upload


async def _row_states(doc_id: int) -> list[tuple[str, str]]:
    async with AsyncSessionLocal() as s:
        rows = (await s.execute(select(SearchIndexOutbox).order_by(SearchIndexOutbox.id))).scalars().all()
    return [
        (r.op, r.status)
        for r in rows
        if r.payload.get('document_id') == doc_id or doc_id in (r.payload.get('document_ids') or [])
    ]


async def _candidate_doc_ids(tenant_id: str) -> set[int]:
    async with AsyncSessionLocal() as s:
        cands = await retrieve_candidates(s, tenant_id, '반품 기간', top_n=20)
    return {c.document_id for c in cands.chunks}


@pytest.mark.asyncio
async def test_색인_전에_삭제하면_색인하지_않고_삭제로_끝난다(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    """OB-21·DOC-104 — 업로드 직후(drain 전) 삭제. 색인 행은 할 일 없이 닫히고 문서는 ready가 되지 않는다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    assert (await client.delete(f'/kms/documents/{doc_id}')).status_code == 204

    assert await ingest(doc_id) == {'done': 2, 'failed': 0}

    doc = await get_doc(doc_id)
    assert doc.status == 'deleted' and doc.is_active is False
    assert await indexed_chunk_texts(doc_id) == []
    assert await _row_states(doc_id) == [('index_document', 'done'), ('drop_documents', 'done')]
    assert doc_id not in await _candidate_doc_ids(tenant_id)


@pytest.mark.asyncio
async def test_색인_도중_삭제되면_ready로_올리지_않고_들어간_청크는_다음_회차에_지운다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """OB-21·DOC-104 — 청크가 엔진에 들어간 뒤, ready 커밋 전에 삭제가 끼어든다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']

    from rag import os_index

    real = os_index.index_parsed_document

    async def _index_then_delete(**kw: object) -> None:
        await real(**kw)
        assert (await client.delete(f'/kms/documents/{doc_id}')).status_code == 204

    monkeypatch.setattr(os_index, 'index_parsed_document', _index_then_delete)  # 색인기는 모듈 속성으로 부른다
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # 시작 시점의 색인 행만 — 삭제 행은 그 뒤에 생겼다

    doc = await get_doc(doc_id)
    assert doc.status == 'deleted' and doc.is_active is False, '삭제된 문서가 ready로 올라갔다'
    assert await indexed_chunk_texts(doc_id) != []  # 전제: 청크가 실제로 들어갔다

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}
    assert await indexed_chunk_texts(doc_id) == []
    assert await _row_states(doc_id) == [('index_document', 'done'), ('drop_documents', 'done')]


@pytest.mark.asyncio
async def test_색인된_문서가_든_폴더를_참조_끄면_빠지고_켜면_돌아온다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed
):
    """FOL-11 — 폴더 토글은 소속 문서 전부를 한 행으로 메타 갱신한다. 폴더 밖 문서는 영향받지 않는다."""
    folder_id = await make_folder(client, '정책')
    inside = (await upload(client, '환불정책.md', MD))['document_id']
    outside = (await upload(client, '배송정책.md', MD))['document_id']
    assert (await client.patch(f'/kms/documents/{inside}', json={'folder_id': folder_id})).status_code == 200
    await ingest(inside)
    await ingest(outside)
    indexed_at = (await get_doc(inside)).indexed_at
    assert {inside, outside} <= await _candidate_doc_ids(tenant_id)

    assert (await client.patch(f'/kms/folders/{folder_id}', json={'is_searchable': False})).status_code == 200
    assert inside in await _candidate_doc_ids(tenant_id)  # drain 전에는 아직 보인다
    assert await ingest(inside) == {'done': 1, 'failed': 0}
    cands = await _candidate_doc_ids(tenant_id)
    assert inside not in cands, '참조 끈 폴더의 문서가 검색 후보에 남았다'
    assert outside in cands

    assert (await client.patch(f'/kms/folders/{folder_id}', json={'is_searchable': True})).status_code == 200
    await ingest(inside)
    assert {inside, outside} <= await _candidate_doc_ids(tenant_id)
    assert (await get_doc(inside)).indexed_at == indexed_at  # 재색인 없이 메타만 바뀐다
