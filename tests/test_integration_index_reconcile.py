"""색인 정합 복구(reconcile) — 색인에서 사라진 ready 문서·FAQ는 재등재하고, PG에 없는 청크는 지운다."""

import pytest

from database import AsyncSessionLocal
from rag import os_reconcile
from rag.search import os_index
from tests.conftest import faq_doc, indexed_chunk_texts, ingest, sync_faq
from tests.helpers_documents import MD, get_doc, upload


@pytest.mark.asyncio
async def test_reconcile은_사라진_ready_문서를_재등재하고_유령_청크를_지운다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed
):
    body = await upload(client, '환불정책.md', MD)
    doc_id = body['document_id']
    await ingest(doc_id)
    # 누락 재현: 색인에서만 지운다(스냅샷 복구 뒤 유실 등). PG는 ready 그대로.
    await os_index._delete_by_terms('document_id', [doc_id])
    assert await indexed_chunk_texts(doc_id) == []
    # 잉여 재현: PG에 없는 document_id의 청크를 이 테넌트로 넣는다.
    ghost = os_index.build_doc(
        os_index.ParsedChunk(
            id=os_index.chunk_os_id(document_id=999_999_001, chunk_index=0),
            tenant_id=tenant_id,
            document_id=999_999_001,
            faq_id=None,
            text='유령',
            heading_path=[],
            page=None,
            meta={},
            dense=[0.01] * 1024,
        ),
        '유령.md',
        1,
        searchable=True,
    )
    await os_index.bulk_index([ghost])

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile(s, tenant_id)  # 테넌트 스코프 — 남의 문서는 건드리지 않는다
    assert r['indexed'] == 1 and r['deleted'] == 1 and r['unit'] == 'document'
    assert (await get_doc(doc_id)).status == 'pending'  # 재등재 = pending 되돌림 + INDEX 행
    assert await indexed_chunk_texts(999_999_001) == []

    assert await ingest(doc_id) == {'done': 1, 'failed': 0}  # 워커가 다시 색인
    assert (await get_doc(doc_id)).status == 'ready'
    assert await indexed_chunk_texts(doc_id)
    async with AsyncSessionLocal() as s:
        assert (await os_reconcile.reconcile(s, tenant_id))['indexed'] == 0


@pytest.mark.asyncio
async def test_reconcile_faqs는_사라진_FAQ를_재등재하고_유령을_지운다(client, tenant_id, fake_embed):
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'variants': [], 'answer': '7일'})
    faq_id = res.json()['id']
    await sync_faq(faq_id)
    assert await faq_doc(faq_id)
    await os_index._delete_by_terms('faq_id', [faq_id])
    ghost = os_index.build_doc(
        os_index.ParsedChunk(
            id=os_index.chunk_os_id(faq_id=999_999_002),
            tenant_id=tenant_id,
            document_id=None,
            faq_id=999_999_002,
            text='Q: 유령 A: 유령',
            heading_path=['유령'],
            page=None,
            meta={},
            dense=[0.01] * 1024,
        ),
        None,
        None,
        searchable=True,
    )
    await os_index.bulk_index([ghost])

    async with AsyncSessionLocal() as s:
        r = await os_reconcile.reconcile_faqs(s, tenant_id)
    assert r['indexed'] == 1 and r['deleted'] == 1 and r['unit'] == 'faq'
    assert await faq_doc(999_999_002) is None
    assert await sync_faq(faq_id) == {'done': 1, 'failed': 0}
    assert await faq_doc(faq_id)


@pytest.mark.asyncio
async def test_색인_도중_reconcile이_끼어들어도_청크_없는_ready가_되지_않는다(
    client, tenant_id, fake_queue, blob_tmp, fake_embed, monkeypatch
):
    """RC-08 — 청크가 엔진에 들어간 뒤 ready 커밋 전에 --apply가 돈다. 색인 중(pending) 문서는 잉여가 아니다."""
    doc_id = (await upload(client, '환불정책.md', MD))['document_id']
    real = os_index.index_parsed_document
    seen: list[dict] = []

    async def _index_then_reconcile(**kw: object) -> None:
        await real(**kw)
        async with AsyncSessionLocal() as s:
            seen.append(await os_reconcile.reconcile(s, tenant_id))

    monkeypatch.setattr(os_index, 'index_parsed_document', _index_then_reconcile)  # 색인기는 모듈 속성으로 부른다
    assert await ingest(doc_id) == {'done': 1, 'failed': 0}

    assert seen[0]['deleted'] == 0, '색인 중인 문서의 청크를 잉여로 지웠다'
    assert seen[0]['indexed'] == 0  # pending은 누락도 아니다
    assert (await get_doc(doc_id)).status == 'ready'
    assert await indexed_chunk_texts(doc_id) != [], '청크 없는 ready'
    async with AsyncSessionLocal() as s:
        after = await os_reconcile.reconcile(s, tenant_id)
    assert after['indexed'] == 0 and after['deleted'] == 0
