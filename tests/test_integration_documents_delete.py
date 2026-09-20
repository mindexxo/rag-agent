"""문서 삭제 — 단건과 일괄이 **같은 경로**(rag.documents.soft_delete_documents)를 탄다 (#174).

소프트 삭제의 네 동작(상태·캐시·색인 DROP·row 보존)과 일괄 삭제의 전부-또는-무 규칙,
그리고 "이미 삭제된 문서는 통과"라는 멱등 규칙을 고정한다.
공용 헬퍼는 tests/helpers_documents.py.
"""
import pytest
from sqlalchemy import delete as sql_delete, select

from database import AsyncSessionLocal
from rag import cache, outbox
from rag.models import AnswerCache as AnswerCacheRow, SearchIndexOutbox
from tests.conftest import ingest
from tests.helpers_documents import MD, chunk_texts, get_doc, upload


@pytest.mark.asyncio
async def test_소프트_삭제_청크와_캐시_제거_row_보존(client, tenant_id, fake_queue, blob_tmp, fake_embed):
    body = await upload(client, '환불정책.md', MD)
    await ingest(body['document_id'])
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, '반품 기간', '14일', [], [body['document_id']])
        await session.commit()

    res = await client.delete(f"/kms/documents/{body['document_id']}")
    assert res.status_code == 204

    doc = await get_doc(body['document_id'])
    assert doc is not None                                       # row 보존 (과거 인용 다운로드용)
    assert doc.status == 'deleted' and doc.is_active is False
    assert await chunk_texts(body['document_id']) != []                  # drain 전 — 가이드대로 아직 색인에 있다 (E2E #4)
    await ingest(body['document_id'])                                     # DROP_DOCUMENTS 행 처리
    assert await chunk_texts(body['document_id']) == []                  # 검색 인덱스에서 제거
    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []                                        # 근거 캐시 무효화


# ── #174 문서 일괄 삭제 (DELETE /kms/documents?ids=) ──────────

async def _bulk_delete(client, ids, tenant=None):
    headers = {'X-Tenant-Id': tenant} if tenant else None
    return await client.request('DELETE', '/kms/documents', params={'ids': ids}, headers=headers)


@pytest.mark.asyncio
async def test_일괄_삭제는_목록에서_사라지고_전_버전을_함께_내린다(client, tenant_id, fake_queue, blob_tmp):
    a_v1 = await upload(client, '환불정책.md', MD)
    await ingest(a_v1['document_id'])
    a_v2 = await upload(client, '환불정책.md', MD)          # 같은 이름 재업로드 → v2
    b = await upload(client, '배송정책.md', MD)

    assert (await _bulk_delete(client, [a_v2['document_id'], b['document_id']])).status_code == 204

    # 요청한 건 v2지만 같은 filename의 v1도 함께 내려간다
    for did in (a_v1['document_id'], a_v2['document_id'], b['document_id']):
        assert (await get_doc(did)).status == 'deleted'
    assert (await client.get('/kms/documents')).json()['items'] == []


@pytest.mark.asyncio
async def test_일괄_삭제는_이미_삭제된_문서를_통과시킨다(client, tenant_id, fake_queue, blob_tmp):
    """멱등 — 삭제는 '그 문서가 없는 상태'를 만드는 것이라 이미 그 상태면 성공이다.
    단건 DELETE가 원래 그렇게 동작했고 같은 규칙을 쓴다."""
    gone = await upload(client, '환불정책.md', MD)
    alive = await upload(client, '배송정책.md', MD)
    assert (await client.delete(f"/kms/documents/{gone['document_id']}")).status_code == 204

    res = await _bulk_delete(client, [gone['document_id'], alive['document_id']])
    assert res.status_code == 204
    assert (await get_doc(alive['document_id'])).status == 'deleted'


@pytest.mark.asyncio
async def test_일괄_삭제_대상이_어긋나면_아무것도_안_지운다(client, tenant_id, other_tenant_id,
                                                        fake_queue, blob_tmp):
    mine = await upload(client, '환불정책.md', MD)
    other = await client.post('/kms/documents', files={'file': ('남의문서.md', MD, 'text/markdown')},
                              headers={'X-Tenant-Id': other_tenant_id})

    for label, bad in (('없는 id', 999999), ('다른 테넌트', other.json()['document_id'])):
        res = await _bulk_delete(client, [mine['document_id'], bad])
        assert res.status_code == 404, f'{label} → {res.status_code}'
        assert (await get_doc(mine['document_id'])).status != 'deleted', f'{label}: 성공분이 지워졌다'
    assert '999999' in (await _bulk_delete(client, [999999])).json()['detail']


@pytest.mark.asyncio
async def test_일괄_삭제_캐시_무효화와_색인_제거(client, tenant_id, fake_queue, blob_tmp):
    """라우터를 거쳐 캐시가 지워지는지, DROP 대기열이 한 행에 전 버전을 담는지."""
    a = await upload(client, '환불정책.md', MD)
    b = await upload(client, '배송정책.md', MD)
    await ingest(a['document_id'])
    await ingest(b['document_id'])

    async with AsyncSessionLocal() as s:
        await cache.save_answer(s, tenant_id, '질의 하나', '답1', [], [a['document_id']])
        await cache.save_answer(s, tenant_id, '질의 둘', '답2', [], [b['document_id']])
        await s.commit()
        await s.execute(sql_delete(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant_id))
        await s.commit()

    assert (await _bulk_delete(client, [a['document_id'], b['document_id']])).status_code == 204

    async with AsyncSessionLocal() as s:
        remain = (await s.execute(select(AnswerCacheRow.answer)
                                  .where(AnswerCacheRow.tenant_id == tenant_id))).scalars().all()
        rows = (await s.execute(select(SearchIndexOutbox)
                                .where(SearchIndexOutbox.tenant_id == tenant_id))).scalars().all()
    assert remain == []                                  # 둘 다 근거가 사라졌다
    assert len(rows) == 1 and rows[0].op == outbox.DROP_DOCUMENTS
    assert sorted(rows[0].payload['document_ids']) == sorted([a['document_id'], b['document_id']])

    await ingest(a['document_id'])                       # 한 행에 둘 다 담겨 있다
    assert await chunk_texts(a['document_id']) == []
    assert await chunk_texts(b['document_id']) == []


@pytest.mark.asyncio
async def test_일괄_삭제_상한과_빈_요청(client, tenant_id, fake_queue, blob_tmp):
    assert (await _bulk_delete(client, list(range(1, 201)))).status_code == 404   # 스키마는 통과
    assert (await _bulk_delete(client, list(range(1, 202)))).status_code == 422
    assert (await _bulk_delete(client, [])).status_code == 204                   # 무동작


@pytest.mark.asyncio
async def test_단건_삭제_동작_보존(client, tenant_id, fake_queue, blob_tmp):
    """공용 함수로 옮긴 뒤에도 단건 계약이 그대로인지 — 없는 id는 404, 재호출은 204(멱등)."""
    doc = await upload(client, '환불정책.md', MD)
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    assert (await client.delete(f"/kms/documents/{doc['document_id']}")).status_code == 204
    assert (await client.delete('/kms/documents/999999')).status_code == 404
