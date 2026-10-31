"""API 계약 정리 — id=0 거절, 문서 상태 어휘, 삭제 문서 단건 404, 대화 목록 동점 정렬.

HTTP로만 확인한다. 목록 필터의 `folder_id=0`·`category_id=0`(=미분류)은 그대로이고, 등록·수정에서만 0을 거절한다.
"""

import pytest
from sqlalchemy import select, update

from database import AsyncSessionLocal
from rag.models import ALIVE_DOCUMENT_STATUSES, Conversation
from schemas.kms import DOC_STATUSES
from tests.conftest import register_faq, sse_meta
from tests.helpers_documents import MD, doc_data, upload
from tests.polling import wait_inflight


def _locs(res) -> list[tuple]:
    return [tuple(e['loc']) for e in res.json()['detail']]


@pytest.mark.asyncio
async def test_등록_수정의_folder_id_0은_422이고_목록_필터의_0은_미분류다(client, tenant_id, fake_queue, blob_tmp):
    """DOC-27 — 0은 null(미분류)이 아니다. 업로드·단건 수정·일괄 수정 셋 다."""
    res = await client.post('/kms/documents', files={'file': ('a.md', MD, 'text/markdown'), **doc_data(folder_id=0)})
    assert res.status_code == 422 and 'folder_id' in res.json()['detail'], (
        res.text
    )  # document-data 파트는 수동 검증이라 문장형

    doc = await upload(client, 'a.md', MD)
    res = await client.patch(f'/kms/documents/{doc["document_id"]}', json={'folder_id': 0})
    assert res.status_code == 422 and ('body', 'folder_id') in _locs(res)
    res = await client.patch('/kms/documents', json={'document_ids': [doc['document_id']], 'folder_id': 0})
    assert res.status_code == 422 and ('body', 'folder_id') in _locs(res)

    assert (await client.patch(f'/kms/documents/{doc["document_id"]}', json={'folder_id': None})).status_code == 200
    items = (await client.get('/kms/documents', params={'folder_id': 0})).json()['items']
    assert [d['document_id'] for d in items] == [doc['document_id']]


@pytest.mark.asyncio
async def test_FAQ_등록_수정의_category_id_0은_422(client, tenant_id):
    """FAQ-C22 — DOC-27과 같은 규칙."""
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'answer': '7일', 'category_id': 0})
    assert res.status_code == 422 and ('body', 'category_id') in _locs(res)

    faq_id = await register_faq(client)
    res = await client.patch(f'/kms/faqs/{faq_id}', json={'category_id': 0})
    assert res.status_code == 422 and ('body', 'category_id') in _locs(res)
    assert (await client.patch(f'/kms/faqs/{faq_id}', json={'category_id': None})).status_code == 200


@pytest.mark.asyncio
async def test_문서_상태_어휘에_parsing_embedding이_없다(client, tenant_id):
    """DOC-70 — 기록된 적 없는 값을 어휘에서 뺐다. 필터에 주면 '알 수 없는 status' 422."""
    assert ALIVE_DOCUMENT_STATUSES == ('pending', 'ready')
    assert DOC_STATUSES == {'pending', 'ready', 'failed'}
    for stale in ('parsing', 'embedding'):
        res = await client.get('/kms/documents', params={'status': stale})
        assert res.status_code == 422 and '알 수 없는 status' in res.json()['detail'], stale
    assert (await client.get('/kms/documents', params={'status': 'pending'})).status_code == 200


@pytest.mark.asyncio
async def test_삭제된_문서는_단건_조회도_404(client, tenant_id, fake_queue, blob_tmp):
    """DOC-77 — 목록·수정·삭제와 같은 기준. 다운로드만 예외로 소프트 삭제 뒤에도 된다(이력 보존, 기존 테스트가 고정)."""
    doc = await upload(client, '삭제될문서.md', MD)
    doc_id = doc['document_id']
    assert (await client.get(f'/kms/documents/{doc_id}')).status_code == 200
    assert (await client.delete(f'/kms/documents/{doc_id}')).status_code == 204

    res = await client.get(f'/kms/documents/{doc_id}')
    assert res.status_code == 404
    assert res.json() == {'detail': 'document not found'}
    assert (await client.get(f'/kms/documents/{doc_id}/download')).status_code == 200


@pytest.mark.asyncio
async def test_대화_목록은_last_used_at_동점이면_id_역순으로_페이지가_겹치지_않는다(
    client, tenant_id, fake_llm, pass_gate
):
    """CNV-13 — 2차 정렬 키가 없으면 동점 행의 순서가 페이지마다 달라질 수 있다."""
    await register_faq(client)
    ids = [sse_meta(await client.post('/kms/query', json={'query': f'질문 {i}'}))['conversation_id'] for i in range(3)]
    assert await wait_inflight(tenant_id, 0) == 0  # 생성 태스크의 last_used_at 갱신이 끝난 뒤에 동점을 만든다

    async with AsyncSessionLocal() as s:
        tie = (await s.execute(select(Conversation.last_used_at).where(Conversation.id == ids[0]))).scalar_one()
        await s.execute(update(Conversation).where(Conversation.id.in_(ids)).values(last_used_at=tie))
        await s.commit()
        stamps = (await s.execute(select(Conversation.last_used_at).where(Conversation.id.in_(ids)))).scalars().all()
    assert len(set(stamps)) == 1

    paged = []
    for offset in range(3):
        res = await client.get('/kms/conversations', params={'limit': 1, 'offset': offset})
        paged += [c['conversation_id'] for c in res.json()['items']]
    assert paged == sorted(ids, reverse=True)
