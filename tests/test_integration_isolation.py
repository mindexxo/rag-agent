"""D-6: 테넌트 격리 확장 통합 테스트.

기존 격리 커버: documents ORM(test_tenant_isolation), FAQ CRUD 404, supersede 교차 보호,
semantic 캐시, last_used_at. 여기서는 남은 표면 — **검색 후보**, 대화 목록/메시지, 폴더.
RLS 없이 WHERE 절이 유일한 방어선이라는 설계 전제의 전수 검증.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from database import AsyncSessionLocal
from rag.retriever import retrieve_candidates
from tests.conftest import sse_meta, sync_faq


def _client_for(tenant: str) -> AsyncClient:
    from main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver',
                       headers={'X-Tenant-Id': tenant})


@pytest.mark.asyncio
async def test_검색_후보에_타_테넌트_청크가_없다(client, tenant_id, other_tenant_id, fake_embed):
    """격리의 최전선 — 후보 단계에서 새면 답변에 타사 수치가 섞인다 (corpus_v2 카나리아의 코드판)."""
    await sync_faq((await client.post('/kms/faqs', json={'question': 'A사 반품 기간은?', 'variants': [], 'answer': '14일'})).json()['id'])
    async with _client_for(other_tenant_id) as other:
        await sync_faq((await other.post('/kms/faqs', json={'question': 'B사 반품 기간은?', 'variants': [], 'answer': '7일'})).json()['id'])
    async with AsyncSessionLocal() as session:
        a_texts = [c.text for c in
                   (await retrieve_candidates(session, tenant_id, '반품 기간', top_n=20)).chunks]
        assert any('A사' in t for t in a_texts)               # 자기 것은 보임 (양성 대조)
        assert not any('B사' in t for t in a_texts)           # 타 테넌트 청크 유입 금지

        b_texts = [c.text for c in
                   (await retrieve_candidates(session, other_tenant_id, '반품 기간', top_n=20)).chunks]
        assert any('B사' in t for t in b_texts)
        assert not any('A사' in t for t in b_texts)


@pytest.mark.asyncio
async def test_대화_목록과_메시지_격리(client, tenant_id, other_tenant_id, fake_llm):
    # A가 대화 생성 (근거 없음 즉시 경로 — 문서 불필요)
    res = await client.post('/kms/query', json={'query': 'A사 질문입니다'})
    conv_id = sse_meta(res)['conversation_id']

    async with _client_for(other_tenant_id) as other:
        listing = (await other.get('/kms/conversations')).json()
        assert all(c['conversation_id'] != conv_id for c in listing['items'])   # 목록 격리 (#10: {items, has_more})

        res = await other.get(f'/kms/conversations/{conv_id}/messages')
        assert res.status_code == 404                                  # 메시지 격리 (#10: 소유 검증으로 404)


@pytest.mark.asyncio
async def test_폴더_격리(client, tenant_id, other_tenant_id):
    res = await client.post('/kms/folders', json={'name': 'A사 내부폴더'})
    assert res.status_code == 200, res.text
    folder_id = res.json()['id']

    async with _client_for(other_tenant_id) as other:
        listing = (await other.get('/kms/folders')).json()
        assert all(f['id'] != folder_id for f in listing)              # 목록 격리
        assert (await other.patch(f'/kms/folders/{folder_id}',
                                  json={'name': '탈취'})).status_code == 404
        assert (await other.delete(f'/kms/folders/{folder_id}')).status_code == 404


@pytest.mark.asyncio
async def test_FAQ_목록_격리(client, tenant_id, other_tenant_id):
    await sync_faq((await client.post('/kms/faqs', json={'question': 'A사 전용', 'variants': [], 'answer': 'a'})).json()['id'])
    async with _client_for(other_tenant_id) as other:
        listing = (await other.get('/kms/faqs')).json()
        assert listing == []


@pytest.mark.asyncio
async def test_문서_일괄변경_격리(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """#166으로 늘어난 표면 — 요청 바디로 id 목록을 받는 첫 API라 격리를 여기에 남긴다.

    이 경로가 새면 피해가 조용하다: 대상 확정이 tenant를 안 걸면 남의 문서 id가 그대로
    outbox payload에 실리고, os_index.sync_meta_documents_now는 document_ids에 tenant 필터를
    걸지 않으므로(호출부의 사전 스코프를 전제한다) 남의 색인 메타가 덮인다.
    """
    md = '# 제목\n\n본문 내용\n'.encode()
    mine = (await client.post('/kms/documents',
                              files={'file': ('A사문서.md', md, 'text/markdown')})).json()

    async with _client_for(other_tenant_id) as other:
        theirs = (await other.post('/kms/documents',
                                   files={'file': ('B사문서.md', md, 'text/markdown')})).json()
        folder = (await other.post('/kms/folders', json={'name': 'B사 폴더'})).json()

        # 남의 문서를 섞어 보내면 전체가 404 — 자기 문서(theirs)도 바뀌면 안 된다
        res = await other.patch('/kms/documents',
                                json={'document_ids': [theirs['document_id'], mine['document_id']],
                                      'is_searchable': False})
        assert res.status_code == 404
        still = (await other.get(f"/kms/documents/{theirs['document_id']}")).json()
        assert still['is_searchable'] is True, '전체 거절인데 자기 문서가 바뀌었다'

        # 남의 폴더로 자기 문서를 옮기려는 시도도 404
        assert (await client.patch('/kms/documents',
                                   json={'document_ids': [mine['document_id']],
                                         'folder_id': folder['id']})).status_code == 404

    # 남의 요청으로 내 문서가 바뀌지 않았다
    after = (await client.get(f"/kms/documents/{mine['document_id']}")).json()
    assert after['is_searchable'] is True and after['folder_id'] is None


@pytest.mark.asyncio
async def test_문서_일괄삭제_격리(client, tenant_id, other_tenant_id, fake_queue, blob_tmp):
    """#174로 늘어난 표면 — 쿼리 파라미터로 받은 id 목록이 tenant 스코프를 지나야 한다.

    이 경로가 새면 피해가 크고 조용하다: 대상 확정이 tenant를 안 걸면 남의 문서가
    soft delete되고, 그 id가 DROP_DOCUMENTS payload에 실려 색인에서도 청크가 지워진다
    (os_index.drop_documents_now는 document_ids에 tenant 필터를 걸지 않는다).
    """
    md = '# 제목\n\n본문 내용\n'.encode()
    mine = (await client.post('/kms/documents',
                              files={'file': ('A사문서.md', md, 'text/markdown')})).json()

    async with _client_for(other_tenant_id) as other:
        theirs = (await other.post('/kms/documents',
                                   files={'file': ('B사문서.md', md, 'text/markdown')})).json()

        # 남의 문서를 섞어 보내면 전체 404 — 자기 문서(theirs)도 지워지면 안 된다
        res = await other.request('DELETE', '/kms/documents',
                                  params={'ids': [theirs['document_id'], mine['document_id']]})
        assert res.status_code == 404
        still = (await other.get(f"/kms/documents/{theirs['document_id']}")).json()
        assert still['status'] != 'deleted', '전체 거절인데 자기 문서가 지워졌다'

        # 남의 문서만 지목해도 404 (존재 자체를 알려주지 않는다)
        assert (await other.request('DELETE', '/kms/documents',
                                    params={'ids': [mine['document_id']]})).status_code == 404

    after = (await client.get(f"/kms/documents/{mine['document_id']}")).json()
    assert after['status'] != 'deleted'


@pytest.mark.asyncio
async def test_같은_파일명이_두_테넌트에_있어도_내_것만_지운다(client, tenant_id, other_tenant_id,
                                                          fake_queue, blob_tmp):
    """삭제는 id가 아니라 **filename**으로 전 버전을 매칭한다(#174) — 그래서 UPDATE에 tenant
    필터가 빠지면 같은 이름을 쓰는 남의 문서까지 지워진다. 흔한 파일명일수록 위험하다."""
    md = '# 제목\n\n본문 내용\n'.encode()
    same_name = '환불정책.md'
    mine = (await client.post('/kms/documents',
                              files={'file': (same_name, md, 'text/markdown')})).json()
    async with _client_for(other_tenant_id) as other:
        theirs = (await other.post('/kms/documents',
                                   files={'file': (same_name, md, 'text/markdown')})).json()

        assert (await client.request('DELETE', '/kms/documents',
                                     params={'ids': [mine['document_id']]})).status_code == 204

        still = (await other.get(f"/kms/documents/{theirs['document_id']}")).json()
        assert still['status'] != 'deleted', '같은 파일명을 쓰는 남의 문서가 지워졌다'
