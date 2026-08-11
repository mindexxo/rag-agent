"""D-6: 테넌트 격리 확장 통합 테스트.

기존 격리 커버: documents ORM(test_tenant_isolation), FAQ CRUD 404, supersede 교차 보호,
semantic 캐시, last_used_at. 여기서는 남은 표면 — **검색 후보**, 대화 목록/메시지, 폴더.
RLS 없이 WHERE 절이 유일한 방어선이라는 설계 전제의 전수 검증.
"""

import pytest
from httpx import ASGITransport, AsyncClient

from database import AsyncSessionLocal
from rag.retriever import retrieve_candidates
from tests.conftest import sse_meta


def _client_for(tenant: str) -> AsyncClient:
    from main import app
    return AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver',
                       headers={'X-Tenant-Id': tenant})


@pytest.mark.asyncio
async def test_검색_후보에_타_테넌트_청크가_없다(client, tenant_id, other_tenant_id, fake_embed):
    """격리의 최전선 — 후보 단계에서 새면 답변에 타사 수치가 섞인다 (corpus_v2 카나리아의 코드판)."""
    await client.post('/kms/faqs', json={'question': 'A사 반품 기간은?', 'variants': [], 'answer': '14일'})
    async with _client_for(other_tenant_id) as other:
        await other.post('/kms/faqs', json={'question': 'B사 반품 기간은?', 'variants': [], 'answer': '7일'})

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
    await client.post('/kms/faqs', json={'question': 'A사 전용', 'variants': [], 'answer': 'a'})
    async with _client_for(other_tenant_id) as other:
        listing = (await other.get('/kms/faqs')).json()
        assert listing == []
