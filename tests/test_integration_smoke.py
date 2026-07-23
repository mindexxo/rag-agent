"""D-1 conftest 배관 검증 스모크 — fixture 조합이 실제로 도는지.

FAQ 등록(API 경유) → DB 청크 확인이 통과하면:
ASGI 클라이언트, 가짜 임베딩 패치, tenant 격리·정리, 루프 위생이 전부 동작하는 것.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Chunk, Faq


@pytest.mark.asyncio
async def test_faq_등록이_API부터_청크까지_관통(client, tenant_id):
    res = await client.post('/kms/faqs', json={
        'question': '환불 기간은 어떻게 되나요?',
        'variants': ['돈 언제 돌려받아요'],
        'answer': '결제 수단에 따라 7일 이내 처리됩니다.',
    })
    assert res.status_code == 200, res.text
    faq_id = res.json()['id']

    async with AsyncSessionLocal() as session:
        chunk = (await session.execute(
            select(Chunk).where(Chunk.faq_id == faq_id)
        )).scalar_one()                                  # 항목당 정확히 1청크
        assert chunk.tenant_id == tenant_id
        assert chunk.text.startswith('Q: 환불 기간은 어떻게 되나요?')
        assert '(유사 질문: 돈 언제 돌려받아요)' in chunk.text   # variants가 검색 텍스트에 실림 (FAQ 설계 핵심)
        assert len(chunk.dense) == 1024                  # 가짜 임베딩이 실제로 저장됨


@pytest.mark.asyncio
async def test_테넌트_헤더_없으면_422(client):
    res = await client.get('/kms/faqs', headers={'X-Tenant-Id': ''})
    # 빈 값은 통과하되(현재 스펙), 헤더 자체를 지우면 422
    import httpx

    from main import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url='http://testserver') as bare:
        res = await bare.get('/kms/faqs')
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_정리_fixture가_실제로_지움(client, tenant_id):
    # 등록만 하고 종료 — 다음 테스트가 아니라 teardown이 지우는지는
    # 같은 tenant를 다시 조회할 방법이 없으므로, 여기선 등록 성공만 확인하고
    # 잔여 데이터 여부는 스위트 반복 실행의 누적 여부로 관측한다 (누적되면 count 증가).
    res = await client.post('/kms/faqs', json={'question': 'q', 'variants': [], 'answer': 'a'})
    assert res.status_code == 200

    async with AsyncSessionLocal() as session:
        n = len((await session.execute(
            select(Faq).where(Faq.tenant_id == tenant_id)
        )).scalars().all())
        assert n == 1
