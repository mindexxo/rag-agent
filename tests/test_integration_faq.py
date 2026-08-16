"""D-2: FAQ 수명주기 통합 테스트.

등록→수정(재임베딩·캐시 무효화)→is_active 토글(검색 제외·복귀·캐시 자가치유)→삭제(cascade).
리뷰 발견(P1-15 PATCH 빈 문자열, P1-16 is_active 캐시 빈틈)의 회귀 방어를 겸한다.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag import cache
from rag.models import AnswerCache as AnswerCacheRow, Chunk, Faq
from rag.retriever import retrieve_candidates


async def _create_faq(client, question='환불 기간은?', answer='7일 이내 처리됩니다.') -> int:
    res = await client.post('/kms/faqs', json={
        'question': question, 'variants': ['돈 언제 돌려받아요'], 'answer': answer,
    })
    assert res.status_code == 200, res.text
    return res.json()['id']


async def _candidate_faq_ids(tenant_id: str) -> set[int]:
    """검색 후보에 오르는 FAQ id들 — is_active 검색 편입/제외 판정용."""
    async with AsyncSessionLocal() as session:
        cands = await retrieve_candidates(session, tenant_id, '아무 질의', top_n=20)
        return {c.faq_id for c in cands.chunks if c.faq_id}


@pytest.mark.asyncio
async def test_내용_수정시_청크_재임베딩_여전히_1개(client, tenant_id):
    faq_id = await _create_faq(client)

    res = await client.patch(f'/kms/faqs/{faq_id}', json={'question': '환불은 며칠 걸리나요?'})
    assert res.status_code == 200

    async with AsyncSessionLocal() as session:
        chunks = (await session.execute(
            select(Chunk).where(Chunk.faq_id == faq_id)
        )).scalars().all()
        assert len(chunks) == 1                              # 재인덱싱은 삭제 후 재삽입 — 늘어나면 안 됨
        assert chunks[0].text.startswith('Q: 환불은 며칠 걸리나요?')


@pytest.mark.asyncio
async def test_variants만_수정해도_재임베딩(client, tenant_id):
    # variants는 dense 매칭을 견인하는 핵심 콘텐츠 — variants-only PATCH가
    # content_changed로 인정돼 청크 텍스트에 반영돼야 한다 (뮤테이션 생존자 M5·M8 킬)
    faq_id = await _create_faq(client)

    res = await client.patch(f'/kms/faqs/{faq_id}', json={'variants': ['새로 추가한 구어체 질문']})
    assert res.status_code == 200

    async with AsyncSessionLocal() as session:
        chunk = (await session.execute(
            select(Chunk).where(Chunk.faq_id == faq_id)
        )).scalar_one()
        assert '(유사 질문: 새로 추가한 구어체 질문)' in chunk.text


@pytest.mark.asyncio
async def test_off_단독_토글도_캐시_무효화(client, tenant_id, fake_embed):
    # turned_off 분기 검증 — semantic 자가치유가 있어도 명시 무효화가 1차 방어선 (M7 킬)
    faq_id = await _create_faq(client)
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, 'q', 'a', [], [-faq_id])
        await session.commit()

    await client.patch(f'/kms/faqs/{faq_id}', json={'is_active': False})

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_내용_수정시_해당_FAQ_근거_캐시_무효화(client, tenant_id, fake_embed):
    faq_id = await _create_faq(client)
    async with AsyncSessionLocal() as session:
        # 이 FAQ(-faq_id)를 근거로 만든 semantic 캐시가 있다고 가정
        await cache.save_answer(session, tenant_id, '환불 얼마나 걸려요', '옛 답변', [], [-faq_id])
        await session.commit()

    res = await client.patch(f'/kms/faqs/{faq_id}', json={'answer': '5일 이내로 변경되었습니다.'})
    assert res.status_code == 200

    async with AsyncSessionLocal() as session:
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []                                    # 음수 네임스페이스 무효화 동작


@pytest.mark.asyncio
async def test_is_active_토글_검색_제외와_복귀(client, tenant_id):
    faq_id = await _create_faq(client)
    assert faq_id in await _candidate_faq_ids(tenant_id)     # 등록 직후 검색 편입

    await client.patch(f'/kms/faqs/{faq_id}', json={'is_active': False})
    assert faq_id not in await _candidate_faq_ids(tenant_id)  # off → 검색 제외

    await client.patch(f'/kms/faqs/{faq_id}', json={'is_active': True})
    assert faq_id in await _candidate_faq_ids(tenant_id)     # on → 복귀


@pytest.mark.asyncio
async def test_꺼진_동안_캐시는_켠_후_doc집합_비교로_자가치유(client, tenant_id, fake_embed):
    """리뷰 P1-16 검증: off 동안 저장된 캐시(FAQ 미반영)가 on 후에도 서빙되는 빈틈.
    exact 캐시 제거 후 semantic의 doc-set 비교가 이를 구조적으로 막는지 확인."""
    faq_id = await _create_faq(client)
    async with AsyncSessionLocal() as session:
        # off 기간의 답변: 문서 123만 근거 (FAQ 미포함)
        await cache.save_answer(session, tenant_id, '배송비 얼마예요', 'FAQ 없던 시절 답', [], [123])
        await session.commit()

        # 같은 근거 집합이면 hit (전제 확인)
        hit = await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [123])
        assert hit is not None

        # FAQ가 켜져 검색 근거가 [123, -faq]로 바뀌면 → 집합 불일치로 miss (자가치유)
        hit = await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [123, -faq_id])
        assert hit is None


@pytest.mark.asyncio
async def test_삭제시_청크_cascade와_캐시_무효화(client, tenant_id, fake_embed):
    faq_id = await _create_faq(client)
    async with AsyncSessionLocal() as session:
        await cache.save_answer(session, tenant_id, 'q', 'a', [], [-faq_id])
        await session.commit()

    res = await client.delete(f'/kms/faqs/{faq_id}')
    assert res.status_code == 204

    async with AsyncSessionLocal() as session:
        assert (await session.get(Faq, faq_id)) is None
        chunks = (await session.execute(
            select(Chunk).where(Chunk.faq_id == faq_id)
        )).scalars().all()
        assert chunks == []                                  # FK CASCADE
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_PATCH_빈_문자열은_422(client, tenant_id):
    faq_id = await _create_faq(client)
    for field in ('question', 'answer'):
        res = await client.patch(f'/kms/faqs/{faq_id}', json={field: '   '})
        assert res.status_code == 422, field                 # P1-15 가드 회귀 방어


@pytest.mark.asyncio
async def test_타_테넌트_FAQ는_404(client, tenant_id):
    import uuid

    import httpx

    from main import app
    faq_id = await _create_faq(client)

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url='http://testserver',
                                 headers={'X-Tenant-Id': str(uuid.uuid4())}) as other:
        assert (await other.patch(f'/kms/faqs/{faq_id}', json={'answer': '탈취'})).status_code == 404
        assert (await other.delete(f'/kms/faqs/{faq_id}')).status_code == 404
