"""D-3: semantic 캐시 동작 통합 테스트.

hit/miss 판정(유사도·doc집합), 테넌트 격리, 거절 답변 캐시 제외, 무효화 정확성.
exact(Redis) 계층은 제거됨 — semantic(PG) 단일 계층 기준.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.cache import AnswerCache
from rag.models import AnswerCache as AnswerCacheRow, Conversation
from rag.prompts import NO_EVIDENCE_ANSWER
from rag.retriever import RetrievalResult
from rag.service import PreparedRag, RagService
from schemas.kms import SourceCitation


@pytest.mark.asyncio
async def test_같은_질의_같은_근거집합이면_hit(tenant_id, fake_embed):
    cache = AnswerCache()
    src = [SourceCitation(document_id=5, filename='정책.pdf', version=1)]
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '배송비 얼마예요', '3천원입니다', src, [5])
        await session.commit()

        hit = await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [5])
        assert hit is not None
        assert hit.answer == '3천원입니다'
        assert hit.sources == src
        assert hit.kind == 'semantic'
        await session.commit()
        row = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalar_one()
        assert row.hit_count == 1                            # 운영 관측 카운터 갱신


@pytest.mark.asyncio
async def test_다른_질의는_miss(tenant_id, fake_embed):
    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '배송비 얼마예요', '3천원', [], [5])
        await session.commit()
        # 가짜 벡터는 텍스트가 다르면 사실상 직교 → 유사도 미달 miss
        assert await cache.get_semantic(session, tenant_id, '환불 규정 알려줘', [5]) is None


@pytest.mark.asyncio
async def test_근거_집합이_다르면_miss(tenant_id, fake_embed):
    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '배송비 얼마예요', '3천원', [], [5, 7])
        await session.commit()
        # 부분집합·초집합 모두 miss — 문서 추가/제거가 답을 바꿀 수 있으므로
        assert await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [5]) is None
        assert await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [5, 7, 9]) is None


@pytest.mark.asyncio
async def test_테넌트_간_캐시_격리(tenant_id, fake_embed):
    import uuid
    cache = AnswerCache()
    other = str(uuid.uuid4())
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '배송비 얼마예요', 'A사 3천원', [], [5])
        await session.commit()
        try:
            # 같은 질의·같은 doc id라도 타 테넌트에선 절대 hit 금지 (캐시판 격리)
            assert await cache.get_semantic(session, other, '배송비 얼마예요', [5]) is None
        finally:
            pass  # other 테넌트엔 아무것도 안 만듦 — 정리 불필요


async def _prepared_with_conversation(session, tenant_id: str) -> PreparedRag:
    """save() 실행에 필요한 최소 컨텍스트 (대화 row + 정상 검색 상태)."""
    conv = Conversation(tenant_id=tenant_id)
    session.add(conv)
    await session.flush()
    return PreparedRag(
        conversation_id=conv.id,
        original_query='배송비 얼마예요',
        standalone_query='배송비 얼마예요',
        prior_turns=[],
        retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
        sources=[SourceCitation(document_id=5, filename='정책.pdf', version=1)],
        source_doc_ids=[5],
    )


@pytest.mark.asyncio
async def test_정상_답변은_캐시_저장(tenant_id, fake_embed):
    async with AsyncSessionLocal() as session:
        prepared = await _prepared_with_conversation(session, tenant_id)
        svc = RagService(tenant_id=tenant_id, session=session)
        await svc.save(prepared, '배송비는 3천원입니다. [정책.pdf v1]')
        await session.commit()

        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert len(rows) == 1                                # 양성 대조


@pytest.mark.asyncio
async def test_거절_답변은_캐시_제외(tenant_id, fake_embed):
    async with AsyncSessionLocal() as session:
        prepared = await _prepared_with_conversation(session, tenant_id)
        svc = RagService(tenant_id=tenant_id, session=session)
        # 모델이 거절 문구 앞뒤에 덧붙이는 변형까지 — in 비교 계약
        await svc.save(prepared, f'죄송합니다. {NO_EVIDENCE_ANSWER} 다른 질문을 주세요.')
        await session.commit()

        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []                                    # 문서 추가되면 답이 바뀌어야 하므로


@pytest.mark.asyncio
async def test_첨부_대화_답변은_공용_캐시_유출_금지(tenant_id, fake_embed):
    # should_cache 검증 — 고객 첨부 문서에 종속된 답변이 공용 캐시로 새면 타 상담 오염 (생존자 킬)
    async with AsyncSessionLocal() as session:
        prepared = await _prepared_with_conversation(session, tenant_id)
        prepared.attachments = [{'filename': '고객영수증.pdf', 'text': '개인 정보'}]
        svc = RagService(tenant_id=tenant_id, session=session)
        await svc.save(prepared, '첨부 기준으로 답변드립니다.')
        await session.commit()

        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_같은_질의_재저장은_upsert(tenant_id, fake_embed):
    # 문서 추가로 doc집합이 바뀌면 같은 cache_key로 재저장됨 — upsert가 없으면 unique 위반 500 (생존자 킬)
    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '배송비 얼마예요', '옛 답', [], [5])
        await session.commit()
        await cache.set(session, tenant_id, '배송비 얼마예요', '새 답', [], [5, 7])
        await session.commit()

        row = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalar_one()                                      # 1행 유지
        assert row.answer == '새 답'
        assert sorted(row.source_doc_ids) == [5, 7]


@pytest.mark.asyncio
async def test_무효화는_해당_문서_참조_행만(tenant_id, fake_embed):
    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '질의 하나', '답1', [], [5, 7])
        await cache.set(session, tenant_id, '질의 둘', '답2', [], [7])
        await cache.set(session, tenant_id, '질의 셋', '답3', [], [9])
        await session.commit()

        await cache.invalidate_document(session, tenant_id, 7)
        await session.commit()

        remain = (await session.execute(
            select(AnswerCacheRow.answer).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert remain == ['답3']                             # 7을 참조한 두 행만 제거
