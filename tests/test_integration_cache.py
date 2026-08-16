"""D-3: semantic 캐시 동작 통합 테스트.

hit/miss 판정(유사도·doc집합), 테넌트 격리, 거절 답변 캐시 제외, 무효화 정확성.
exact(Redis) 계층은 제거됨 — semantic(PG) 단일 계층 기준.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.cache import AnswerCache
from rag.models import AnswerCache as AnswerCacheRow, Conversation
from rag.prompt_texts import NO_EVIDENCE_ANSWER
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


# ── #16 캐시 경화: FAQ 낙관적 검증 + fail-open ──────────────────

@pytest.mark.asyncio
async def test_생성_중_FAQ_수정되면_캐시_저장_스킵(tenant_id, fake_embed):
    """write-back 레이스(#16): prepare 스냅샷 이후 FAQ가 바뀌면 set이 저장을 스킵.
    FAQ는 id 불변이라 doc집합 비교가 자가치유 못 하는 유일한 출처 — 이 검증이 마지막 방어선."""
    from rag.cache import snapshot_faq_versions
    from rag.models import Faq

    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        faq = Faq(tenant_id=tenant_id, question='반품 기간?', answer='14일')
        session.add(faq)
        await session.commit()

        snap = await snapshot_faq_versions(session, tenant_id, [-faq.id])
        assert snap                                          # 스냅샷에 FAQ 잡힘

        # 생성 구간에 끼어든 FAQ 수정 (별도 세션 = 별도 트랜잭션의 커밋)
        async with AsyncSessionLocal() as s2:
            row = (await s2.execute(select(Faq).where(Faq.id == faq.id))).scalar_one()
            row.answer = '7일'
            await s2.commit()

        await cache.set(session, tenant_id, '반품 기간 알려줘', '14일입니다', [], [-faq.id],
                        faq_versions=snap)
        await session.commit()
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []                                    # 옛 내용 기반 답변 저장 안 됨


@pytest.mark.asyncio
async def test_FAQ_변경_없으면_스냅샷_검증_통과_저장(tenant_id, fake_embed):
    from rag.cache import snapshot_faq_versions
    from rag.models import Faq

    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        faq = Faq(tenant_id=tenant_id, question='반품 기간?', answer='14일')
        session.add(faq)
        await session.commit()

        snap = await snapshot_faq_versions(session, tenant_id, [-faq.id])
        await cache.set(session, tenant_id, '반품 기간 알려줘', '14일입니다', [], [-faq.id],
                        faq_versions=snap)
        await session.commit()
        row = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalar_one()
        assert row.answer == '14일입니다'


@pytest.mark.asyncio
async def test_캐시_조회_저장_실패는_요청을_죽이지_않는다(tenant_id, monkeypatch):
    """fail-open(#16): 임베딩(TEI) 실패 시 조회는 miss, 저장은 스킵 — 예외 전파 금지."""
    async def boom(text):
        raise RuntimeError('TEI down')
    monkeypatch.setattr('rag.cache.embed_query', boom)

    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        assert await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [5]) is None
        await cache.set(session, tenant_id, '배송비 얼마예요', '3천원', [], [5])   # 예외 없이 통과
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_보존기간_지난_미히트_캐시만_청소(tenant_id, fake_embed):
    """sweep_stale(#16): last_hit_at이 cache_retention_days를 넘긴 row만 삭제."""
    from sqlalchemy import func, update as sa_update

    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        await cache.set(session, tenant_id, '오래된 질문', '옛 답', [], [5])
        await cache.set(session, tenant_id, '최근 질문', '새 답', [], [7])
        await session.commit()
        # 한 행을 보존기간(90일) 밖으로 백데이트
        await session.execute(
            sa_update(AnswerCacheRow)
            .where(AnswerCacheRow.tenant_id == tenant_id)
            .where(AnswerCacheRow.query_text == '오래된 질문')
            .values(last_hit_at=func.now() - func.make_interval(0, 0, 0, 91)))

        deleted = await cache.sweep_stale(session)
        await session.commit()

        assert deleted >= 1                                  # 전 테넌트 일괄이라 정확 수 대신 하한
        remain = (await session.execute(
            select(AnswerCacheRow.answer).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert remain == ['새 답']                            # 이 테넌트에선 미히트 옛 행만 제거


@pytest.mark.asyncio
async def test_FAQ_스냅샷_검증도_테넌트_격리(tenant_id, other_tenant_id, fake_embed):
    """snapshot_faq_versions/_faqs_unchanged는 타 테넌트의 같은 faq_id를 보면 안 된다
    (WHERE-clause 격리 전략 — 새 테넌트 스코프 쿼리 경로마다 통합 테스트가 계약)."""
    from rag.cache import snapshot_faq_versions
    from rag.models import Faq

    cache = AnswerCache()
    async with AsyncSessionLocal() as session:
        faq = Faq(tenant_id=other_tenant_id, question='반품 기간?', answer='14일')
        session.add(faq)
        await session.commit()

        # 타 테넌트 FAQ id로는 스냅샷이 비어야 한다
        assert await snapshot_faq_versions(session, tenant_id, [-faq.id]) == {}

        # 빈 스냅샷을 기준으로 한 검증은 '변경됨' 판정 → 저장 스킵 (보수적 안전)
        await cache.set(session, tenant_id, '반품 기간 알려줘', '14일입니다', [], [-faq.id],
                        faq_versions={})
        await session.commit()
        rows = (await session.execute(
            select(AnswerCacheRow).where(AnswerCacheRow.tenant_id == tenant_id)
        )).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_주입상한0이어도_신규첨부가_있으면_캐시를_우회한다(
        client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """#36 버그 수정 — 캐시 가드는 주입용·신규분 첨부를 **둘 다** 봐야 한다.

    max_attachments<=0이면 주입용(attachment_dicts)이 강제로 비므로, 신규분을 함께 보지 않으면
    첨부가 있는 턴이 캐시 가드를 통과한다. 그러면 캐시 답변이 재생되고 캐시-히트 PreparedRag가
    new_attachments를 안 채워, save()에서 **이번 턴 첨부가 조용히 유실**된다.
    """
    from config import settings
    from rag.models import Message
    from tests.conftest import register_faq, sse_meta

    await register_faq(client)
    # 1턴: 캐시를 채운다 (첨부 없음)
    first = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})
    assert sse_meta(first)['cache_kind'] is None

    monkeypatch.setattr(settings, 'max_attachments', 0)   # 주입 상한 0 — 주입용은 항상 빈다

    # 2턴: 같은 질의 + 이번 턴 첨부 → 캐시를 타면 첨부가 사라진다
    second = await client.post('/kms/query', json={
        'query': '환불 기간 알려줘',
        'attachments': [{'filename': '계약서.txt', 'text': '특약: 환불 30일'}],
    })
    assert sse_meta(second)['cache_kind'] is None, '첨부가 있는데 캐시가 히트했다 (#36)'

    async with AsyncSessionLocal() as session:
        user_msg = (await session.execute(
            select(Message).where(Message.tenant_id == tenant_id)
            .where(Message.role == 'user').order_by(Message.id.desc())
        )).scalars().first()
    assert user_msg.attachments, '이번 턴 첨부가 저장되지 않았다 — 데이터 유실 (#36)'
    assert user_msg.attachments[0]['filename'] == '계약서.txt'
