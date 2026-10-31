"""D-2: FAQ 수명주기 통합 테스트.

등록→수정(재임베딩·캐시 무효화)→is_active 토글(검색 제외·복귀·캐시 자가치유)→삭제(cascade).
리뷰 발견(P1-15 PATCH 빈 문자열, P1-16 is_active 캐시 빈틈)의 회귀 방어를 겸한다.
"""
import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from tests.conftest import FakeLlm, faq_doc, sync_faq
from rag import cache
from rag.models import AnswerCache as AnswerCacheRow, Faq
from rag.search.retriever import retrieve_candidates


async def _create_faq(client, question='환불 기간은?', answer='7일 이내 처리됩니다.') -> int:
    """등록 + 대기열 drain — 라우터는 색인을 직접 하지 않으므로(#139) 검색 상태를 보려면 drain이 필요하다."""
    res = await client.post('/kms/faqs', json={
        'question': question, 'variants': ['돈 언제 돌려받아요'], 'answer': answer,
    })
    assert res.status_code == 200, res.text
    faq_id = res.json()['id']
    await sync_faq(faq_id)
    return faq_id


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

    assert await sync_faq(faq_id) == {'done': 1, 'failed': 0}   # INDEX_FAQ 재색인 행 하나
    chunk = await faq_doc(faq_id)                                 # id가 결정적이라 늘어날 수 없다 — 덮어쓴다
    assert chunk is not None and chunk['text'].startswith('Q: 환불은 며칠 걸리나요?')


@pytest.mark.asyncio
async def test_variants만_수정해도_재임베딩(client, tenant_id):
    # variants는 dense 매칭을 견인하는 핵심 콘텐츠 — variants-only PATCH가
    # content_changed로 인정돼 청크 텍스트에 반영돼야 한다 (뮤테이션 생존자 M5·M8 킬)
    faq_id = await _create_faq(client)

    res = await client.patch(f'/kms/faqs/{faq_id}', json={'variants': ['새로 추가한 구어체 질문']})
    assert res.status_code == 200

    await sync_faq(faq_id)
    chunk = await faq_doc(faq_id)
    assert chunk is not None and '(유사 질문: 새로 추가한 구어체 질문)' in chunk['text']


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
    assert faq_id in await _candidate_faq_ids(tenant_id)      # drain 전 — 제품 가이드대로 아직 보인다(최대 1~5분)
    await sync_faq(faq_id)                                    # META_FAQS 부분 갱신
    assert faq_id not in await _candidate_faq_ids(tenant_id)  # off → 검색 제외

    await client.patch(f'/kms/faqs/{faq_id}', json={'is_active': True})
    await sync_faq(faq_id)
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

        # 같은 근거 집합이면 hit (전제 확인 — 판정 승인 없이는 히트 없음 #113)
        hit = await cache.get_semantic(session, tenant_id, '배송비 얼마예요', [123],
                                       llm=FakeLlm())
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

    await sync_faq(faq_id)                                   # DROP_FAQS
    assert await faq_doc(faq_id) is None                     # 색인에서 제거됨
    async with AsyncSessionLocal() as session:
        assert (await session.get(Faq, faq_id)) is None
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


# ---------- #276 입력 상한·중복 거절 ----------

@pytest.mark.asyncio
async def test_상한_초과는_422(client, tenant_id):
    res = await client.post('/kms/faqs', json={'question': '가' * 101, 'answer': 'a'})
    assert res.status_code == 422 and res.json()['detail'][0]['loc'] == ['body', 'question'], res.text
    faq_id = await _create_faq(client)
    res = await client.patch(f'/kms/faqs/{faq_id}', json={'answer': '가' * 501})
    assert res.status_code == 422 and res.json()['detail'][0]['loc'] == ['body', 'answer'], res.text


@pytest.mark.asyncio
async def test_같은_대표_질문은_409에_기존_id가_보인다(client, tenant_id):
    faq_id = await _create_faq(client, question='환불 기간은?')
    res = await client.post('/kms/faqs', json={'question': '  환불 기간은?  ', 'answer': '다른 답'})
    assert res.status_code == 409, res.text
    assert f'#{faq_id}' in res.json()['detail'] and '환불 기간은?' in res.json()['detail']


@pytest.mark.asyncio
async def test_다른_FAQ의_유사_질문과_같아도_409(client, tenant_id):
    faq_id = await _create_faq(client)                       # variants=['돈 언제 돌려받아요']
    # 내 대표 질문 = 남의 유사 질문
    res = await client.post('/kms/faqs', json={'question': '돈 언제 돌려받아요', 'answer': 'a'})
    assert res.status_code == 409 and f'#{faq_id}' in res.json()['detail'], res.text
    # 내 유사 질문 = 남의 대표 질문
    res = await client.post('/kms/faqs', json={'question': '새 질문', 'variants': ['환불 기간은?'], 'answer': 'a'})
    assert res.status_code == 409 and f'#{faq_id}' in res.json()['detail'], res.text


@pytest.mark.asyncio
async def test_느슨한_일치는_막지_않는다(client, tenant_id):
    """완전 일치만 — 띄어쓰기·대소문자가 다르면 정당한 변형으로 본다."""
    await _create_faq(client, question='환불 기간은?')
    res = await client.post('/kms/faqs', json={'question': '환불기간은?', 'answer': 'a'})
    assert res.status_code == 200, res.text


@pytest.mark.asyncio
async def test_수정시_자기_자신은_중복이_아니고_남과_겹치면_409(client, tenant_id):
    a = await _create_faq(client, question='환불 기간은?')              # variants=['돈 언제 돌려받아요']
    res = await client.post('/kms/faqs', json={'question': '배송 기간은?', 'variants': ['택배 며칠'], 'answer': 'a'})
    assert res.status_code == 200, res.text
    b = res.json()['id']
    res = await client.patch(f'/kms/faqs/{a}', json={'question': '환불 기간은?', 'answer': '바뀐 답'})   # 자기 질문 그대로
    assert res.status_code == 200, res.text
    res = await client.patch(f'/kms/faqs/{b}', json={'variants': ['환불 기간은?']})
    assert res.status_code == 409 and f'#{a}' in res.json()['detail'], res.text
    async with AsyncSessionLocal() as session:
        row = await session.get(Faq, b)
        assert row.variants == ['택배 며칠']                   # 409면 저장 안 됨


@pytest.mark.asyncio
async def test_타_테넌트의_같은_질문은_허용(client, tenant_id, other_tenant_id):
    await _create_faq(client, question='환불 기간은?')
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'answer': 'a'},
                            headers={'X-Tenant-Id': other_tenant_id})
    assert res.status_code == 200, res.text


@pytest.mark.asyncio
async def test_앱_검사를_지나친_경쟁은_DB_UNIQUE가_409로(client, tenant_id, monkeypatch):
    """검사와 커밋 사이에 같은 질문이 들어온 경우 — 검사를 비활성화해 그 경로만 재현."""
    import routers.faqs as faqs_router
    await _create_faq(client, question='환불 기간은?')
    async def _noop(*a, **k): ...
    monkeypatch.setattr(faqs_router, '_reject_duplicate', _noop)
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'answer': 'a'})
    assert res.status_code == 409 and '환불 기간은?' in res.json()['detail'], res.text


@pytest.mark.asyncio
async def test_PATCH로_대표_질문을_남의_것과_같게_바꾸면_409(client, tenant_id):
    """리뷰 지적 1 — 대입을 먼저 하면 autoflush UPDATE가 DB UNIQUE에 걸려 500이었다."""
    a = await _create_faq(client, question='환불 기간은?')
    res = await client.post('/kms/faqs', json={'question': '배송 기간은?', 'answer': 'a'})
    b = res.json()['id']
    res = await client.patch(f'/kms/faqs/{b}', json={'question': '환불 기간은?'})
    assert res.status_code == 409 and f'#{a}' in res.json()['detail'], res.text
    async with AsyncSessionLocal() as session:
        assert (await session.get(Faq, b)).question == '배송 기간은?'


@pytest.mark.asyncio
async def test_PATCH_경쟁도_DB_UNIQUE가_409로(client, tenant_id, monkeypatch):
    """리뷰 지적 2 — rollback 뒤 만료된 faq.question을 읽으면 500이었다."""
    import routers.faqs as faqs_router
    await _create_faq(client, question='환불 기간은?')
    res = await client.post('/kms/faqs', json={'question': '배송 기간은?', 'answer': 'a'})
    b = res.json()['id']
    async def _noop(*a, **k): ...
    monkeypatch.setattr(faqs_router, '_reject_duplicate', _noop)
    res = await client.patch(f'/kms/faqs/{b}', json={'question': '환불 기간은?'})
    assert res.status_code == 409 and '환불 기간은?' in res.json()['detail'], res.text
