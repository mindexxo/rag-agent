"""FAQ·카테고리 입력 경계 — 상한 정확값, 빈 값·누락, 중복 판정 조합, 없는 대상 404, 목록 파라미터, 카테고리 이름 규칙.

HTTP로만 확인한다. 상한 상수는 스키마에서 가져와 경계 양쪽(정확값 통과·+1 거절)을 같이 본다.
"""

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Faq, SearchIndexOutbox
from schemas.kms import (
    DOC_LIST_MAX_LIMIT,
    FAQ_ANSWER_MAX,
    FAQ_CATEGORY_NAME_MAX,
    FAQ_QUESTION_MAX,
    FAQ_VARIANT_MAX,
    FAQ_VARIANTS_MAX,
)

NOT_FOUND = {'detail': 'FAQ not found'}


async def _faq(client, question: str = '환불 기간은?', **fields: object) -> dict:
    res = await client.post('/kms/faqs', json={'question': question, 'answer': '7일', **fields})
    assert res.status_code == 200, res.text
    return res.json()


def _locs(res) -> list[tuple]:
    return [tuple(e['loc']) for e in res.json()['detail']]


async def _outbox_count(tenant: str) -> int:
    async with AsyncSessionLocal() as s:
        return len((await s.execute(select(SearchIndexOutbox).where(SearchIndexOutbox.tenant_id == tenant))).all())


# ── 상한 정확값 ──


@pytest.mark.asyncio
async def test_질문_답변_유사질문_상한은_정확값까지_받는다(client, tenant_id):
    """FAQ-C02·C04·C05 — 단위 테스트가 스키마를 고정하고, 여기서는 HTTP 왕복과 저장값."""
    body = {
        'question': 'ㄱ' * FAQ_QUESTION_MAX,
        'answer': 'ㄴ' * FAQ_ANSWER_MAX,
        'variants': [f'변형 {i}' for i in range(FAQ_VARIANTS_MAX)],
    }
    res = await client.post('/kms/faqs', json=body)
    assert res.status_code == 200, res.text
    saved = res.json()
    assert len(saved['question']) == FAQ_QUESTION_MAX and len(saved['answer']) == FAQ_ANSWER_MAX
    assert len(saved['variants']) == FAQ_VARIANTS_MAX


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('field', 'value', 'loc'),
    [
        ('answer', 'ㄴ' * (FAQ_ANSWER_MAX + 1), ('body', 'answer')),
        ('variants', [f'변형 {i}' for i in range(FAQ_VARIANTS_MAX + 1)], ('body', 'variants')),
        ('variants', ['정상', 'ㄱ' * (FAQ_VARIANT_MAX + 1)], ('body', 'variants', 1)),
    ],
    ids=['answer-501', 'variants-11', 'variant-item-101'],
)
async def test_상한을_넘으면_422이고_원인_위치가_보인다(client, tenant_id, field, value, loc):
    """FAQ-C04·C05·C06."""
    res = await client.post('/kms/faqs', json={'question': '환불 기간은?', 'answer': '7일', field: value})
    assert res.status_code == 422 and loc in _locs(res)


@pytest.mark.asyncio
async def test_길이는_공백을_뗀_뒤_재고_빈_유사질문은_개수에서_뺀다(client, tenant_id):
    """FAQ-C07·C08 — 공백 포함 원문은 상한을 넘지만 strip 뒤 정확히 상한이면 통과, 저장도 strip 값."""
    padded = '  ' + 'ㄱ' * FAQ_QUESTION_MAX + '  '
    variants = [f'변형 {i}' for i in range(FAQ_VARIANTS_MAX)] + ['', '   ']
    saved = await _faq(client, padded, variants=variants)
    assert saved['question'] == 'ㄱ' * FAQ_QUESTION_MAX
    assert saved['variants'] == [f'변형 {i}' for i in range(FAQ_VARIANTS_MAX)]


# ── 빈 값·누락·없는 대상 ──


@pytest.mark.asyncio
async def test_등록의_빈_질문_답변과_누락(client, tenant_id):
    """FAQ-C09·C10 — 공백뿐인 값은 strip 뒤 빈 값이라 라우터 문구로 422, 필드 누락은 스키마 422."""
    for body in ({'question': '   ', 'answer': '7일'}, {'question': '환불?', 'answer': ''}):
        res = await client.post('/kms/faqs', json=body)
        assert res.status_code == 422 and res.json() == {'detail': '질문/답변이 비어 있습니다.'}, body
    res = await client.post('/kms/faqs', json={'answer': '7일'})
    assert res.status_code == 422 and ('body', 'question') in _locs(res)


@pytest.mark.asyncio
async def test_수정의_빈_질문_답변은_각각의_문구로_422(client, tenant_id):
    """FAQ-U04."""
    faq_id = (await _faq(client))['id']
    res = await client.patch(f'/kms/faqs/{faq_id}', json={'question': '  '})
    assert res.status_code == 422 and res.json() == {'detail': '질문은 비울 수 없습니다.'}
    res = await client.patch(f'/kms/faqs/{faq_id}', json={'answer': ''})
    assert res.status_code == 422 and res.json() == {'detail': '답변은 비울 수 없습니다.'}


@pytest.mark.asyncio
async def test_없는_카테고리_등록_없는_FAQ_수정_재삭제는_404(client, tenant_id):
    """FAQ-C20·U20·D02."""
    res = await client.post('/kms/faqs', json={'question': '환불?', 'answer': '7일', 'category_id': 999999999})
    assert res.status_code == 404 and res.json() == {'detail': 'category not found'}
    res = await client.patch('/kms/faqs/999999999', json={'answer': '8일'})
    assert res.status_code == 404 and res.json() == NOT_FOUND

    faq_id = (await _faq(client))['id']
    assert (await client.delete(f'/kms/faqs/{faq_id}')).status_code == 204
    res = await client.delete(f'/kms/faqs/{faq_id}')
    assert res.status_code == 404 and res.json() == NOT_FOUND


@pytest.mark.asyncio
async def test_빈_바디_수정은_변경자만_덮고_대기열에_남기지_않는다(client, tenant_id):
    """FAQ-U17 — 헤더가 없으면 updated_by는 NULL로 덮인다."""
    faq_id = (
        await client.post('/kms/faqs', json={'question': '환불?', 'answer': '7일'}, headers={'X-User-Id': 'agent-a'})
    ).json()['id']
    before = await _outbox_count(tenant_id)
    res = await client.patch(f'/kms/faqs/{faq_id}', json={})
    assert res.status_code == 200
    assert res.json()['updated_by'] is None and res.json()['created_by'] == 'agent-a'
    assert await _outbox_count(tenant_id) == before


# ── 중복 판정 ──


@pytest.mark.asyncio
async def test_유사질문이_다른_FAQ의_유사질문과_같으면_409(client, tenant_id):
    """FAQ-C13 — 대표↔대표·대표↔유사는 기존 테스트가 고정."""
    first = await _faq(client, '환불 기간은?', variants=['돈 언제 돌려받아요'])
    res = await client.post(
        '/kms/faqs', json={'question': '반품 기간은?', 'answer': '14일', 'variants': ['돈 언제 돌려받아요']}
    )
    assert res.status_code == 409
    assert res.json() == {'detail': f"같은 질문의 FAQ(#{first['id']})가 이미 있습니다: '돈 언제 돌려받아요'"}


@pytest.mark.asyncio
async def test_대소문자만_다른_질문은_다른_질문이다(client, tenant_id):
    """FAQ-C15 — 완전 일치만 중복으로 본다."""
    await _faq(client, 'Refund period?')
    assert (await client.post('/kms/faqs', json={'question': 'refund period?', 'answer': '7일'})).status_code == 200
    async with AsyncSessionLocal() as s:
        assert len((await s.execute(select(Faq).where(Faq.tenant_id == tenant_id))).all()) == 2


# ── 목록 ──


@pytest.mark.asyncio
async def test_목록_파라미터_보정(client, tenant_id):
    """FAQ-L03 — limit 0·201은 상한(200)으로, offset 음수는 0으로, 범위 밖 offset은 빈 목록."""
    for i in range(3):
        await _faq(client, f'질문 {i}?')
    for limit in (0, DOC_LIST_MAX_LIMIT + 1):
        assert len((await client.get('/kms/faqs', params={'limit': limit})).json()['items']) == 3, limit
    assert len((await client.get('/kms/faqs', params={'offset': -1})).json()['items']) == 3
    page = (await client.get('/kms/faqs', params={'offset': 10})).json()
    assert page['items'] == [] and page['has_more'] is False and page['total'] == 3


@pytest.mark.asyncio
async def test_q_검색은_와일드카드를_글자로_보고_공백뿐이면_필터가_아니다(client, tenant_id):
    """FAQ-L06 — '%'는 기존 테스트가, 여기서는 '_'와 공백뿐인 q."""
    await _faq(client, '쿠폰_코드 사용법?')
    await _faq(client, '쿠폰X코드 사용법?')
    hits = (await client.get('/kms/faqs', params={'q': '쿠폰_코드'})).json()['items']
    assert [f['question'] for f in hits] == ['쿠폰_코드 사용법?']
    assert (await client.get('/kms/faqs', params={'q': '   '})).json()['total'] == 2


@pytest.mark.asyncio
async def test_category_id_필터의_없는_id와_음수는_빈_목록(client, tenant_id):
    """FAQ-L09 — N·0은 기존 테스트가 고정. 없는 id는 404가 아니라 빈 목록."""
    await _faq(client)
    for cid in (999999999, -1):
        res = await client.get('/kms/faqs', params={'category_id': cid})
        assert res.status_code == 200 and res.json()['items'] == [], cid


# ── 카테고리 ──


@pytest.mark.asyncio
async def test_카테고리_이름_50자는_받고_자기_이름_재설정도_200(client, tenant_id):
    """CAT-02·CAT-09 — 공백·51자 거절은 기존 테스트가 고정."""
    name = 'ㄱ' * FAQ_CATEGORY_NAME_MAX
    res = await client.post('/kms/faq-categories', json={'name': name})
    assert res.status_code == 200
    cid = res.json()['id']
    assert (await client.patch(f'/kms/faq-categories/{cid}', json={'name': name})).status_code == 200


@pytest.mark.asyncio
async def test_다른_테넌트는_같은_카테고리_이름을_쓸_수_있다(client, tenant_id, other_tenant_id):
    """CAT-05."""
    assert (await client.post('/kms/faq-categories', json={'name': '배송'})).status_code == 200
    from main import app

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url='http://testserver', headers={'X-Tenant-Id': other_tenant_id}
    ) as other:
        assert (await other.post('/kms/faq-categories', json={'name': '배송'})).status_code == 200


@pytest.mark.asyncio
async def test_비활성_FAQ만_든_카테고리도_삭제를_막는다(client, tenant_id):
    """CAT-11 — 활성 FAQ는 기존 테스트가 고정."""
    cid = (await client.post('/kms/faq-categories', json={'name': '배송'})).json()['id']
    faq_id = (await _faq(client, category_id=cid))['id']
    assert (await client.patch(f'/kms/faqs/{faq_id}', json={'is_active': False})).status_code == 200
    res = await client.delete(f'/kms/faq-categories/{cid}')
    assert res.status_code == 409
    assert res.json() == {
        'detail': '카테고리에 FAQ 1개가 있어 삭제할 수 없습니다. FAQ를 옮기거나 삭제한 뒤 다시 시도해 주세요.'
    }
