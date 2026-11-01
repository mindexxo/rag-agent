"""통계 API 경계 — days·limit 범위, 기간 시작 경계, 0 나누기, 끝나지 못한 턴 제외, 혼합 순위 상한, 시각 직렬화.

행은 DB에 직접 심는다(created_at을 정확히 맞추려고). 기간 경계는 라우터와 같은 KST 자정 식을 DB에서 계산해 쓴다.
"""

from datetime import timedelta

import pytest
from sqlalchemy import text

from database import AsyncSessionLocal
from rag.models import Conversation, Message
from schemas.common import KST


async def _turn(
    tenant: str,
    question: str,
    *,
    status: str = 'done',
    cited_docs: list | None = None,
    cited_faq_ids: list | None = None,
    created_at: object | None = None,
) -> None:
    async with AsyncSessionLocal() as s:
        conv = Conversation(tenant_id=tenant, created_by='test-user')
        s.add(conv)
        await s.flush()
        u = Message(tenant_id=tenant, conversation_id=conv.id, role='user', content=question)
        if created_at is not None:
            u.created_at = created_at
        s.add(u)
        await s.flush()
        a = Message(
            tenant_id=tenant,
            conversation_id=conv.id,
            role='assistant',
            content='답',
            status=status,
            intent='KNOWLEDGE',
            question_message_id=u.id,
        )
        # None은 대입하지 않는다 — JSONB 컬럼에 None을 넣으면 SQL NULL이 아니라 JSON null(스칼라)이 된다
        if cited_docs is not None:
            a.cited_docs = cited_docs
        if cited_faq_ids is not None:
            a.cited_faq_ids = cited_faq_ids
        if created_at is not None:
            a.created_at = created_at
        s.add(a)
        await s.commit()


async def _period_start(days: int) -> object:
    """라우터의 _SINCE_KST와 같은 식 — 오늘 포함 최근 days일의 KST 자정."""
    async with AsyncSessionLocal() as s:
        return (
            await s.execute(
                text(
                    "SELECT (((now() AT TIME ZONE :tz) - make_interval(days => :days))::date + 1)::timestamp AT TIME ZONE :tz"
                ),
                {'tz': KST.key, 'days': days},
            )
        ).scalar_one()


# ── 파라미터 범위 ──


@pytest.mark.asyncio
async def test_days는_1에서_90까지이고_기본은_7(client, tenant_id):
    """STAT-01·02."""
    for bad in (0, -1, 91, 'abc'):
        assert (await client.get('/kms/stats', params={'days': bad})).status_code == 422, bad
        assert (await client.get('/kms/stats/unanswered', params={'days': bad})).status_code == 422, bad
    for days in (1, 90):
        res = await client.get('/kms/stats', params={'days': days})
        assert res.status_code == 200 and res.json()['period_days'] == days
        assert len(res.json()['daily']) == days
    assert (await client.get('/kms/stats')).json()['period_days'] == 7


@pytest.mark.asyncio
async def test_미답변_목록은_최신순이고_limit은_1에서_200(client, tenant_id):
    """STAT-21 — 기본 50. 0·201은 422."""
    for i in range(3):
        await _turn(tenant_id, f'질문 {i}', cited_docs=[])
    items = (await client.get('/kms/stats/unanswered')).json()
    assert [x['question'] for x in items] == ['질문 2', '질문 1', '질문 0']
    assert len((await client.get('/kms/stats/unanswered', params={'limit': 1})).json()) == 1
    assert (await client.get('/kms/stats/unanswered', params={'limit': 200})).status_code == 200
    for bad in (0, 201):
        assert (await client.get('/kms/stats/unanswered', params={'limit': bad})).status_code == 422, bad


# ── 집계 규칙 ──


@pytest.mark.asyncio
async def test_기간_시작_1초_전_행은_세지_않는다(client, tenant_id):
    """STAT-07."""
    start = await _period_start(7)
    await _turn(tenant_id, '경계 전', cited_docs=['a.md'], created_at=start - timedelta(seconds=1))
    await _turn(tenant_id, '경계', cited_docs=['a.md'], created_at=start)
    body = (await client.get('/kms/stats', params={'days': 7})).json()
    assert body['questions'] == 1 and body['knowledge_done'] == 1


@pytest.mark.asyncio
async def test_지식_답변이_없으면_근거미확인율은_0(client, tenant_id):
    """STAT-13 — 0으로 나누지 않는다."""
    body = (await client.get('/kms/stats')).json()
    assert body['knowledge_done'] == 0 and body['ungrounded_rate'] == 0.0


@pytest.mark.asyncio
async def test_취소_실패_턴은_분모에서_빠지고_인용_NULL은_근거없음(client, tenant_id):
    """STAT-15."""
    await _turn(tenant_id, '취소', status='cancelled', cited_docs=[])
    await _turn(tenant_id, '실패', status='failed', cited_docs=[])
    await _turn(tenant_id, '인용 NULL', cited_docs=None, cited_faq_ids=None)
    await _turn(tenant_id, '근거 있음', cited_docs=['a.md'])
    body = (await client.get('/kms/stats')).json()
    assert body['questions'] == 4
    assert body['knowledge_done'] == 2 and body['ungrounded'] == 1 and body['ungrounded_rate'] == 0.5
    assert [x['question'] for x in (await client.get('/kms/stats/unanswered')).json()] == ['인용 NULL']


@pytest.mark.asyncio
async def test_인용_top은_문서와_FAQ를_합산해_10개까지(client, tenant_id):
    """STAT-17 — 같은 횟수끼리 순서는 비결정이라 집합과 상위만 본다."""
    faq_id = (await client.post('/kms/faqs', json={'question': '환불 기간은?', 'answer': '7일'})).json()['id']
    for _ in range(3):
        await _turn(tenant_id, 'FAQ 인용', cited_docs=[], cited_faq_ids=[faq_id])
    await _turn(tenant_id, '문서 인용', cited_docs=['환불정책.md', '환불정책.md'])
    for i in range(11):
        await _turn(tenant_id, f'잡문서 {i}', cited_docs=[f'문서{i}.md'])
    top = (await client.get('/kms/stats')).json()['top_documents']
    assert len(top) == 10
    assert top[0] == {'filename': 'FAQ: 환불 기간은?', 'citations': 3}
    assert top[1] == {'filename': '환불정책.md', 'citations': 2}


@pytest.mark.asyncio
async def test_미답변_시각은_KST_오프셋으로_나간다(client, tenant_id):
    """STAT-25."""
    await _turn(tenant_id, '질문', cited_docs=[])
    asked_at = (await client.get('/kms/stats/unanswered')).json()[0]['asked_at']
    assert asked_at.endswith('+09:00')
