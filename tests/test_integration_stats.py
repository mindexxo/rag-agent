"""운영 리포트 MVP 통합 테스트 — 지표 씨앗 저장 + 집계 API 정확성.

실제 쿼리 흐름(fake_llm)으로 데이터를 만들고 /kms/stats가 맞게 세는지 검증.
"""
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Message


@pytest.fixture
def pass_gate(monkeypatch):
    import rag.retriever as rt
    monkeypatch.setattr(rt, 'apply_gate', lambda cands, max_dense_distance=0.6: (False, None))


async def _ask(client, query: str, user: str | None = None):
    headers = {'X-User-Id': user} if user else {}
    return await client.post('/kms/query', json={'query': query}, headers=headers)


@pytest.mark.asyncio
async def test_씨앗_저장_user_id_latency_cache_kind(client, tenant_id, fake_llm, pass_gate):
    await client.post('/kms/faqs', json={'question': '환불 기간은?', 'variants': [], 'answer': '7일'})
    await _ask(client, '환불 기간 알려줘', user='agent-kim')          # 생성 경로
    await _ask(client, '환불 기간 알려줘', user='agent-lee')          # semantic 캐시 히트 (즉시 경로)

    async with AsyncSessionLocal() as session:
        msgs = (await session.execute(
            select(Message).where(Message.tenant_id == tenant_id).order_by(Message.id)
        )).scalars().all()
        users = [m for m in msgs if m.role == 'user']
        answers = [m for m in msgs if m.role == 'assistant']

        assert [u.user_id for u in users] == ['agent-kim', 'agent-lee']   # 씨앗 1
        assert all(a.latency_ms is not None and a.latency_ms >= 0 for a in answers)  # 씨앗 2
        assert answers[0].cache_kind is None                              # 신규 생성
        assert answers[1].cache_kind == 'semantic'                        # 캐시 재생 표시
        # 저장 시 확정된 사실: 짝 FK(생성·즉시 경로 모두) + 거절 아님 플래그
        assert [a.question_message_id for a in answers] == [u.id for u in users]
        assert all(a.is_refusal is False for a in answers)


@pytest.mark.asyncio
async def test_stats_집계_정확성(client, tenant_id, fake_llm, pass_gate):
    await client.post('/kms/faqs', json={'question': '환불 기간은?', 'variants': [], 'answer': '7일'})
    fake_llm.answer = '7일 이내 처리됩니다. [FAQ]'              # 실인용 라벨 포함
    await _ask(client, '환불 기간 알려줘', user='agent-kim')     # 답변 1 (생성)
    await _ask(client, '환불 기간 알려줘', user='agent-kim')     # 답변 2 (캐시)
    # 거절 유도: pass_gate를 우회할 수 없는 다른 질의? — 게이트가 항상 통과라 거절은
    # LLM 자체 거절로 만든다
    fake_llm.answer = '해당 내용은 제공된 문서에서 확인할 수 없습니다.'
    await _ask(client, '주차장 있나요?', user='agent-park')      # 답변 3 (거절)

    res = await client.get('/kms/stats?days=1')
    assert res.status_code == 200
    body = res.json()

    assert body['questions'] == 3
    assert body['active_users'] == 2                             # kim, park
    assert body['refusals'] == 1
    assert body['blocked'] == 0 and body['failed'] == 0
    assert body['avg_latency_ms'] is not None
    assert body['daily'] and sum(d['questions'] for d in body['daily']) == 3
    assert body['top_documents'][0]['filename'] == 'FAQ'         # 실인용 기준 top
    assert body['top_documents'][0]['citations'] == 2             # 생성 1 + 캐시 재생 1 (미인용은 미집계)


@pytest.mark.asyncio
async def test_unanswered_지식갭_목록(client, tenant_id, fake_llm):
    # 게이트 미우회 + 문서 없음 → 자연 거절 경로
    await _ask(client, '오프라인 매장이 있나요?', user='agent-kim')
    await _ask(client, '해외 배송 되나요?', user='agent-kim')

    res = await client.get('/kms/stats/unanswered?days=1')
    assert res.status_code == 200
    questions = [r['question'] for r in res.json()]
    assert questions == ['해외 배송 되나요?', '오프라인 매장이 있나요?']   # 최신순


@pytest.mark.asyncio
async def test_stats_테넌트_격리(client, tenant_id, other_tenant_id, fake_llm):
    await _ask(client, '질문 하나')                              # tenant A에 거절 데이터
    async with AsyncClient(transport=ASGITransport(app=__import__('main').app),
                           base_url='http://testserver',
                           headers={'X-Tenant-Id': other_tenant_id}) as other:
        body = (await other.get('/kms/stats?days=1')).json()
        assert body['questions'] == 0                            # 타 테넌트 수치 미노출
        assert (await other.get('/kms/stats/unanswered?days=1')).json() == []
