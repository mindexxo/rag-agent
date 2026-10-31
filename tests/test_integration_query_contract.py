"""질의 API의 FE 계약 — 동시 요청 상한(429), SSE 봉투(이벤트 어휘·필드·헤더), 스트림 전 오류 응답.

HTTP로만 확인한다. 동시 요청은 가짜 LLM을 생성 도중 멈춰 슬롯을 실제로 쥔 상태를 만든다.
"""

import asyncio

import httpx
import pytest
from sqlalchemy import func, select

from database import AsyncSessionLocal
from rag.models import Message, TenantQuota
from tests.conftest import register_faq, sse_done, sse_events
from tests.polling import wait_inflight

EVENTS = {'meta', 'delta', 'ping', 'error', 'done'}
META_KEYS = {'conversation_id', 'assistant_message_id', 'cached', 'cache_kind', 'reason'}
DONE_KEYS = {'finish_reason', 'latency_ms', 'citations'}
BUSY = {'detail': '동시 요청이 많아 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.'}


async def _message_count(tenant: str) -> int:
    async with AsyncSessionLocal() as s:
        return (
            await s.execute(select(func.count()).select_from(Message).where(Message.tenant_id == tenant))
        ).scalar_one()


async def _set_quota(tenant: str, tenant_limit: int, user_limit: int) -> None:
    async with AsyncSessionLocal() as s:
        s.add(TenantQuota(tenant_id=tenant, concurrency_limit=tenant_limit, user_concurrency=user_limit))
        await s.commit()


async def _hold_slot(client, fake_llm, user: str) -> asyncio.Task:
    """생성 도중 멈춘 질의 하나를 띄워 슬롯을 쥐고 있게 한다. 풀 때는 _release."""
    fake_llm.pause_after_tokens = 2
    task = asyncio.create_task(
        client.post('/kms/query', json={'query': '환불 기간 알려줘'}, headers={'X-User-Id': user})
    )
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=10)
    return task


async def _release(fake_llm, task: asyncio.Task, tenant: str) -> None:
    fake_llm.pause_after_tokens = None
    fake_llm.resume.set()
    assert (await task).status_code == 200
    assert await wait_inflight(tenant, 0) == 0


@pytest.mark.asyncio
async def test_같은_사용자의_두번째_동시_요청은_429이고_행을_남기지_않는다(client, tenant_id, fake_llm, pass_gate):
    """QRY-52 — 사용자 상한 1. 거절은 턴을 열기 전이라 질문 행도 남지 않는다."""
    await register_faq(client)
    await _set_quota(tenant_id, tenant_limit=50, user_limit=1)  # 환경 기본값에 기대지 않는다
    held = await _hold_slot(client, fake_llm, 'agent-a')
    try:
        before = await _message_count(tenant_id)
        res = await client.post('/kms/query', json={'query': '또 질문'}, headers={'X-User-Id': 'agent-a'})
        assert res.status_code == 429
        assert res.json() == BUSY
        assert await _message_count(tenant_id) == before
    finally:
        await _release(fake_llm, held, tenant_id)

    again = await client.post('/kms/query', json={'query': '또 질문'}, headers={'X-User-Id': 'agent-a'})
    assert again.status_code == 200, '슬롯을 반납한 뒤에도 거절된다'
    assert await wait_inflight(tenant_id, 0) == 0  # 생성 경로의 반납은 백그라운드 — 정리 전에 기다린다


@pytest.mark.asyncio
async def test_테넌트_상한을_넘으면_다른_사용자도_429(client, tenant_id, fake_llm, pass_gate):
    """QRY-52 — 테넌트 상한은 사용자와 무관하게 센다. 상한 행이 기본값보다 우선한다."""
    await register_faq(client)
    await _set_quota(tenant_id, tenant_limit=1, user_limit=5)
    held = await _hold_slot(client, fake_llm, 'agent-a')
    try:
        before = await _message_count(tenant_id)
        res = await client.post('/kms/query', json={'query': '다른 사람 질문'}, headers={'X-User-Id': 'agent-b'})
        assert res.status_code == 429
        assert res.json() == BUSY
        assert await _message_count(tenant_id) == before
    finally:
        await _release(fake_llm, held, tenant_id)


def _assert_envelope(res: httpx.Response, *, reason: str) -> list[str]:
    assert res.status_code == 200
    assert res.headers['content-type'].startswith('text/event-stream')
    assert res.headers['cache-control'] == 'no-cache'
    assert res.headers['x-accel-buffering'] == 'no'

    events = sse_events(res)
    names = [e for e, _ in events]
    assert set(names) <= EVENTS, f'계약 밖 이벤트: {set(names) - EVENTS}'
    assert names[0] == 'meta' and names[-1] == 'done'
    assert names.count('meta') == 1 and names.count('done') == 1

    meta = events[0][1]
    assert set(meta) == META_KEYS
    assert meta['reason'] == reason
    assert isinstance(meta['conversation_id'], int) and isinstance(meta['assistant_message_id'], int)
    assert set(sse_done(res)) == DONE_KEYS
    return names


@pytest.mark.asyncio
async def test_SSE_계약_생성_경로(client, tenant_id, fake_llm, pass_gate):
    """QRY-69 — 생성 경로: meta 한 번 → delta 여러 번 → done 한 번, 프록시 버퍼링을 끄는 헤더."""
    await register_faq(client)
    names = _assert_envelope(await client.post('/kms/query', json={'query': '환불 기간 알려줘'}), reason='ok')
    assert names.count('delta') >= 2 and 'error' not in names


@pytest.mark.asyncio
async def test_SSE_계약_즉시_경로(client, tenant_id, fake_llm):
    """QRY-69 — 근거 없음 즉시 경로도 같은 봉투·헤더를 쓴다(FE가 경로를 구별하지 않는다)."""
    res = await client.post('/kms/query', json={'query': '아무 근거 없는 질문'})
    names = _assert_envelope(res, reason='no_evidence')
    assert names == ['meta', 'delta', 'done']
    assert sse_done(res)['citations'] == []


@pytest.mark.asyncio
async def test_SSE_계약_생성_실패는_error_다음_done으로_닫힌다(client, tenant_id, fake_llm, pass_gate):
    """QRY-69 — 실패 신호는 error, 종료 신호는 done. 예외 원문은 내보내지 않는다."""
    await register_faq(client)
    fake_llm.raise_after_tokens = 2
    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})
    names = _assert_envelope(res, reason='ok')

    assert names[-2:] == ['error', 'done']
    error = dict(sse_events(res))['error']
    assert error == {
        'code': 'generation_failed',
        'message': '답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.',
    }
    assert sse_done(res)['finish_reason'] == 'failed'


@pytest.mark.asyncio
async def test_스트림_전_오류는_SSE가_아니라_JSON(client, tenant_id, fake_llm):
    """QRY-69 — 404·422는 스트림을 열기 전에 끝나 일반 JSON 본문이다."""
    res = await client.post('/kms/query', json={'query': '질문', 'conversation_id': 999999999})
    assert res.status_code == 404
    assert res.headers['content-type'].startswith('application/json')
    assert res.json() == {'detail': '대화를 찾을 수 없습니다.'}

    res = await client.post('/kms/query', json={'query': ''})
    assert res.status_code == 422
    assert res.headers['content-type'].startswith('application/json')


@pytest.mark.asyncio
async def test_대화와_무관한_ValueError도_현재는_대화없음_404로_나간다(
    client, tenant_id, fake_llm, pass_gate, monkeypatch
):
    """QRY-42 현재 동작 재현 — 질의 준비 중 어떤 ValueError든 '대화를 찾을 수 없습니다' 404가 된다.

    진짜 대화 없음은 턴을 열기 전이라 행이 없지만, 여기서는 질문과 실패 처리된 답변 행이 남는다 — 같은 404가
    다른 두 사건을 덮는다는 증거다. 오류 매핑을 고칠 때 이 테스트의 기대값을 뒤집는다.
    """
    await register_faq(client)

    async def _broken_embed(texts: list[str], kind: str = 'document') -> list:
        raise ValueError('임베딩 응답 차원이 맞지 않는다')

    import rag.retriever

    monkeypatch.setattr(rag.retriever, 'embed_texts', _broken_embed)  # fake_embed와 같은 지점
    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})

    assert res.status_code == 404
    assert res.json() == {'detail': '대화를 찾을 수 없습니다.'}
    async with AsyncSessionLocal() as s:
        msgs = (
            (await s.execute(select(Message).where(Message.tenant_id == tenant_id).order_by(Message.id)))
            .scalars()
            .all()
        )
    assert [(m.role, m.status) for m in msgs] == [('user', 'done'), ('assistant', 'failed')]
