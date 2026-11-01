"""생성 취소 계약 — 명시적 정지는 실제로 멈추고, 부분 답변을 cancelled로 남기고, 정리(리미터 반납·큐 sentinel·스팬 종료)를 끝낸다.

연결 끊김(완주해서 저장, test_stream_disconnect.py)과 정반대 방향이다. 정지 지점은 시간이 아니라
FakeLlm.pause_after_tokens로 잡는다 — sleep 기반이면 플레이크가 난다.
HTTP 엔드포인트·pub/sub 배선은 tests/test_cancellation_http.py가 본다.
"""

import asyncio
import contextlib

import pytest

from database import AsyncSessionLocal
from rag import limiter, otel
from rag.models import Message
from rag.turn import cancellation
from rag.turn.streaming import _run_generation, spawn_generation
from tests.conftest import register_faq
from tests.helpers_turn import prepare_turn
from tests.polling import inflight_count, wait_message_status

USER = {'X-User-Id': 'agent-x'}


# ── 순수: 레지스트리 규약 ────────────────────────────────────


@pytest.mark.asyncio
async def test_pop_then_cancel은_두번_눌러도_한번만_취소한다():
    """따닥 두 번의 방어선. 두 번째 cancel()이 정리 중인 finally를 파괴하는 걸 막는다."""
    cancels = []

    async def dummy():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancels.append(1)
            raise

    task = asyncio.create_task(dummy())
    await asyncio.sleep(0)  # 태스크가 await 지점까지 진행하게 —
    #                                                   시작 전에 취소하면 except 절에 못 들어간다
    cancellation.register(-1, task)
    assert cancellation.cancel_local(-1) is True  # 첫 요청이 태스크를 가져간다
    assert cancellation.cancel_local(-1) is False  # 두 번째는 대상이 없다
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert cancels == [1]  # cancel은 정확히 한 번만 전달됐다


@pytest.mark.asyncio
async def test_unregister는_멱등():
    cancellation.unregister(-999)  # 없는 키도 예외 없이
    task = asyncio.create_task(asyncio.sleep(0))
    cancellation.register(-2, task)
    cancellation.unregister(-2)
    cancellation.unregister(-2)
    assert cancellation.cancel_local(-2) is False
    await task


def test_잘못된_payload는_무시된다():
    cancellation._handle_signal(b'not-a-number')  # 예외 없이 통과해야 한다
    cancellation._handle_signal(None)


# ── 단위: _run_generation 취소 분기 ─────────────────────────


@pytest.mark.asyncio
async def test_취소하면_부분답변이_cancelled로_남고_정리가_끝난다(client, tenant_id, fake_llm, pass_gate):
    """이 기능의 정본 테스트 — 취소 분기가 없으면 generating으로 고착되고 스윕이 failed로 만든다."""
    await register_faq(client)
    fake_llm.pause_after_tokens = 2  # 2토큰 내보낸 뒤 정지
    prepared = await prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id

    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    assert await inflight_count(tenant_id) == 1

    queue: asyncio.Queue = asyncio.Queue()
    turn = otel.start_turn()
    task = asyncio.create_task(_run_generation(prepared, queue, lease, 0.0, turn.span))
    try:
        await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)  # 정지 지점 도달까지
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        otel.detach_turn(turn.token)

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'cancelled'
        assert msg.content  # 부분 답변이 남았다
        assert msg.content in fake_llm.answer + ' '  # 실제로 생성된 접두부다
        assert msg.sources == [] and msg.cited_docs == []  # done이 아니면 인용을 남기지 않는다

    assert await inflight_count(tenant_id) == 0  # 리미터 반납됨
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert drained[-1] is None  # 큐 sentinel — FE 스트림이 done으로 끝난다
    assert cancellation.cancel_local(assistant_id) is False  # 레지스트리에 잔재 없음


@pytest.mark.asyncio
async def test_취소하면_done이_cancelled와_빈_인용을_싣는다(client, tenant_id, fake_llm, pass_gate):
    """인용은 done에서만 확정된다 — 취소 턴은 done.finish_reason='cancelled'·citations=[]로 닫힌다(저장 규칙과 같은 값)."""
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await prepare_turn(tenant_id)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)

    queue: asyncio.Queue = asyncio.Queue()
    turn = otel.start_turn()
    task = asyncio.create_task(_run_generation(prepared, queue, lease, 0.0, turn.span))
    try:
        await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        otel.detach_turn(turn.token)

    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    done_events = [payload for item in drained if item for kind, payload in [item] if kind == 'done']
    assert len(done_events) == 1, '취소 턴도 done으로 닫혀야 한다'
    assert done_events[0]['finish_reason'] == 'cancelled'
    assert done_events[0]['citations'] == []
    async with AsyncSessionLocal() as session:
        assert (await session.get(Message, prepared.assistant_message_id)).sources == []


@pytest.mark.asyncio
async def test_스윕이_failed로_바꿔도_살아있는_태스크는_취소된다(client, tenant_id, fake_llm, pass_gate):
    """300초 스윕은 '정말 진행 중'과 '고착'을 구분하지 못해 살아있는 생성도 failed로 바꾼다.
    상태만 믿으면 그 순간부터 정지 버튼이 무력해지므로, 태스크가 손에 있으면 멈춘다."""
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    lease.handed_off = True

    turn = otel.start_turn()
    otel.detach_turn(turn.token)
    spawn_generation(prepared, asyncio.Queue(), lease, 0.0, turn.span)
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)

    async with AsyncSessionLocal() as session:  # 스윕이 지나간 상황을 만든다
        msg = await session.get(Message, assistant_id)
        msg.status = 'failed'
        await session.commit()

    res = await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)
    assert res.status_code == 204, 'DB가 failed라도 살아있는 태스크는 멈춰야 한다'
    assert await wait_message_status(assistant_id, 'cancelled') == 'cancelled'


@pytest.mark.asyncio
async def test_발행_실패는_503(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """Redis 순단 시 202("접수했다")는 거짓이고 500은 원인을 감춘다 — 재시도 가능함을 알린다."""
    await register_faq(client)
    prepared = await prepare_turn(tenant_id)  # generating, 로컬 태스크 없음

    async def _boom(_mid):
        raise ConnectionError('redis down')

    monkeypatch.setattr(cancellation, 'request_cancel', _boom)

    res = await client.post(f'/kms/messages/{prepared.assistant_message_id}/cancel', headers=USER)
    assert res.status_code == 503


@pytest.mark.asyncio
async def test_첫_토큰_전에_취소하면_빈_답변으로_남는다(client, tenant_id, fake_llm, pass_gate):
    await register_faq(client)
    fake_llm.pause_after_tokens = 0  # 첫 토큰 전에 정지
    prepared = await prepare_turn(tenant_id)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)

    turn = otel.start_turn()
    task = asyncio.create_task(_run_generation(prepared, asyncio.Queue(), lease, 0.0, turn.span))
    try:
        await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        otel.detach_turn(turn.token)

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, prepared.assistant_message_id)
        assert msg.status == 'cancelled' and msg.content == ''
    assert await inflight_count(tenant_id) == 0


@pytest.mark.asyncio
async def test_정리_도중_취소요청은_대상을_못_찾는다(client, tenant_id, fake_llm, pass_gate):
    """레이스 회귀 고정: 답변이 done으로 커밋된 뒤 finally에서 반납하는 찰나에 취소가 오면
    self-pop이 없으면 정리가 파괴된다. 자기를 먼저 빼두므로 요청은 빈손이어야 한다."""
    await register_faq(client)
    prepared = await prepare_turn(tenant_id)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)

    turn = otel.start_turn()
    try:
        await _run_generation(prepared, asyncio.Queue(), lease, 0.0, turn.span)  # 완주
    finally:
        otel.detach_turn(turn.token)

    # 완주 직후 취소 요청이 도착한 상황 — 레지스트리에 없어야 한다
    assert cancellation.cancel_local(prepared.assistant_message_id) is False
    async with AsyncSessionLocal() as session:
        assert (await session.get(Message, prepared.assistant_message_id)).status == 'done'
    assert await inflight_count(tenant_id) == 0  # 정리가 온전했다
