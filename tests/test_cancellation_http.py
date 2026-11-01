"""생성 취소의 HTTP 엔드포인트와 인스턴스 간 pub/sub 배선 — 소유 검증·멱등·상한 무관·원격 신호·구독 재연결.

취소 분기 자체(부분 답변 저장·정리)와 레지스트리 규약은 tests/test_cancellation.py가 본다.
"""

import asyncio
import contextlib

import pytest
import pytest_asyncio

from rag import limiter, otel
from rag.turn import cancellation
from rag.turn.streaming import spawn_generation
from tests.conftest import register_faq
from tests.helpers_turn import prepare_turn
from tests.polling import wait_inflight, wait_message_status
from tests.tenant_client import client_for

USER = {'X-User-Id': 'agent-x'}
OTHER_USER = {'X-User-Id': 'agent-other'}


@pytest_asyncio.fixture
async def cancel_subscriber(fresh_redis):
    """취소 채널 구독 루프만 띄운다 — 공용 Redis는 fresh_redis가 이 테스트 루프용으로 교체해 둔다.

    httpx ASGITransport가 lifespan을 호출하지 않으므로 main.py의 배선으로는 구독
    루프가 테스트에서 뜨지 않는다. 그래서 함수를 직접 기동한다.
    """
    task = asyncio.create_task(cancellation.subscribe_forever())
    await asyncio.sleep(0.1)  # 구독이 붙을 시간 — pub/sub은 늦게 붙으면 메시지를 놓친다
    yield task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


# ── 통합: HTTP 엔드포인트 ───────────────────────────────────


@pytest.mark.asyncio
async def test_생성_중_취소요청은_204이고_cancelled로_귀결(client, tenant_id, fake_llm, pass_gate):
    """HTTP 취소 엔드포인트 → 레지스트리 → 태스크 → DB까지 한 줄로 검증.

    스트림을 열어둔 채 취소를 보내는 형태로는 쓸 수 없다 — httpx ASGITransport는 응답 본문이
    끝날 때까지 client.stream()을 반환하지 않아서 생성 중에 다른 요청을 보낼 수가 없다.
    실서버(uvicorn)는 점진 전송이라 FE의 동시 POST가 정상 동작하지만, 그 배선의 최종 확인은
    실기동 몫이다 — test_stream_disconnect.py가 같은 한계를 docstring에 적어둔 것과 같은 이유.
    그래서 여기서는 spawn_generation으로 실제 태스크를 띄워 레지스트리에 올린 뒤 HTTP로 취소한다.
    """
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    lease.handed_off = True  # 반납 책임은 태스크가 진다 (kms 라우터와 동일)

    turn = otel.start_turn()
    otel.detach_turn(turn.token)  # 요청 컨텍스트가 아니므로 즉시 분리
    spawn_generation(prepared, asyncio.Queue(), lease, 0.0, turn.span)
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)

    cancel = await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)
    assert cancel.status_code == 204
    assert await wait_message_status(assistant_id, 'cancelled') == 'cancelled'
    assert await wait_inflight(tenant_id) == 0  # 취소 경로도 슬롯을 반납한다

    # 멱등 — 두 번째 클릭도 성공으로 (결과가 같다)
    again = await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)
    assert again.status_code == 204


@pytest.mark.asyncio
async def test_동시_상한이_가득_차도_취소는_동작한다(client, tenant_id, fake_llm, pass_gate):
    """concurrency_guard를 거치지 않는 이유 — 429가 나는 상황이 취소가 가장 필요한 때다."""
    await register_faq(client)
    prepared = await prepare_turn(tenant_id)
    leases = [await limiter.try_acquire(tenant_id, 1, 'agent-x', 1)]
    assert leases[0] is not None
    assert await limiter.try_acquire(tenant_id, 1, 'agent-x', 1) is None  # 상한 포화 확인

    res = await client.post(f'/kms/messages/{prepared.assistant_message_id}/cancel', headers=USER)
    assert res.status_code != 429  # 슬롯을 요구하지 않는다
    await limiter.release(leases[0])


@pytest.mark.asyncio
async def test_취소_대상이_아니면_404(client, tenant_id, fake_llm, pass_gate):
    await register_faq(client)
    res = await client.post('/kms/query', headers=USER, json={'query': '환불 기간 알려줘'})
    assert res.status_code == 200
    from tests.conftest import sse_meta

    assistant_id = sse_meta(res)['assistant_message_id']
    assert await wait_message_status(assistant_id, 'done') == 'done'

    assert (await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)).status_code == 404  # 이미 끝난 턴
    assert (await client.post('/kms/messages/99999999/cancel', headers=USER)).status_code == 404  # 없는 메시지
    assert (
        await client.post(f'/kms/messages/{assistant_id}/cancel', headers={'X-User-Id': 'agent-other'})
    ).status_code == 404  # 남의 대화


@pytest.mark.asyncio
async def test_진행_중인_남의_생성은_취소할_수_없다(client, tenant_id, other_tenant_id, fake_llm, pass_gate):
    """격리 회귀 고정 — 레지스트리는 message_id만 키로 쓰므로 소유 검증을 **먼저** 해야 한다.

    검증 순서가 뒤집히면(cancel_local을 먼저 부르면) messages.id가 전 테넌트 공용 시퀀스라
    id 추측만으로 남의 진행 중 생성을 죽일 수 있다. 404만 보는 게 아니라 **태스크가 살아
    있는지**까지 확인해야 이 결함을 잡는다.
    """
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    lease.handed_off = True

    turn = otel.start_turn()
    otel.detach_turn(turn.token)
    task = spawn_generation(prepared, asyncio.Queue(), lease, 0.0, turn.span)
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)

    # ① 같은 테넌트의 다른 상담원
    assert (await client.post(f'/kms/messages/{assistant_id}/cancel', headers=OTHER_USER)).status_code == 404
    # ② 다른 테넌트 (헤더만 바꿔 붙은 클라이언트)
    async with client_for(other_tenant_id) as other:
        assert (await other.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)).status_code == 404

    assert not task.cancelled() and not task.done(), '남의 요청이 생성을 죽였다 — 격리 위반'

    # 주인은 취소할 수 있다 (양성 대조)
    assert (await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)).status_code == 204
    assert await wait_message_status(assistant_id, 'cancelled') == 'cancelled'


@pytest.mark.asyncio
async def test_로컬에_없으면_발행하고_202(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """다른 인스턴스 소유로 추정되는 경우 — generating인데 내 레지스트리에 없다."""
    await register_faq(client)
    prepared = await prepare_turn(tenant_id)  # generating 상태, 태스크는 없음
    published = []
    monkeypatch.setattr(cancellation, 'request_cancel', lambda mid: published.append(mid) or asyncio.sleep(0))

    res = await client.post(f'/kms/messages/{prepared.assistant_message_id}/cancel', headers=USER)
    assert res.status_code == 202
    assert published == [prepared.assistant_message_id]


@pytest.mark.asyncio
async def test_구독이_끊겨도_재연결해_계속_받는다(monkeypatch):
    """포기하면 그 프로세스는 재기동 전까지 원격 취소를 전부 놓치고, 그게 조용한 장애다.

    첫 연결을 실패시킨 뒤 신호가 여전히 도달하는지 본다 — 재시도가 없으면 이 테스트는 멈춘다.
    """
    import redis.asyncio as aioredis

    from config import settings
    from rag import clients

    monkeypatch.setattr(cancellation, 'SUBSCRIBE_RETRY_MIN_SECONDS', 0.05)
    clients.shared_redis = aioredis.from_url(settings.redis_url)
    real_pubsub, attempts = clients.shared_redis.pubsub, []

    def flaky_pubsub():
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionError('첫 연결 실패')  # 순단 재현
        return real_pubsub()

    monkeypatch.setattr(clients.shared_redis, 'pubsub', flaky_pubsub)

    loop_task = asyncio.create_task(cancellation.subscribe_forever())
    for _ in range(100):  # 재연결로 구독이 붙을 때까지
        if len(attempts) >= 2:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)  # 구독 확립 여유

    cancelled = []

    async def dummy():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    task = asyncio.create_task(dummy())
    await asyncio.sleep(0)
    cancellation.register(-77, task)
    await cancellation.request_cancel(-77)
    for _ in range(100):
        if cancelled:
            break
        await asyncio.sleep(0.05)

    loop_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await loop_task
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await clients.shared_redis.aclose()
    clients.shared_redis = aioredis.from_url(settings.redis_url)

    assert len(attempts) >= 2, '끊긴 뒤 재연결을 시도하지 않았다'
    assert cancelled == [1], '재연결 후에도 신호를 받지 못했다'


@pytest.mark.asyncio
async def test_원격_신호가_구독으로_전달돼_취소된다(cancel_subscriber):
    """발행 → 구독 → 로컬 취소 배선. lifespan 없이 구독 루프를 직접 띄워 검증한다."""
    cancelled = []

    async def dummy():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(1)
            raise

    task = asyncio.create_task(dummy())
    await asyncio.sleep(0)  # 태스크가 await 지점까지 진행하게
    cancellation.register(-42, task)
    await cancellation.request_cancel(-42)

    for _ in range(100):
        if cancelled:
            break
        await asyncio.sleep(0.05)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert cancelled == [1], '구독 루프가 원격 취소 신호를 처리하지 못했다'
