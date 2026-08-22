"""생성 취소 계약 테스트 (#30).

검증하는 계약은 하나다 — **명시적 정지는 실제로 멈추고, 부분 답변을 cancelled로 남기고,
정리(리미터 반납·큐 sentinel·스팬 종료)를 온전히 끝낸다.** 연결 끊김(#26, 완주해서 저장)과
정반대 방향이라 그쪽 테스트(test_stream_disconnect.py)와 짝으로 읽어야 한다.

레이어를 셋으로 나눈다:
  순수      레지스트리 pop-then-cancel 규약 (asyncio만, DB·Redis 없음)
  단위      _run_generation을 직접 취소 — 취소 분기·정리 보장의 정본
  통합      HTTP 취소 엔드포인트 + 실제 Redis pub/sub 왕복

FakeLlm.pause_after_tokens로 정지 지점을 잡는다 — 시간(sleep) 기반이면 CI에서 플레이크가
난다. astream에 진짜 await가 없으면 취소가 전달될 지점 자체가 없다는 것도 여기서 전제한다.
"""
import asyncio
import contextlib

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from database import AsyncSessionLocal
from rag import cancellation, limiter, otel
from rag.models import Message
from rag.service import RagService
from rag.streaming import _run_generation, spawn_generation
from tests.conftest import register_faq

USER = {'X-User-Id': 'agent-x'}
OTHER_USER = {'X-User-Id': 'agent-other'}


@pytest_asyncio.fixture
async def cancel_subscriber():
    """취소 채널 구독 루프만 띄운다.

    httpx ASGITransport가 lifespan을 호출하지 않으므로(실측) main.py의 배선으로는 구독
    루프가 테스트에서 뜨지 않는다. 그래서 함수를 직접 기동한다.

    공용 Redis 클라이언트(clients.shared_redis)를 이 테스트의 루프에 맞는 새 인스턴스로
    교체한다. 앞선 테스트의 루프에 묶인 커넥션이 남아 있으면 구독 루프가 "Future attached to
    a different loop"로 죽고(실측), 그 커넥션은 루프가 이미 닫혀 disconnect()조차 "Event loop
    is closed"로 실패한다. 그래서 정리가 아니라 교체다 — _loop_hygiene가 http_async에 쓰는 방식.
    limiter·취소가 이 속성을 호출 시점에 읽으므로 교체가 양쪽에 그대로 먹는다.
    conftest에 두지 않는 이유: 이 파일만 쓰는 fixture다(리포 관례 — 단일 파일 전용은 로컬).
    """
    import redis.asyncio as aioredis

    from config import settings
    from rag import clients

    clients.shared_redis = aioredis.from_url(settings.redis_url)
    task = asyncio.create_task(cancellation.subscribe_forever())
    await asyncio.sleep(0.1)                  # 구독이 붙을 시간 — pub/sub은 늦게 붙으면 메시지를 놓친다
    yield task
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    await clients.shared_redis.aclose()
    clients.shared_redis = aioredis.from_url(settings.redis_url)   # lazy — 다음 사용자를 위해 성한 것으로


async def _inflight_count(tenant_id: str) -> int:
    return await limiter._redis().zcard(f'kms:inflight:t:{tenant_id}')


async def _prepare_turn(tenant_id: str):
    """정상 경로로 prepare + 자리표시(generating)까지 만들고 그 뒤를 테스트가 조작한다."""
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant_id, session=session, user_id='agent-x')
        prepared = await svc.prepare('환불 기간 알려줘')   # 자리표시는 prepare가 커밋한다 (#72)
        return prepared


async def _wait_status(assistant_id: int, expected: str, tries: int = 100) -> str:
    """백그라운드 태스크의 커밋을 기다린다 (기존 test_stream_disconnect와 같은 폴링 패턴)."""
    for _ in range(tries):
        async with AsyncSessionLocal() as session:
            msg = await session.get(Message, assistant_id)
            if msg is not None and msg.status == expected:
                return msg.status
        await asyncio.sleep(0.05)
    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        return msg.status if msg else 'MISSING'


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
    await asyncio.sleep(0)                            # 태스크가 await 지점까지 진행하게 —
    #                                                   시작 전에 취소하면 except 절에 못 들어간다
    cancellation.register(-1, task)
    assert cancellation.cancel_local(-1) is True      # 첫 요청이 태스크를 가져간다
    assert cancellation.cancel_local(-1) is False     # 두 번째는 대상이 없다
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert cancels == [1]                             # cancel은 정확히 한 번만 전달됐다


@pytest.mark.asyncio
async def test_unregister는_멱등():
    cancellation.unregister(-999)                     # 없는 키도 예외 없이
    task = asyncio.create_task(asyncio.sleep(0))
    cancellation.register(-2, task)
    cancellation.unregister(-2)
    cancellation.unregister(-2)
    assert cancellation.cancel_local(-2) is False
    await task


def test_잘못된_payload는_무시된다():
    cancellation._handle_signal(b'not-a-number')      # 예외 없이 통과해야 한다
    cancellation._handle_signal(None)


# ── 단위: _run_generation 취소 분기 ─────────────────────────

@pytest.mark.asyncio
async def test_취소하면_부분답변이_cancelled로_남고_정리가_끝난다(client, tenant_id, fake_llm, pass_gate):
    """이 기능의 정본 테스트 — 취소 분기가 없으면 generating으로 고착되고 스윕이 failed로 만든다."""
    await register_faq(client)
    fake_llm.pause_after_tokens = 2                   # 2토큰 내보낸 뒤 정지
    prepared = await _prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id

    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    assert await _inflight_count(tenant_id) == 1

    queue: asyncio.Queue = asyncio.Queue()
    root_span, token = otel.start_turn()
    task = asyncio.create_task(_run_generation(prepared, queue, lease, 0.0, root_span))
    try:
        await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)   # 정지 지점 도달까지
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        otel.detach_turn(token)

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'cancelled'
        assert msg.content                            # 부분 답변이 남았다
        assert msg.content in fake_llm.answer + ' '   # 실제로 생성된 접두부다
        assert msg.sources == [] and msg.cited_docs == []   # done이 아니면 인용을 남기지 않는다

    assert await _inflight_count(tenant_id) == 0      # 리미터 반납됨
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert drained[-1] is None                        # 큐 sentinel — FE 스트림이 done으로 끝난다
    assert cancellation.cancel_local(assistant_id) is False   # 레지스트리에 잔재 없음


@pytest.mark.asyncio
async def test_취소하면_done이_cancelled와_빈_인용을_싣는다(client, tenant_id, fake_llm, pass_gate):
    """#56 계약: 인용은 done에서만 확정된다 — 구 계약의 '낙관 전송 후 [] 정정' 자체가 사라졌고,
    취소 턴은 done.finish_reason='cancelled'·citations=[]로 닫힌다(저장 규칙과 같은 값)."""
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await _prepare_turn(tenant_id)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)

    queue: asyncio.Queue = asyncio.Queue()
    root_span, token = otel.start_turn()
    task = asyncio.create_task(_run_generation(prepared, queue, lease, 0.0, root_span))
    try:
        await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        otel.detach_turn(token)

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
    prepared = await _prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    lease.handed_off = True

    root_span, token = otel.start_turn()
    otel.detach_turn(token)
    spawn_generation(prepared, asyncio.Queue(), lease, 0.0, root_span)
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)

    async with AsyncSessionLocal() as session:      # 스윕이 지나간 상황을 만든다
        msg = await session.get(Message, assistant_id)
        msg.status = 'failed'
        await session.commit()

    res = await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)
    assert res.status_code == 204, 'DB가 failed라도 살아있는 태스크는 멈춰야 한다'
    assert await _wait_status(assistant_id, 'cancelled') == 'cancelled'


@pytest.mark.asyncio
async def test_발행_실패는_503(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """Redis 순단 시 202("접수했다")는 거짓이고 500은 원인을 감춘다 — 재시도 가능함을 알린다."""
    await register_faq(client)
    prepared = await _prepare_turn(tenant_id)       # generating, 로컬 태스크 없음

    async def _boom(_mid):
        raise ConnectionError('redis down')
    monkeypatch.setattr(cancellation, 'request_cancel', _boom)

    res = await client.post(f'/kms/messages/{prepared.assistant_message_id}/cancel', headers=USER)
    assert res.status_code == 503


@pytest.mark.asyncio
async def test_첫_토큰_전에_취소하면_빈_답변으로_남는다(client, tenant_id, fake_llm, pass_gate):
    await register_faq(client)
    fake_llm.pause_after_tokens = 0                   # 첫 토큰 전에 정지
    prepared = await _prepare_turn(tenant_id)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)

    root_span, token = otel.start_turn()
    task = asyncio.create_task(_run_generation(prepared, asyncio.Queue(), lease, 0.0, root_span))
    try:
        await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        otel.detach_turn(token)

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, prepared.assistant_message_id)
        assert msg.status == 'cancelled' and msg.content == ''
    assert await _inflight_count(tenant_id) == 0


@pytest.mark.asyncio
async def test_정리_도중_취소요청은_대상을_못_찾는다(client, tenant_id, fake_llm, pass_gate):
    """레이스 회귀 고정: 답변이 done으로 커밋된 뒤 finally에서 반납하는 찰나에 취소가 오면
    self-pop이 없을 때 정리가 파괴된다(실측). 자기를 먼저 빼두므로 요청은 빈손이어야 한다."""
    await register_faq(client)
    prepared = await _prepare_turn(tenant_id)
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)

    root_span, token = otel.start_turn()
    try:
        await _run_generation(prepared, asyncio.Queue(), lease, 0.0, root_span)   # 완주
    finally:
        otel.detach_turn(token)

    # 완주 직후 취소 요청이 도착한 상황 — 레지스트리에 없어야 한다
    assert cancellation.cancel_local(prepared.assistant_message_id) is False
    async with AsyncSessionLocal() as session:
        assert (await session.get(Message, prepared.assistant_message_id)).status == 'done'
    assert await _inflight_count(tenant_id) == 0       # 정리가 온전했다


# ── 통합: HTTP 엔드포인트 ───────────────────────────────────

@pytest.mark.asyncio
async def test_생성_중_취소요청은_204이고_cancelled로_귀결(client, tenant_id, fake_llm, pass_gate):
    """HTTP 취소 엔드포인트 → 레지스트리 → 태스크 → DB까지 한 줄로 검증.

    스트림을 열어둔 채 취소를 보내는 형태로는 쓸 수 없다 — httpx ASGITransport는 응답 본문이
    끝날 때까지 client.stream()을 반환하지 않아서(실측) 생성 중에 다른 요청을 보낼 수가 없다.
    실서버(uvicorn)는 점진 전송이라 FE의 동시 POST가 정상 동작하지만, 그 배선의 최종 확인은
    실기동 몫이다 — test_stream_disconnect.py가 같은 한계를 docstring에 적어둔 것과 같은 이유.
    그래서 여기서는 spawn_generation으로 실제 태스크를 띄워 레지스트리에 올린 뒤 HTTP로 취소한다.
    """
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await _prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    lease.handed_off = True                            # 반납 책임은 태스크가 진다 (kms 라우터와 동일)

    root_span, token = otel.start_turn()
    otel.detach_turn(token)                            # 요청 컨텍스트가 아니므로 즉시 분리
    spawn_generation(prepared, asyncio.Queue(), lease, 0.0, root_span)
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)

    cancel = await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)
    assert cancel.status_code == 204
    assert await _wait_status(assistant_id, 'cancelled') == 'cancelled'
    assert await _inflight_count(tenant_id) == 0        # 취소 경로도 슬롯을 반납한다

    # 멱등 — 두 번째 클릭도 성공으로 (결과가 같다)
    again = await client.post(f'/kms/messages/{assistant_id}/cancel', headers=USER)
    assert again.status_code == 204


@pytest.mark.asyncio
async def test_동시_상한이_가득_차도_취소는_동작한다(client, tenant_id, fake_llm, pass_gate):
    """concurrency_guard를 거치지 않는 이유 — 429가 나는 상황이 취소가 가장 필요한 때다."""
    await register_faq(client)
    prepared = await _prepare_turn(tenant_id)
    leases = [await limiter.try_acquire(tenant_id, 1, 'agent-x', 1)]
    assert leases[0] is not None
    assert await limiter.try_acquire(tenant_id, 1, 'agent-x', 1) is None   # 상한 포화 확인

    res = await client.post(f'/kms/messages/{prepared.assistant_message_id}/cancel', headers=USER)
    assert res.status_code != 429                       # 슬롯을 요구하지 않는다
    await limiter.release(leases[0])


@pytest.mark.asyncio
async def test_취소_대상이_아니면_404(client, tenant_id, fake_llm, pass_gate):
    await register_faq(client)
    res = await client.post('/kms/query', headers=USER, json={'query': '환불 기간 알려줘'})
    assert res.status_code == 200
    from tests.conftest import sse_meta
    assistant_id = sse_meta(res)['assistant_message_id']
    assert await _wait_status(assistant_id, 'done') == 'done'

    assert (await client.post(f'/kms/messages/{assistant_id}/cancel',
                              headers=USER)).status_code == 404      # 이미 끝난 턴
    assert (await client.post('/kms/messages/99999999/cancel',
                              headers=USER)).status_code == 404      # 없는 메시지
    assert (await client.post(f'/kms/messages/{assistant_id}/cancel',
                              headers={'X-User-Id': 'agent-other'})).status_code == 404   # 남의 대화


@pytest.mark.asyncio
async def test_진행_중인_남의_생성은_취소할_수_없다(client, tenant_id, other_tenant_id, fake_llm, pass_gate):
    """격리 회귀 고정 — 레지스트리는 message_id만 키로 쓰므로 소유 검증을 **먼저** 해야 한다.

    검증 순서가 뒤집히면(cancel_local을 먼저 부르면) messages.id가 전 테넌트 공용 시퀀스라
    id 추측만으로 남의 진행 중 생성을 죽일 수 있다. 404만 보는 게 아니라 **태스크가 살아
    있는지**까지 확인해야 이 결함을 잡는다.
    """
    await register_faq(client)
    fake_llm.pause_after_tokens = 2
    prepared = await _prepare_turn(tenant_id)
    assistant_id = prepared.assistant_message_id
    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    lease.handed_off = True

    root_span, token = otel.start_turn()
    otel.detach_turn(token)
    task = spawn_generation(prepared, asyncio.Queue(), lease, 0.0, root_span)
    await asyncio.wait_for(fake_llm.paused.wait(), timeout=5)

    # ① 같은 테넌트의 다른 상담원
    assert (await client.post(f'/kms/messages/{assistant_id}/cancel',
                              headers=OTHER_USER)).status_code == 404
    # ② 다른 테넌트 (헤더만 바꿔 붙은 클라이언트)
    from main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://testserver',
                           headers={'X-Tenant-Id': other_tenant_id}) as other:
        assert (await other.post(f'/kms/messages/{assistant_id}/cancel',
                                 headers=USER)).status_code == 404

    assert not task.cancelled() and not task.done(), '남의 요청이 생성을 죽였다 — 격리 위반'

    # 주인은 취소할 수 있다 (양성 대조)
    assert (await client.post(f'/kms/messages/{assistant_id}/cancel',
                              headers=USER)).status_code == 204
    assert await _wait_status(assistant_id, 'cancelled') == 'cancelled'


@pytest.mark.asyncio
async def test_로컬에_없으면_발행하고_202(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """다른 인스턴스 소유로 추정되는 경우 — generating인데 내 레지스트리에 없다."""
    await register_faq(client)
    prepared = await _prepare_turn(tenant_id)          # generating 상태, 태스크는 없음
    published = []
    monkeypatch.setattr(cancellation, 'request_cancel',
                        lambda mid: published.append(mid) or asyncio.sleep(0))

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
            raise ConnectionError('첫 연결 실패')     # 순단 재현
        return real_pubsub()
    monkeypatch.setattr(clients.shared_redis, 'pubsub', flaky_pubsub)

    loop_task = asyncio.create_task(cancellation.subscribe_forever())
    for _ in range(100):                              # 재연결로 구독이 붙을 때까지
        if len(attempts) >= 2:
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.2)                          # 구독 확립 여유

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
    await asyncio.sleep(0)                            # 태스크가 await 지점까지 진행하게
    cancellation.register(-42, task)
    await cancellation.request_cancel(-42)

    for _ in range(100):
        if cancelled:
            break
        await asyncio.sleep(0.05)
    with contextlib.suppress(asyncio.CancelledError):
        await task
    assert cancelled == [1], '구독 루프가 원격 취소 신호를 처리하지 못했다'
