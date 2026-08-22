"""클라이언트 조기 disconnect 계약 테스트 (#26).

이 기능의 존재 이유를 검증한다 — **연결이 끊겨도 생성 태스크는 완주하고, DB finalize와
리미터 슬롯 반납이 보장된다.** 이게 깨지면 이미 GPU를 쓴 답변이 버려지고(이력 구멍),
동시 상한이 유출된 슬롯에 잠식된다.

통합(HTTP 레이어에서 실제로 끊기)과 단위(소비자 없이 태스크만) 두 각도로 본다 —
ASGITransport의 중단 시맨틱이 실서버와 다를 수 있어 통합만으로는 계약을 못 박는다.
"""
import asyncio

import pytest
from sqlalchemy import select

from database import AsyncSessionLocal
from rag.models import Message
from rag.service import PreparedRag, RagService
from rag.streaming import _run_generation
from rag.retriever import RetrievalResult
from rag import limiter, otel
from tests.conftest import register_faq


async def _inflight_count(tenant_id: str) -> int:
    """리미터 ZSET에 남아 있는 in-flight 항목 수 (슬롯 유출 관측)."""
    return await limiter._redis().zcard(f'kms:inflight:t:{tenant_id}')


@pytest.mark.asyncio
async def test_소비자가_없어도_태스크는_완주해_finalize한다(client, tenant_id, fake_llm, pass_gate):
    """단위: 큐를 아무도 읽지 않는 상태(=연결 끊김과 동일)에서 _run_generation을 직접 돌린다.

    무한 큐라 put이 블록되지 않아 태스크가 끝까지 가고, finally에서 리미터 반납까지 해야 한다.
    """
    await register_faq(client)

    # prepare까지는 정상 경로로 만들고(자리표시 확보), 그 다음을 소비자 없이 돌린다
    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant_id, session=session, user_id='agent-x')
        prepared = await svc.prepare('환불 기간 알려줘')   # 자리표시는 prepare가 커밋한다 (#72)
        assistant_id = prepared.assistant_message_id
    assert assistant_id is not None

    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    assert lease is not None
    assert await _inflight_count(tenant_id) == 1

    queue: asyncio.Queue = asyncio.Queue()          # 아무도 get()하지 않는다
    root_span, token = otel.start_turn()
    try:
        await _run_generation(prepared, queue, lease, t_request=0.0, root_span=root_span)
    finally:
        otel.detach_turn(token)

    # 완주 증거 ① DB가 done으로 채워졌다
    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'done'
        assert '테스트 답변입니다.' in msg.content

    # 완주 증거 ② 리미터 슬롯이 반납됐다 (유출 없음)
    assert await _inflight_count(tenant_id) == 0

    # 완주 증거 ③ 읽지 않은 큐에도 토큰과 종료 sentinel이 다 들어와 있다
    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert drained[-1] is None                       # 리더 종료 sentinel
    assert any(item and item[0] == 'delta' for item in drained)


@pytest.mark.asyncio
async def test_캐시_저장이_실패해도_턴은_done이다(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """캐시 적재는 done 뒤의 부가물(#56 재배치) — 저장 실패(TEI 블립 등)가 완결된 턴을
    failed로 오염시키지 않아야 한다. 재배치 전엔 finalize 안에 있어서 턴 전체가 failed였다."""
    await register_faq(client)
    from rag import cache

    async def boom(*a, **kw):
        raise RuntimeError('TEI 블립 재현')

    monkeypatch.setattr(cache, 'save_answer', boom)

    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant_id, session=session, user_id='agent-x')
        prepared = await svc.prepare('환불 기간 알려줘')   # 자리표시는 prepare가 커밋한다 (#72)
        assistant_id = prepared.assistant_message_id

    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    queue: asyncio.Queue = asyncio.Queue()
    root_span, token = otel.start_turn()
    try:
        await _run_generation(prepared, queue, lease, t_request=0.0, root_span=root_span)
    finally:
        otel.detach_turn(token)

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'done', '캐시 저장 실패가 턴을 오염시켰다'

    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    done_events = [d for item in drained if item for k, d in [item] if k == 'done']
    assert done_events and done_events[0]['finish_reason'] == 'done'   # done도 정상 전송
    assert not any(item and item[0] == 'error' for item in drained)    # 에러 이벤트 없음


@pytest.mark.asyncio
async def test_생성_실패도_failed로_남고_정리가_끝난다(client, tenant_id, fake_llm, pass_gate, monkeypatch):
    """단위: 생성 도중 예외 → except Exception 분기(#54에서 스팬 기록 추가)가 통째로 도는지.

    이 분기는 다른 테스트가 아무도 안 탄다(리뷰 지적) — failed 기록·에러 이벤트·슬롯 반납·
    no-op 스팬 기록(_record_turn_result)이 예외 없이 지나가는 것까지 여기서 고정한다.
    """
    await register_faq(client)

    async with AsyncSessionLocal() as session:
        svc = RagService(tenant_id=tenant_id, session=session, user_id='agent-x')
        prepared = await svc.prepare('환불 기간 알려줘')   # 자리표시는 prepare가 커밋한다 (#72)
        assistant_id = prepared.assistant_message_id

    async def _boom(self, prepared):
        raise RuntimeError('LLM down')
        yield   # async generator로 만들기 위한 도달 불가 yield

    monkeypatch.setattr(RagService, 'generate', _boom)

    lease = await limiter.try_acquire(tenant_id, 10, 'agent-x', 10)
    queue: asyncio.Queue = asyncio.Queue()
    root_span, token = otel.start_turn()
    try:
        await _run_generation(prepared, queue, lease, t_request=0.0, root_span=root_span)
    finally:
        otel.detach_turn(token)

    # 실패 기록: _finalize_out_of_band가 새 세션으로 failed를 남긴다 (latency는 규칙상 NULL)
    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'failed'
        assert msg.latency_ms is None

    assert await _inflight_count(tenant_id) == 0     # 슬롯 반납 (finally)

    drained = []
    while not queue.empty():
        drained.append(queue.get_nowait())
    assert drained[-1] is None                       # 리더 종료 sentinel
    assert any(item and item[0] == 'error' for item in drained)   # 에러 이벤트 전달


@pytest.mark.asyncio
async def test_스트림_중단해도_턴이_done으로_남는다(client, tenant_id, fake_llm, pass_gate):
    """통합: HTTP 레이어에서 meta만 읽고 응답을 닫는다 → 태스크는 계속 돌아 finalize해야 한다."""
    await register_faq(client)

    assistant_id = None
    async with client.stream('POST', '/kms/query', json={'query': '환불 기간 알려줘'}) as res:
        assert res.status_code == 200
        async for raw in res.aiter_lines():          # meta 한 줄만 확인하고 즉시 이탈
            if raw.startswith('data: '):
                import json
                assistant_id = json.loads(raw.removeprefix('data: '))['assistant_message_id']
                break
    assert assistant_id is not None                  # placeholder는 스트림 전에 commit됨

    # 태스크는 응답과 무관하게 돌고 있다 — 완료까지 폴링 (고정 sleep은 느리고 불안정)
    for _ in range(100):
        async with AsyncSessionLocal() as session:
            msg = await session.get(Message, assistant_id)
            if msg.status != 'generating':
                break
        await asyncio.sleep(0.05)

    async with AsyncSessionLocal() as session:
        msg = await session.get(Message, assistant_id)
        assert msg.status == 'done', '연결이 끊기면 생성이 중단됐다 — 태스크 분리가 깨졌다'
        assert '테스트 답변입니다.' in msg.content

    assert await _inflight_count(tenant_id) == 0     # 슬롯도 반납됨
