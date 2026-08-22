"""SSE 큐 리더 단위 테스트 — streaming.queue_reader.

백그라운드 생성(_run_generation)이 큐에 넣은 이벤트를 SSE로 흘리는 코어.
LLM·DB 없이 큐에 이벤트를 직접 넣어 순서·형식을 검증한다.
#56 계약: 리더는 meta만 자체 생성하고 나머지는 순수 전달 — done도 큐로 온다(리더가 합성하지 않음).
"""
import asyncio

import pytest

from config import settings
from rag.retriever import RetrievalResult
from rag.service import PreparedRag
from rag.streaming import queue_reader
from tests.conftest import sse_events


def _prepared(**kw) -> PreparedRag:
    base = dict(conversation_id=1, assistant_message_id=1, original_query='q', standalone_query='q',
                prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
                sources=[], source_doc_ids=[])
    return PreparedRag(**{**base, **kw})


async def _collect(prepared, queue) -> list[tuple[str, object]]:
    """queue_reader가 yield한 SSE 청크들을 (event, data) 리스트로."""
    return sse_events(''.join([chunk async for chunk in queue_reader(prepared, queue)]))


_DONE = {'finish_reason': 'done', 'latency_ms': 10, 'citations': []}


@pytest.mark.asyncio
async def test_이벤트_순서_meta_delta_done():
    queue = asyncio.Queue()
    await queue.put(('delta', {'text': '안녕'}))
    await queue.put(('delta', {'text': '하세요'}))
    await queue.put(('done', _DONE))
    await queue.put(None)                                   # 생성 태스크의 종료 sentinel

    events = await _collect(_prepared(), queue)
    assert [e for e, _ in events] == ['meta', 'delta', 'delta', 'done']
    assert events[1][1] == {'text': '안녕'}


@pytest.mark.asyncio
async def test_done은_리더가_합성하지_않고_큐의_페이로드를_그대로_전달():
    # 구 계약에선 리더가 빈 done을 하드코딩으로 붙였다 — 이제 최종 상태는 생성 태스크만 안다 (#56)
    queue = asyncio.Queue()
    payload = {'finish_reason': 'cancelled', 'latency_ms': 77,
               'citations': [{'document_id': 5, 'filename': '정책.pdf', 'version': 2}]}
    await queue.put(('done', payload))
    await queue.put(None)

    events = await _collect(_prepared(), queue)
    assert events == [('meta', events[0][1]), ('done', payload)]


@pytest.mark.asyncio
async def test_done_없이_sentinel만_오면_done_없이_닫힘():
    # 비정상 종료(태스크 급사 등) — 리더가 done을 지어내면 FE가 실패를 정상으로 오인한다
    queue = asyncio.Queue()
    await queue.put(None)
    events = await _collect(_prepared(), queue)
    assert [e for e, _ in events] == ['meta']


@pytest.mark.asyncio
async def test_에러_이벤트도_통과_후_done으로_닫힘():
    queue = asyncio.Queue()
    await queue.put(('error', {'code': 'generation_failed', 'message': 'x'}))
    await queue.put(('done', {'finish_reason': 'failed', 'latency_ms': None, 'citations': []}))
    await queue.put(None)

    events = await _collect(_prepared(), queue)
    assert [e for e, _ in events] == ['meta', 'error', 'done']
    assert dict(events)['done']['finish_reason'] == 'failed'


@pytest.mark.asyncio
async def test_소비_시작_후_도착하는_이벤트도_대기해_수신():
    # 리더는 큐가 비어 있으면 대기(blocking get)해야 함 — get_nowait로 바뀌면 프로덕션에서 스트림 즉사
    queue = asyncio.Queue()

    async def late_producer():
        await asyncio.sleep(0.05)
        await queue.put(('delta', {'text': '늦게 도착'}))
        await queue.put(('done', _DONE))
        await queue.put(None)

    task = asyncio.create_task(late_producer())
    events = await _collect(_prepared(), queue)
    await task
    assert [e for e, _ in events] == ['meta', 'delta', 'done']


@pytest.mark.asyncio
async def test_큐_get_취소_직전에_들어온_아이템은_유실되지_않는다():
    """ping 구현(wait_for가 get을 취소)의 전제를 결정적으로 재현 — put으로 깨어난 getter를
    루프가 돌기 전에 취소해도 아이템은 큐에 남아 다음 get이 집는다 (타이밍 의존 없음)."""
    import contextlib
    queue: asyncio.Queue = asyncio.Queue()
    getter = asyncio.ensure_future(queue.get())
    await asyncio.sleep(0)                     # getter가 대기에 진입
    queue.put_nowait(('delta', {'text': 'x'}))  # getter 깨우기 예약
    getter.cancel()                            # 깨어나기 전에 취소 — 정확히 그 경합
    with contextlib.suppress(asyncio.CancelledError):
        await getter
    assert queue.get_nowait() == ('delta', {'text': 'x'})   # 유실 없음


@pytest.mark.asyncio
async def test_유휴_시_ping이_나오고_이후_이벤트는_유실_없이_도착(monkeypatch):
    """ping 타임아웃(wait_for)이 큐 대기를 취소해도 아이템이 유실되지 않는다 (#56 회귀 고정).

    시간 sleep 최소화: 주기를 0.05초로 줄이고 생산자는 그보다 늦게 넣는다.
    """
    monkeypatch.setattr(settings, 'sse_ping_interval_seconds', 0.05)
    queue = asyncio.Queue()

    async def late_producer():
        await asyncio.sleep(0.12)                            # ping 최소 1회 발생할 시간
        await queue.put(('delta', {'text': '유실되면 안 됨'}))
        await queue.put(('done', _DONE))
        await queue.put(None)

    task = asyncio.create_task(late_producer())
    events = await _collect(_prepared(), queue)
    await task
    names = [e for e, _ in events]
    assert names.count('ping') >= 1                          # 유휴 구간에 ping이 나왔다
    # ping을 걷어내면 정확히 원래 시퀀스 — 아이템 유실·중복 없음
    assert [e for e in names if e != 'ping'] == ['meta', 'delta', 'done']
    assert dict(events)['delta'] == {'text': '유실되면 안 됨'}
