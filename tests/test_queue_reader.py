"""SSE 큐 리더 단위 테스트 — kms._queue_reader.

백그라운드 생성(_run_generation)이 큐에 넣은 이벤트를 SSE로 흘리는 코어.
LLM·DB 없이 큐에 이벤트를 직접 넣어 순서·형식을 검증한다.
"""
import asyncio
import json

import pytest

from rag.retriever import RetrievalResult
from rag.service import PreparedRag
from routers.kms import _queue_reader
from schemas.kms import SourceCitation


def _prepared(**kw) -> PreparedRag:
    base = dict(conversation_id=1, original_query='q', standalone_query='q',
                prior_turns=[], retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None),
                sources=[], source_doc_ids=[])
    return PreparedRag(**{**base, **kw})


def _events(sse_chunks: list[str]) -> list[tuple[str, object]]:
    """SSE 문자열 리스트 → (event, data) 튜플로 파싱."""
    out = []
    for raw in sse_chunks:
        lines = raw.strip().splitlines()
        event = lines[0].removeprefix('event: ')
        data = json.loads(lines[1].removeprefix('data: '))
        out.append((event, data))
    return out


async def _collect(prepared, no_evidence, queue) -> list[tuple[str, object]]:
    return _events([chunk async for chunk in _queue_reader(prepared, no_evidence, queue)])


@pytest.mark.asyncio
async def test_이벤트_순서_meta_sources_token_done():
    queue = asyncio.Queue()
    await queue.put(('token', {'text': '안녕'}))
    await queue.put(('token', {'text': '하세요'}))
    await queue.put(None)                                   # 생성 태스크의 종료 sentinel

    events = await _collect(_prepared(), False, queue)
    assert [e for e, _ in events] == ['meta', 'sources', 'token', 'token', 'done']
    assert events[2][1] == {'text': '안녕'}


@pytest.mark.asyncio
async def test_에러_이벤트도_통과_후_done으로_닫힘():
    queue = asyncio.Queue()
    await queue.put(('error', {'code': 'generation_failed', 'message': 'x'}))
    await queue.put(None)

    events = await _collect(_prepared(), False, queue)
    assert [e for e, _ in events] == ['meta', 'sources', 'error', 'done']


@pytest.mark.asyncio
async def test_no_evidence면_sources가_있어도_빈_배열():
    # sources가 비어있지 않아야 가드 유무가 관측됨 (기본값 []면 가드를 지워도 통과하는 허상)
    queue = asyncio.Queue()
    await queue.put(None)
    p = _prepared(sources=[SourceCitation(document_id=5, filename='정책.pdf', version=1)])
    events = await _collect(p, True, queue)
    assert events[0][1]['reason'] == 'no_evidence'   # meta도 함께 — 하드코딩 뮤테이션 방지
    assert dict(events)['sources'] == []


@pytest.mark.asyncio
async def test_생산_이벤트_전체가_순서대로_통과():
    # _run_generation이 넣을 수 있는 이벤트 타입 전부 (실제 흐름에선 sources 정정과 blocked_output이
    # 상호배타지만, 리더는 타입 무관 패스스루가 스펙이라 한 번에 검증)
    queue = asyncio.Queue()
    await queue.put(('token', {'text': 'x'}))
    await queue.put(('sources', []))                        # LLM 자체 거절 시 인용 정정
    await queue.put(('blocked_output', {'message': '차단'}))
    await queue.put(None)

    events = await _collect(_prepared(), False, queue)
    assert [e for e, _ in events] == ['meta', 'sources', 'token', 'sources', 'blocked_output', 'done']


@pytest.mark.asyncio
async def test_소비_시작_후_도착하는_이벤트도_대기해_수신():
    # 리더는 큐가 비어 있으면 대기(blocking get)해야 함 — get_nowait로 바뀌면 프로덕션에서 스트림 즉사
    queue = asyncio.Queue()

    async def late_producer():
        await asyncio.sleep(0.05)
        await queue.put(('token', {'text': '늦게 도착'}))
        await queue.put(None)

    task = asyncio.create_task(late_producer())
    events = await _collect(_prepared(), False, queue)
    await task
    assert [e for e, _ in events] == ['meta', 'sources', 'token', 'done']


@pytest.mark.asyncio
async def test_sources_인용_직렬화():
    queue = asyncio.Queue()
    await queue.put(None)
    p = _prepared(sources=[SourceCitation(document_id=5, filename='정책.pdf', version=2)])
    events = await _collect(p, False, queue)
    assert dict(events)['sources'] == [{'document_id': 5, 'filename': '정책.pdf', 'version': 2}]
