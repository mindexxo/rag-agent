"""생성 태스크(_run_generation)의 이벤트·저장·정리 순서 — DB·Redis·LLM 없이 가짜로 고정한다.

통합 테스트는 최종 상태만 보므로, "done은 finalize·commit 뒤", "error 뒤에도 done", "취소 뒤 재전파",
"finally의 sentinel → 스트림 종료 → 리미터 반납 → 스팬 종료" 순서는 여기서만 잡힌다.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from rag import cancellation, limiter, stream_resume, streaming
from rag.citation_labels import TAIL_END, TAIL_START
from rag.models import TurnStatus
from rag.prepared import PreparedRag
from rag.retrieval_types import RetrievalResult
from schemas.kms import SourceCitation

SOURCE = SourceCitation(document_id=1, filename='a.pdf', version=1)


def _prepared(**kw) -> PreparedRag:
    base = dict(
        conversation_id=1,
        assistant_message_id=7,
        original_query='q',
        standalone_query='q',
        prior_turns=[],
        retrieval=None,  # route=other ⟺ 검색 없음(PreparedRag 불변식)
        sources=[],
        source_doc_ids=[],
        route='other',
    )
    return PreparedRag(**{**base, **kw})


class _Log(list):
    """호출 순서 기록 — 큐·스트림·DB·정리 전부 한 줄로 모은다."""


class _Queue(asyncio.Queue):
    def __init__(self, log: _Log) -> None:
        super().__init__()
        self.log = log

    async def put(self, item) -> None:
        self.log.append('queue:' + (item[0] if item else 'None'))
        await super().put(item)


class _Writer:
    def __init__(self, log: _Log) -> None:
        self.log = log

    def add(self, event: str, data: dict) -> None:
        self.log.append(f'writer:{event}')

    async def aclose(self) -> None:
        self.log.append('writer:aclose')


class _Span:
    """루트 스팬 — 턴 결과 기록(set_attribute 묶음)을 'span.record' 한 줄로 남긴다."""

    def __init__(self, log: _Log) -> None:
        self.log = log

    def is_recording(self) -> bool:
        return True

    def set_attribute(self, key: str, value) -> None:
        if self.log[-1:] != ['span.record']:
            self.log.append('span.record')

    def end(self) -> None:
        self.log.append('span.end')


def _install(monkeypatch, log: _Log, tokens: list[str], raise_after: BaseException | None = None) -> dict:
    """streaming이 쓰는 협력자를 전부 가짜로 — generate는 tokens를 내고 끝에 raise_after를 올린다."""
    calls: dict = {'finalize': [], 'maybe_cache': 0}

    class Session:
        async def commit(self) -> None:
            log.append('commit')

    @asynccontextmanager
    async def session_local():
        yield Session()

    class Service:
        last_finish_reason = None
        last_usage = None

        def __init__(self, tenant_id: str, session) -> None:
            pass

        async def generate(self, prepared):
            for t in tokens:
                yield t
            if raise_after is not None:
                raise raise_after

        async def finalize(self, prepared, answer, citations, status, latency_ms) -> None:
            calls['finalize'].append((status, answer, [c.filename for c in citations]))
            log.append(f'finalize:{status}')

        async def maybe_cache(self, prepared, answer, citations) -> None:
            calls['maybe_cache'] += 1
            log.append('maybe_cache')

    async def release(lease) -> None:
        log.append('release')

    monkeypatch.setattr(streaming, 'AsyncSessionLocal', session_local)
    monkeypatch.setattr(streaming, 'RagService', Service)
    monkeypatch.setattr(stream_resume, 'StreamWriter', lambda tenant_id, message_id: _Writer(log))
    monkeypatch.setattr(limiter, 'release', release)
    monkeypatch.setattr(cancellation, 'unregister', lambda message_id: log.append('unregister'))
    return calls


async def _run(log: _Log, prepared: PreparedRag) -> list:
    queue = _Queue(log)
    await streaming._run_generation(prepared, queue, SimpleNamespace(tenant_id='t'), 0.0, _Span(log))
    items = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


FINALLY = ['unregister', 'queue:None', 'writer:aclose', 'release', 'span.end']


@pytest.mark.asyncio
async def test_성공은_finalize_commit_뒤_done이고_캐시는_done_뒤다(monkeypatch):
    log = _Log()
    calls = _install(monkeypatch, log, ['안', '녕'])
    items = await _run(log, _prepared())
    assert log == [
        'writer:meta',
        'queue:delta', 'writer:delta',
        'queue:delta', 'writer:delta',
        'finalize:done', 'commit',
        'span.record', 'queue:done', 'writer:done',
        'unregister',  # 캐시 적재 중 늦은 취소가 done 턴을 덮지 않게 먼저 빠진다
        'maybe_cache', 'commit',
        *FINALLY,
    ]  # fmt: skip
    assert calls['finalize'] == [(TurnStatus.DONE, '안녕', [])]
    assert [e for e, _ in items[:-1]] == ['delta', 'delta', 'done'] and items[-1] is None
    assert items[2][1]['finish_reason'] == TurnStatus.DONE and items[2][1]['citations'] == []


@pytest.mark.asyncio
async def test_knowledge는_꼬리를_delta에서_떼고_인용으로_바꾼다(monkeypatch):
    log = _Log()
    calls = _install(monkeypatch, log, ['답변 ', f'{TAIL_START}1{TAIL_END}'])
    items = await _run(
        log,
        _prepared(
            route='knowledge', sources=[SOURCE], retrieval=RetrievalResult(chunks=[], no_evidence=False, reason=None)
        ),
    )
    deltas = [d['text'] for e, d in items[:-1] if e == 'delta']
    assert deltas == ['답변 ']  # 꼬리는 본문이 아니다
    done = dict(items[:-1])['done']
    assert done['citations'] == [SOURCE.model_dump()]
    assert calls['finalize'] == [(TurnStatus.DONE, '답변 ', ['a.pdf'])]


@pytest.mark.asyncio
async def test_취소는_부분답변을_cancelled로_남기고_done_뒤_재전파한다(monkeypatch):
    log = _Log()
    calls = _install(monkeypatch, log, ['부분'], raise_after=asyncio.CancelledError())
    queue = _Queue(log)
    with pytest.raises(asyncio.CancelledError):
        await streaming._run_generation(_prepared(), queue, SimpleNamespace(tenant_id='t'), 0.0, _Span(log))
    assert log == [
        'writer:meta',
        'queue:delta', 'writer:delta',
        'finalize:cancelled', 'commit',  # 새 세션(out-of-band) — 본 세션은 끊겼을 수 있다
        'span.record', 'queue:done', 'writer:done',
        *FINALLY,
    ]  # fmt: skip
    assert calls['finalize'] == [(TurnStatus.CANCELLED, '부분', [])]
    assert calls['maybe_cache'] == 0


@pytest.mark.asyncio
async def test_실패는_error_뒤에도_done으로_닫고_예외를_삼킨다(monkeypatch):
    log = _Log()
    calls = _install(monkeypatch, log, ['부분'], raise_after=RuntimeError('LLM down'))
    items = await _run(log, _prepared())
    assert log == [
        'writer:meta',
        'queue:delta', 'writer:delta',
        'finalize:failed', 'commit',
        'span.record', 'queue:error', 'writer:error',  # 스팬 기록 → error → done
        'queue:done', 'writer:done',
        *FINALLY,
    ]  # fmt: skip
    assert calls['finalize'] == [(TurnStatus.FAILED, '부분', [])]
    error, done = dict(items[:-1])['error'], dict(items[:-1])['done']
    assert error == {
        'code': 'generation_failed',
        'message': '답변 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요.',
    }
    assert done['finish_reason'] == TurnStatus.FAILED and done['citations'] == []


@pytest.mark.asyncio
async def test_finalize_실패도_failed_경로로_닫는다(monkeypatch):
    """성공 생성 뒤 저장이 죽으면 — 본 세션 finalize 예외가 except Exception으로 가 failed로 기록된다."""
    log = _Log()
    calls = _install(monkeypatch, log, ['답'])
    real_finalize = streaming.RagService.finalize
    state = {'n': 0}

    async def flaky(self, prepared, answer, citations, status, latency_ms):
        state['n'] += 1
        if state['n'] == 1:
            raise RuntimeError('db down')
        await real_finalize(self, prepared, answer, citations, status, latency_ms)

    monkeypatch.setattr(streaming.RagService, 'finalize', flaky)
    items = await _run(log, _prepared())
    assert calls['finalize'] == [(TurnStatus.FAILED, '답', [])]
    assert [e for e, _ in items[:-1]] == ['delta', 'error', 'done']


@pytest.mark.asyncio
async def test_캐시_적재_실패는_done_턴을_건드리지_않는다(monkeypatch):
    log = _Log()
    calls = _install(monkeypatch, log, ['답'])

    async def boom(self, prepared, answer, citations):
        log.append('maybe_cache')
        raise RuntimeError('cache down')

    monkeypatch.setattr(streaming.RagService, 'maybe_cache', boom)
    items = await _run(log, _prepared())
    assert calls['finalize'] == [(TurnStatus.DONE, '답', [])]  # failed로 덮어쓰지 않는다
    assert [e for e, _ in items[:-1]] == ['delta', 'done']
    assert log[-6:] == ['maybe_cache', *FINALLY]  # 예외는 삼키고 정리는 그대로
