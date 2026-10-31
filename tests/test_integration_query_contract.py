"""질의 API의 FE 계약 — 동시 요청 상한(429), SSE 봉투(이벤트 어휘·필드·헤더), 스트림 전 오류 응답과 상태코드 매핑.

HTTP로만 확인한다. 동시 요청은 가짜 LLM을 생성 도중 멈춰 슬롯을 실제로 쥔 상태를 만든다.
"""

import asyncio

import httpx
import litellm
import openai
import opensearchpy
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


# ── prepare 단계 오류 매핑 — 404는 대화 없음에만, 503은 일시 장애, 그 외는 500 ──

UNAVAILABLE = {'detail': '일시적으로 처리할 수 없습니다. 잠시 후 다시 시도해 주세요.'}
_REQ = httpx.Request('POST', 'http://llm.test/v1/chat/completions')


@pytest.fixture
def span_errors(monkeypatch) -> list[BaseException]:
    """루트 스팬 ERROR 마킹을 가로채 기록한다 — 5xx만 남기고 404는 남기지 않는 규칙 검증용."""
    import rag.otel

    seen: list[BaseException] = []
    monkeypatch.setattr(rag.otel, 'mark_error', lambda sp, exc: seen.append(exc))
    return seen


async def _turn_states(tenant: str) -> list[tuple[str, str]]:
    async with AsyncSessionLocal() as s:
        msgs = (
            (await s.execute(select(Message).where(Message.tenant_id == tenant).order_by(Message.id))).scalars().all()
        )
    return [(m.role, m.status) for m in msgs]


def _lenient_client(tenant: str) -> httpx.AsyncClient:
    """앱 예외를 테스트로 되던지지 않고 500 응답으로 받는 클라이언트 — 예상외 오류 경로 전용."""
    from main import app

    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url='http://testserver',
        headers={'X-Tenant-Id': tenant},
    )


@pytest.mark.asyncio
async def test_스트림_전_오류는_SSE가_아니라_JSON(client, tenant_id, fake_llm, span_errors):
    """QRY-69 — 404·422는 스트림을 열기 전에 끝나 일반 JSON 본문이다. 클라이언트 오류라 스팬 ERROR도 없다."""
    res = await client.post('/kms/query', json={'query': '질문', 'conversation_id': 999999999})
    assert res.status_code == 404
    assert res.headers['content-type'].startswith('application/json')
    assert res.json() == {'detail': '대화를 찾을 수 없습니다.'}
    assert span_errors == []

    res = await client.post('/kms/query', json={'query': ''})
    assert res.status_code == 422
    assert res.headers['content-type'].startswith('application/json')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'exc',
    [
        openai.APIConnectionError(request=_REQ),
        openai.APITimeoutError(request=_REQ),
        openai.InternalServerError('upstream', response=httpx.Response(502, request=_REQ), body=None),
        openai.RateLimitError('slow down', response=httpx.Response(429, request=_REQ), body=None),
        litellm.exceptions.ServiceUnavailableError('busy', 'anthropic', 'claude'),
    ],
    ids=['connect', 'timeout', '5xx', '429', 'litellm-503'],
)
async def test_LLM_호출_실패는_503이고_턴은_failed로_닫힌다(
    client, tenant_id, fake_llm, pass_gate, span_errors, monkeypatch, exc
):
    """QRY-40 — 분류 LLM에 닿지 못하거나 상대가 429·5xx를 주면 형식 오류(LlmJudgmentFailed)와 같은 503. 상용 공급자(litellm) 예외도 같다."""
    await register_faq(client)

    async def _down(*a, **kw):
        raise exc

    monkeypatch.setattr(fake_llm, 'acomplete', _down)
    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})

    assert res.status_code == 503
    assert res.json() == UNAVAILABLE
    assert await _turn_states(tenant_id) == [('user', 'done'), ('assistant', 'failed')]
    assert span_errors == [exc]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('module', 'name', 'exc'),
    [
        ('rag.os_search', 'search_dense_per_query', opensearchpy.ConnectionError('N/A', 'connection refused', None)),
        ('rag.retriever', 'embed_texts', httpx.ConnectError('connection refused', request=_REQ)),
    ],
    ids=['opensearch', 'embedding'],
)
async def test_검색_저장소와_임베딩_장애는_503(
    client, tenant_id, fake_llm, pass_gate, span_errors, monkeypatch, module, name, exc
):
    """QRY-40 — 검색 단계의 연결 장애도 LLM 장애와 같은 503·같은 문구."""
    import importlib

    await register_faq(client)

    async def _down(*a, **kw):
        raise exc

    monkeypatch.setattr(importlib.import_module(module), name, _down)
    res = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})

    assert res.status_code == 503
    assert res.json() == UNAVAILABLE
    assert await _turn_states(tenant_id) == [('user', 'done'), ('assistant', 'failed')]
    assert span_errors == [exc]


@pytest.mark.asyncio
async def test_대화와_무관한_ValueError는_500이고_턴은_failed로_닫힌다(
    client, tenant_id, fake_llm, pass_gate, span_errors, monkeypatch
):
    """QRY-42 — 서버 결함(임베딩 차원 불일치 등)은 404로 위장하지 않는다. 질문은 남고 자리표시는 failed."""
    await register_faq(client)
    exc = ValueError('임베딩 응답 차원이 맞지 않는다')

    async def _broken_embed(texts: list[str], kind: str = 'document') -> list:
        raise exc

    import rag.retriever

    monkeypatch.setattr(rag.retriever, 'embed_texts', _broken_embed)  # fake_embed와 같은 지점
    async with _lenient_client(tenant_id) as lenient:
        res = await lenient.post('/kms/query', json={'query': '환불 기간 알려줘'})

    assert res.status_code == 500
    assert await _turn_states(tenant_id) == [('user', 'done'), ('assistant', 'failed')]
    assert span_errors == [exc]
