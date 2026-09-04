"""서빙 관측 단위 테스트 (#129) — DB·Redis·vLLM 불필요 (빠른 서브셋 편입 가능).

검증 대상 셋:
1. LlmClient.astream의 on_finish_reason 배선 — 마지막 청크(델타 없음)가 예전처럼 유실되지
   않고 콜백으로 종료 사유가 나오는가. 텍스트 스트림은 기존과 동일한가(동작 보존).
2. rag/metrics.py의 지표가 기록되는가 — 전역 REGISTRY 누적 대비 **델타**로 단언한다
   (절대값 단언은 테스트 실행 순서에 따라 깨진다 — 이 파일의 모든 지표 단언 공통 규율).
3. RagService.generate가 last_vllm_finish_reason을 채우는가 (콜백→보관 배선).
"""
import asyncio
from types import SimpleNamespace

import pytest
from prometheus_client import REGISTRY

from rag.llm import LlmClient
from rag.metrics import FINISH_REASON_TOTAL, TTFT_SECONDS


# ── 1. LlmClient.astream 콜백 배선 ─────────────────────────


class _FakeOpenAiStream:
    """AsyncOpenAI 스트림 흉내 — 델타 청크들 + finish_reason만 실린 마지막 청크."""

    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration

    async def close(self):
        self.closed = True


def _chunk(content: str | None, finish_reason: str | None = None):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=content), finish_reason=finish_reason)])


def _patch_stream(client: LlmClient, chunks) -> _FakeOpenAiStream:
    fake = _FakeOpenAiStream(chunks)

    async def _create(**kwargs):
        return fake

    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_create)))
    return fake


@pytest.mark.asyncio
async def test_마지막_청크의_finish_reason이_콜백으로_나온다():
    client = LlmClient()
    fake = _patch_stream(client, [
        _chunk('안'), _chunk('녕'),
        _chunk(None, finish_reason='length'),   # 예전 `if delta:` 가드가 삼키던 청크
    ])
    got: list[str] = []
    text = [t async for t in client.astream([], on_finish_reason=got.append)]
    assert text == ['안', '녕']            # 동작 보존 — 텍스트는 그대로
    assert got == ['length']               # 종료 사유가 유실되지 않는다
    assert fake.closed                     # 기존 cleanup 동작 보존


@pytest.mark.asyncio
async def test_콜백_없이도_기존_동작_그대로():
    client = LlmClient()
    _patch_stream(client, [_chunk('ok'), _chunk(None, finish_reason='stop')])
    assert [t async for t in client.astream([])] == ['ok']


@pytest.mark.asyncio
async def test_델타와_finish_reason이_같은_청크에_실려도_둘_다_처리():
    # vLLM 구현에 따라 마지막 토큰과 종료 사유가 한 청크에 올 수 있다 — 어느 쪽도 잃으면 안 된다.
    client = LlmClient()
    _patch_stream(client, [_chunk('끝', finish_reason='stop')])
    got: list[str] = []
    text = [t async for t in client.astream([], on_finish_reason=got.append)]
    assert text == ['끝'] and got == ['stop']


# ── 2. 지표 기록 (델타 단언) ───────────────────────────────


def _sample(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_지표_기록_히스토그램과_카운터():
    # 실제 코드가 절대 안 쓰는 라벨값으로 다른 테스트와의 간섭을 차단
    before_h = _sample('kms_ttft_seconds_count', {'route': '__test__'})
    TTFT_SECONDS.labels(route='__test__').observe(0.3)
    assert _sample('kms_ttft_seconds_count', {'route': '__test__'}) == before_h + 1

    before_c = _sample('kms_llm_finish_reason_total', {'route': '__test__', 'reason': 'length'})
    FINISH_REASON_TOTAL.labels(route='__test__', reason='length').inc()
    assert _sample('kms_llm_finish_reason_total',
                   {'route': '__test__', 'reason': 'length'}) == before_c + 1


# ── 3. RagService 배선 — 완주 시 last_vllm_finish_reason 보관 ─


@pytest.mark.asyncio
async def test_generate_완주시_vllm_finish_reason_보관():
    from test_service_pure import _prepared
    from rag.service import RagService

    class _Llm:
        async def astream(self, prompt, extra_body=None, on_finish_reason=None):
            yield '답'
            if on_finish_reason:
                on_finish_reason('length')

    svc = RagService(tenant_id='t', session=None)
    svc._llm = _Llm()
    prepared = _prepared(original_query='q', standalone_query='q')
    assert ''.join([t async for t in svc.generate(prepared)]) == '답'
    assert svc.last_vllm_finish_reason == 'length'


@pytest.mark.asyncio
async def test_스트림_중단시_finish_reason은_None():
    from test_service_pure import _prepared
    from rag.service import RagService

    class _Llm:
        async def astream(self, prompt, extra_body=None, on_finish_reason=None):
            yield '일'
            raise RuntimeError('중단 (테스트 주입)')

    svc = RagService(tenant_id='t', session=None)
    svc._llm = _Llm()
    prepared = _prepared(original_query='q', standalone_query='q')
    with pytest.raises(RuntimeError):
        async for _ in svc.generate(prepared):
            pass
    assert svc.last_vllm_finish_reason is None   # 완주 아님 — 집계 대상에서 빠져야 한다
