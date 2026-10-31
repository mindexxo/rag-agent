"""상용 벤더 구현 계약 (#288) — 키·네트워크 없이, litellm.acompletion을 가짜로 바꿔 요청 인자와 응답 해석을 고정한다.

고정하는 것(#288 결정):
1. 모델 접두·키는 공통 설정에서(결정 6), max_tokens는 vLLM과 같은 값, temperature·reasoning_effort는 설정됐을 때만(결정 5)
2. schema= → JSON 모드만(결정 3 = b), citations= → 요청에 아무것도 싣지 않는다(결정 2 = a, 파서 검증)
3. 캐시 표시는 anthropic에만 붙고(openai·gemini는 자동 캐싱 — Gemini에 표시하면 명시적 캐싱으로 번역돼 429) 호출부 메시지는 안 바뀐다(결정 4)
4. 스트리밍: 텍스트 조각·종료 사유(OpenAI 어휘 → 공통 어휘)·usage(캐시 토큰) 콜백 — include_usage는 세 벤더 모두 요청
5. 등록부: gemini·openai·anthropic 이름으로 생긴다
실제 벤더가 요청을 받아 주는지(JSON 모드·캐싱 적중·꼬리 위반율)는 키 확보 후 라이브로 잰다 — 여기 범위 밖.
"""
from types import SimpleNamespace

import pytest

from config import settings
from rag.llm import CitationTail, LlmUsage, create_llm
from rag.llm.litellm import CACHE_MARK, LiteLlmClient, with_cache_mark
from rag.turn.llm_schemas import RouteDecision

MSGS = [{'role': 'system', 'content': '규칙'}, {'role': 'user', 'content': '질문'}]


def _resp(text='답', finish='stop', cached=0, prompt=10, completion=2, write=0):
    usage = SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
                            cache_creation_input_tokens=write)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)], usage=usage)


def _chunk(text=None, finish=None, usage=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)], usage=usage)


class _Stream:
    def __init__(self, chunks): self._it = iter(chunks)
    def __aiter__(self): return self
    async def __anext__(self):
        try: return next(self._it)
        except StopIteration: raise StopAsyncIteration


@pytest.fixture
def fake_litellm(monkeypatch):
    """litellm.acompletion을 가로채 호출 인자를 기록하고 준비된 응답을 돌려준다."""
    import litellm
    calls: list[dict] = []
    state = {'response': _resp(), 'chunks': [_chunk('안'), _chunk('녕', finish='stop', usage=_resp(cached=1200).usage)]}

    async def fake(**kwargs):
        calls.append(kwargs)
        return _Stream(state['chunks']) if kwargs.get('stream') else state['response']

    monkeypatch.setattr(litellm, 'acompletion', fake)
    monkeypatch.setattr(settings, 'llm_model', 'test-model')
    monkeypatch.setattr(settings, 'llm_api_key', 'sk-test')
    return SimpleNamespace(calls=calls, state=state)


class TestRequest:
    def test_모델_접두와_키는_공통_설정에서(self):
        for vendor in ('gemini', 'openai', 'anthropic'):
            kw = LiteLlmClient(vendor, model='m', api_key='k')._request_kwargs(None, None)
            assert kw['model'] == f'{vendor}/m' and kw['api_key'] == 'k'
            assert kw['max_tokens'] == settings.generation_reserve_tokens      # vLLM과 같은 상한
        with pytest.raises(ValueError):
            LiteLlmClient('없는벤더')

    def test_온도와_추론수준은_설정됐을_때만(self, monkeypatch):
        monkeypatch.setattr(settings, 'llm_temperature', 0.2)
        monkeypatch.setattr(settings, 'llm_reasoning_effort', None)
        kw = LiteLlmClient('gemini')._request_kwargs(None, None)
        assert kw['temperature'] == 0.2 and 'reasoning_effort' not in kw
        monkeypatch.setattr(settings, 'llm_temperature', None)
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'low')
        kw = LiteLlmClient('anthropic')._request_kwargs(None, None)
        assert 'temperature' not in kw and kw['reasoning_effort'] == 'low'    # 거부 조합은 벤더 400으로 드러난다(조용히 안 뺌)

    def test_스키마는_JSON_모드만_꼬리는_안_보낸다(self):
        kw = LiteLlmClient('openai')._request_kwargs(RouteDecision, None)
        assert kw['response_format'] == {'type': 'json_object'}                # 결정 3 = b — 스키마 강제 아님
        kw = LiteLlmClient('openai')._request_kwargs(None, CitationTail(count=3, start='«', end='»'))
        assert 'response_format' not in kw and 'extra_body' not in kw        # 결정 2 = a — 프롬프트 + 파서


class TestCacheMark:
    def test_system에만_표시가_붙고_원본은_안_바뀐다(self):
        marked = with_cache_mark(MSGS)
        assert marked[0] == {'role': 'system', 'content': [{'type': 'text', 'text': '규칙', 'cache_control': CACHE_MARK}]}
        assert marked[1] == MSGS[1]
        assert MSGS[0]['content'] == '규칙'                                      # 호출부 메시지 불변

    def test_표시는_anthropic에만(self):
        assert LiteLlmClient('anthropic')._messages(MSGS)[0]['content'][0]['cache_control'] == CACHE_MARK
        for vendor in ('gemini', 'openai'):
            assert LiteLlmClient(vendor)._messages(MSGS) is MSGS                  # 그대로 — Gemini는 표시하면 명시적 캐싱(과금·429)


class TestCalls:
    @pytest.mark.asyncio
    async def test_acomplete는_표시된_메시지로_부르고_usage를_올린다(self, fake_litellm):
        got = []
        text = await LiteLlmClient('anthropic').acomplete(MSGS, schema=RouteDecision, on_usage=got.append)
        assert text == '답'
        call = fake_litellm.calls[0]
        assert call['model'] == 'anthropic/test-model' and call['api_key'] == 'sk-test' and call['stream'] is False
        assert call['messages'][0]['content'][0]['cache_control'] == CACHE_MARK
        assert call['response_format'] == {'type': 'json_object'}
        assert got == [LlmUsage(prompt_tokens=10, completion_tokens=2, cached_tokens=0, cache_write_tokens=0)]

    @pytest.mark.asyncio
    async def test_astream은_조각_종료사유_usage를_올린다(self, fake_litellm):
        reasons, usages = [], []
        text = [t async for t in LiteLlmClient('gemini').astream(MSGS, on_finish_reason=reasons.append, on_usage=usages.append)]
        assert text == ['안', '녕'] and reasons == ['stop']
        assert usages[-1].cached_tokens == 1200                                 # 캐시 적중 토큰 — 비용 검증의 근거
        assert fake_litellm.calls[0]['stream_options'] == {'include_usage': True}   # 세 벤더 모두 — LiteLLM이 켜야만 usage 청크를 준다
        assert fake_litellm.calls[0]['messages'] is MSGS                        # gemini: 표시 없음

    @pytest.mark.asyncio
    async def test_잘림은_length_모르는_값은_other(self, fake_litellm):
        fake_litellm.state['chunks'] = [_chunk('a', finish='length'), _chunk(None, finish='content_filter')]
        reasons = []
        [t async for t in LiteLlmClient('openai').astream(MSGS, on_finish_reason=reasons.append)]
        assert reasons == ['length', 'other']


class TestRegistry:
    def test_세_벤더가_설정_이름으로_생긴다(self):
        for name in ('gemini', 'openai', 'anthropic'):
            client = create_llm(name)
            assert isinstance(client, LiteLlmClient) and client.vendor == name

    def test_설정의_추론수준은_vllm_thinking_토글로(self, monkeypatch):
        from rag.llm.vllm import VllmClient
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'low')
        assert VllmClient()._request_kwargs(None, None)['extra_body']['chat_template_kwargs'] == {'enable_thinking': True}
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'none')
        assert VllmClient()._request_kwargs(None, None)['extra_body']['chat_template_kwargs'] == {'enable_thinking': False}

    def test_옛_thinking_설정_이름은_기동_실패(self, monkeypatch):
        from config import Settings
        monkeypatch.setenv('LLM_ENABLE_THINKING', 'false')
        with pytest.raises(ValueError, match='LLM_REASONING_EFFORT'):
            Settings()
