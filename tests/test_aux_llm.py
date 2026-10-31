"""보조 LLM 호출 분리(#297) — 의도 분류·질의 재작성·캐시 재사용 판정만 LLM_AUX_*의 모델·생각 수준으로, 생성은 그대로.

두 층:
  - 단위: 요청 바디(설정이 비면 이전과 같다·설정하면 보조만 바뀐다), 스팬 모델 기록.
  - 통합: HTTP 턴 전체에서 세 보조 호출은 보조 클라이언트로, 생성은 생성 클라이언트로 가는지(배선 — 한 고리만 빠져도 조용히 생성 모델로 돈다).
"""
from pathlib import Path

import pytest

from config import settings
from rag.llm import aux_overridden, create_aux_llm, create_llm
from rag.turn.llm_schemas import RouteDecision


@pytest.fixture
def no_aux(monkeypatch):
    monkeypatch.setattr(settings, 'llm_aux_model', '')
    monkeypatch.setattr(settings, 'llm_aux_reasoning_effort', '')


class TestRequestBody:
    @pytest.mark.parametrize('provider', ['vllm', 'openai', 'anthropic', 'gemini'])
    def test_설정이_비면_보조_요청은_생성과_같다(self, monkeypatch, no_aux, provider):
        monkeypatch.setattr(settings, 'llm_provider', provider)
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'low')
        assert not aux_overridden()
        for schema in (None, RouteDecision):
            assert create_aux_llm()._request_kwargs(schema, None) == create_llm()._request_kwargs(schema, None)

    def test_설정이_비면_보조_클라이언트는_생성_클라이언트_그_자체다(self, no_aux):
        # 커넥션 풀도 하나 — rag/clients가 aux_overridden()이 거짓이면 shared_llm을 그대로 쓴다(모듈을 다시 읽지 않고 분기만 본다:
        # reload하면 http_async·shared_redis가 새로 생기고 옛 것이 안 닫힌다, 리뷰 반영)
        import rag.clients as clients
        src = Path(clients.__file__).read_text(encoding='utf-8')
        assert 'shared_aux_llm = create_aux_llm() if aux_overridden() else shared_llm' in src
        assert clients.shared_aux_llm is clients.shared_llm          # 테스트 환경(.env에 LLM_AUX_* 없음)의 실제 값

    def test_vLLM_보조_모델과_생각_수준(self, monkeypatch):
        monkeypatch.setattr(settings, 'llm_provider', 'vllm')
        monkeypatch.setattr(settings, 'llm_model', 'big')
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'high')
        monkeypatch.setattr(settings, 'llm_aux_model', 'small')
        monkeypatch.setattr(settings, 'llm_aux_reasoning_effort', 'none')
        aux, gen = create_aux_llm()._request_kwargs(None, None), create_llm()._request_kwargs(None, None)
        assert (aux['model'], aux['extra_body']['chat_template_kwargs']['enable_thinking']) == ('small', False)
        assert (gen['model'], gen['extra_body']['chat_template_kwargs']['enable_thinking']) == ('big', True)   # 생성은 그대로

    def test_상용_보조_모델과_생각_수준(self, monkeypatch):
        monkeypatch.setattr(settings, 'llm_provider', 'openai')
        monkeypatch.setattr(settings, 'llm_model', 'gpt-5')
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'medium')
        monkeypatch.setattr(settings, 'llm_aux_model', 'gpt-5-mini')
        monkeypatch.setattr(settings, 'llm_aux_reasoning_effort', 'minimal')
        aux, gen = create_aux_llm(), create_llm()
        assert aux.model_name == 'openai/gpt-5-mini' and aux._request_kwargs(None, None)['reasoning_effort'] == 'minimal'
        assert gen.model_name == 'openai/gpt-5' and gen._request_kwargs(None, None)['reasoning_effort'] == 'medium'

    def test_한쪽만_설정하면_나머지는_생성_값을_따른다(self, monkeypatch):
        monkeypatch.setattr(settings, 'llm_provider', 'anthropic')
        monkeypatch.setattr(settings, 'llm_model', 'claude-sonnet-5')
        monkeypatch.setattr(settings, 'llm_reasoning_effort', 'low')
        monkeypatch.setattr(settings, 'llm_aux_model', '')
        monkeypatch.setattr(settings, 'llm_aux_reasoning_effort', 'none')       # 생각만 끈다
        aux = create_aux_llm()
        assert aux.model_name == 'anthropic/claude-sonnet-5'
        assert aux._request_kwargs(None, None)['reasoning_effort'] == 'none'
        monkeypatch.setattr(settings, 'llm_aux_model', 'claude-haiku-4-5')
        monkeypatch.setattr(settings, 'llm_aux_reasoning_effort', '')           # 모델만 바꾼다
        aux = create_aux_llm()
        assert aux.model_name == 'anthropic/claude-haiku-4-5'
        assert aux._request_kwargs(None, None)['reasoning_effort'] == 'low'


class TestSpanModel:
    @pytest.mark.asyncio
    async def test_보조_호출_스팬에_모델_이름(self, monkeypatch):
        import types
        import rag.turn.llm_schemas as ls
        recorded = {}
        monkeypatch.setattr(ls.otel, 'set_attrs', lambda sp, attrs: recorded.update(attrs))

        class _Llm:
            model_name = 'openai/gpt-5-mini'

            async def acomplete(self, messages, *, schema=None, **_):
                return '{"safe": true, "intent": "KNOWLEDGE"}'

        await ls.acomplete_validated(_Llm(), [{'role': 'user', 'content': 'q'}], RouteDecision, span=types.SimpleNamespace())
        assert recorded[ls.otel.LLM_MODEL] == 'openai/gpt-5-mini'


@pytest.mark.asyncio
async def test_한_턴의_보조_호출은_보조_클라이언트로_생성은_생성_클라이언트로(client, tenant_id, pass_gate, fake_embed, monkeypatch):
    """분류·재작성·캐시 판정 세 곳 모두 보조 클라이언트를 타야 한다 — 한 곳만 빠져도 에러 없이 생성 모델로 돈다."""
    import rag.turn.service
    from tests.conftest import FakeLlm, register_faq, sse_meta
    gen, aux = FakeLlm(), FakeLlm()
    monkeypatch.setattr(rag.turn.service, 'shared_llm', gen)
    monkeypatch.setattr(rag.turn.service, 'shared_aux_llm', aux)
    await register_faq(client)

    first = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})        # 분류 + 생성
    conv_id = sse_meta(first)['conversation_id']
    await client.post('/kms/query', json={'query': '그럼 교환은?', 'conversation_id': conv_id})   # + 재작성
    hit = await client.post('/kms/query', json={'query': '환불 기간 알려줘'})          # + 캐시 재사용 판정
    assert sse_meta(hit)['cache_kind'] == 'semantic'

    assert 'intent' in aux.calls and 'reuse_judge' in aux.calls
    assert {'condense', 'condense_multi'} & set(aux.calls)
    assert set(gen.calls) == {'stream:generate'}                                         # 생성 클라이언트엔 생성(스트리밍)만
    assert not any('generate' in c for c in aux.calls)
