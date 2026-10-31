"""LLM 공급자 분리 계약 (#287) — DB·서버 불필요.

고정하는 것:
1. 기본 공급자(vllm)가 서버로 보내는 요청 인자가 #287 전과 **같다**(동작 보존 — 리팩토링 원칙 2).
   기대값은 #287 전 단일 모듈 클라이언트·rag/turn/llm_schemas.py·rag/turn/prompts.py가 만들던 딕셔너리를 그대로 적었다.
2. 의도 → 문법 변환은 구현 몫: 바탕(OpenAI 호환)은 제약을 받으면 ConstraintUnsupported를 낸다(조용한 무시 금지).
3. create_llm은 설정 이름으로 구현을 고르고, 모르는 이름은 즉시 실패한다.
4. 종료 사유는 공통 어휘(stop|length|other)로 맞춘다.
5. 옛 설정 이름(VLLM_*)이 환경에 남아 있으면 기동이 멈춘다 — 기본값으로 조용히 붙는 사고 방지.
"""
import pytest
from pydantic import BaseModel

from config import settings
from rag.llm import CitationTail, ConstraintUnsupported, create_llm
from rag.llm.openai_compat import OpenAiCompatClient, normalize_finish_reason
from rag.llm.vllm import VllmClient
from rag.turn.llm_schemas import RouteDecision
from rag.turn.prompt_texts import TAIL_END, TAIL_START


def _vllm_kwargs(**intent) -> dict:
    return VllmClient()._request_kwargs(intent.get('schema'), intent.get('citations'))


class TestVllm요청은_이전과_같다:
    """#287 전 요청 바디(model·max_tokens·temperature·extra_body)를 그대로 고정한다."""

    def test_제약_없는_호출(self):
        assert _vllm_kwargs() == {
            'model': settings.llm_model,
            'max_tokens': settings.generation_reserve_tokens,
            'temperature': settings.llm_temperature,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},   # llm_reasoning_effort 미설정 = 끔
        }

    def test_스키마_의도는_structured_outputs로(self):
        body = _vllm_kwargs(schema=RouteDecision)['extra_body']
        assert body['structured_outputs'] == {'json': RouteDecision.model_json_schema()}
        assert body['chat_template_kwargs'] == {'enable_thinking': False}   # 병합 — 유실 금지

    def test_꼬리_의도는_structural_tag로(self):
        rf = _vllm_kwargs(citations=CitationTail(count=3, start=TAIL_START, end=TAIL_END))['extra_body']['response_format']
        assert rf['type'] == 'structural_tag'
        assert rf['triggers'] == [TAIL_START]                   # 이 문자열이 나오면 제약이 시작된다
        structure = rf['structures'][0]
        assert structure['begin'] == TAIL_START and structure['end'] == TAIL_END
        assert structure['schema'] == {'type': 'array', 'items': {'type': 'integer', 'enum': [1, 2, 3]}}   # 4가 없다 = 범위 밖 차단

    def test_후보_없으면_빈_배열만_강제(self):
        schema = _vllm_kwargs(citations=CitationTail(count=0, start=TAIL_START, end=TAIL_END))['extra_body']['response_format']['structures'][0]['schema']
        assert schema == {'type': 'array', 'maxItems': 0}       # enum:[] 컴파일은 미검증 — 길이로 강제

    def test_빈_목록도_합법이다(self):
        """거절의 자유 — minItems를 걸지 않는다. 후보가 있어도 모델이 []를 낼 수 있어야 한다
        (프롬프트의 인용 표시 규칙). 라이브에서도 확인했다 — 답할 수 없는 질문에 후보 4개를 주면 2/2 []."""
        schema = _vllm_kwargs(citations=CitationTail(count=3, start=TAIL_START, end=TAIL_END))['extra_body']['response_format']['structures'][0]['schema']
        assert 'minItems' not in schema

    def test_temperature는_설정됐을_때만(self, monkeypatch):
        monkeypatch.setattr(settings, 'llm_temperature', None)
        assert 'temperature' not in _vllm_kwargs()


class TestBase:
    def test_바탕은_제약을_받으면_실패한다(self):
        """조용히 무시하지 않는다 — 제약을 받아 놓고 강제하지 않는 서버가 실제로 있었다(#61)."""
        client = OpenAiCompatClient()
        assert client._request_kwargs(None, None)['model'] == settings.llm_model
        with pytest.raises(ConstraintUnsupported):
            client._request_kwargs(RouteDecision, None)
        with pytest.raises(ConstraintUnsupported):
            client._request_kwargs(None, CitationTail(count=1, start='<', end='>'))

    def test_종료_사유_공통_어휘(self):
        assert normalize_finish_reason('stop') == 'stop'
        assert normalize_finish_reason('length') == 'length'
        assert normalize_finish_reason('content_filter') == 'other'   # 모르는 값은 other — 대시보드가 벤더를 몰라야 한다


class TestCreateLlm:
    def test_설정_이름으로_고른다(self, monkeypatch):
        monkeypatch.setattr(settings, 'llm_provider', 'vllm')   # 실행 환경의 .env(상용 스모크 중일 수 있음)에 묶이지 않게
        assert isinstance(create_llm(), VllmClient)
        assert isinstance(create_llm('vllm'), VllmClient)

    def test_모르는_이름은_즉시_실패(self):
        with pytest.raises(ValueError, match='알 수 없는 LLM 공급자'):
            create_llm('없는벤더')


class TestSettings:
    def test_옛_이름이_환경에_남아_있으면_기동_실패(self, monkeypatch):
        from config import Settings
        monkeypatch.setenv('VLLM_MODEL', 'x')
        with pytest.raises(ValueError, match='VLLM_MODEL → LLM_MODEL'):
            Settings()

    def test_모르는_공급자_이름은_기동_실패(self, monkeypatch):
        from config import Settings
        monkeypatch.setenv('LLM_PROVIDER', 'gpt9')
        with pytest.raises(Exception):
            Settings()
