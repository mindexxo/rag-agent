"""구조화 출력 공용 헬퍼 계약 테스트 (#43) — rag/turn/llm_schemas.acomplete_validated.

고정하는 것: (a) 스키마가 **의도**(schema=)로 공급자에 넘어간다(#287 — 서버 문법은 구현 몫)
(b) 응답의 펜스·부연 방어 (c) 비JSON·스키마 위반→LlmJudgmentFailed, 부분 복구 없음
(d) 호출 자체 실패→예외 전파(재시도 없음). 구 방어 파서(_as_bool)가 pydantic lax 파싱으로
대체됐음도 여기서 회귀 고정한다. 스키마 거부 서버를 위한 무스키마 재시도는 #287에서 지웠다.
"""
import pytest

from rag.turn.llm_schemas import CondenseMultiResult, LlmJudgmentFailed, RouteDecision, acomplete_validated


class _StubLlm:
    """응답 시퀀스를 순서대로 돌려주는 최소 스텁 — 호출 기록으로 넘어간 의도를 검증한다."""

    def __init__(self, responses: list):
        self._responses = iter(responses)
        self.calls: list = []          # 호출별 schema 의도

    async def acomplete(self, messages, *, schema=None, citations=None, on_usage=None):
        self.calls.append(schema)
        item = next(self._responses)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.asyncio
async def test_정상_스키마_경로():
    llm = _StubLlm(['{"safe": true, "intent": "KNOWLEDGE"}'])
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision == RouteDecision(safe=True, intent='KNOWLEDGE')
    assert llm.calls == [RouteDecision]   # 스키마가 의도로 실렸다 — 서버 문법은 공급자 구현이 만든다


def test_RETRY는_스키마_화이트리스트에_있다():
    """#59 — 전이 인텐트 RETRY가 guided enum·검증 양쪽에서 유효해야 분류기가 낼 수 있다."""
    assert RouteDecision(safe=True, intent='RETRY').intent == 'RETRY'
    assert 'RETRY' in RouteDecision.model_json_schema()['properties']['intent']['enum']


def test_ATTACHMENT는_스키마_화이트리스트에_있다():
    """#63 — RETRY와 같은 전이 인텐트 확장 패턴."""
    assert RouteDecision(safe=True, intent='ATTACHMENT').intent == 'ATTACHMENT'
    assert 'ATTACHMENT' in RouteDecision.model_json_schema()['properties']['intent']['enum']


@pytest.mark.asyncio
async def test_응답의_펜스와_부연은_방어된다():
    # 스키마를 못 거는 공급자가 와도 같은 검증을 탄다 — 펜스·부연은 벗기고 JSON만 읽는다
    llm = _StubLlm(['판단 결과입니다:\n```json\n{"safe": true, "intent": "OTHER"}\n```'])
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision.intent == 'OTHER'


@pytest.mark.asyncio
async def test_비JSON_응답은_판단실패():
    llm = _StubLlm(['중괄호가 전혀 없는 자유 서술'])
    with pytest.raises(LlmJudgmentFailed):
        await acomplete_validated(llm, [], RouteDecision)


@pytest.mark.asyncio
async def test_스키마_위반은_판단실패_부분복구_없음():
    # #72로 None 반환이 예외가 됐다 — 판단하지 못한 상태에서 호출부가 기본값을 추측하던
    # 폴백을 없앴기 때문. 부분·퍼지 복구를 하지 않는다는 계약 자체는 그대로다.
    llm = _StubLlm(['{"safe": true, "intent": "knowledge"}'])       # enum 밖 — 소문자 이탈
    with pytest.raises(LlmJudgmentFailed):
        await acomplete_validated(llm, [], RouteDecision)
    llm = _StubLlm(['{"intent": "KNOWLEDGE"}'])                     # 필수 필드 누락 — required 유지의 계약
    with pytest.raises(LlmJudgmentFailed):
        await acomplete_validated(llm, [], RouteDecision)


@pytest.mark.asyncio
async def test_호출_실패는_재시도_없이_전파():
    """타임아웃도, 스키마 거부(400)도 그대로 전파된다 — #287에서 무스키마 재시도를 지웠다."""
    llm = _StubLlm([TimeoutError('서버 무응답')])
    with pytest.raises(TimeoutError):
        await acomplete_validated(llm, [], RouteDecision)
    assert len(llm.calls) == 1
    rejected = Exception('unknown field structured_outputs'); rejected.status_code = 400
    llm = _StubLlm([rejected])
    with pytest.raises(Exception, match='unknown field'):
        await acomplete_validated(llm, [], RouteDecision)
    assert len(llm.calls) == 1                       # 제약 없이 다시 보내지 않는다


@pytest.mark.asyncio
async def test_pydantic_lax_bool이_구_as_bool을_대체():
    llm = _StubLlm(['{"safe": "false", "intent": "OTHER"}'])
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision.safe is False


@pytest.mark.asyncio
async def test_condense_멀티_스키마():
    llm = _StubLlm(['{"standalone": "환불 기간", "variants": ["환불 처리 기한", "반품 기간"]}'])
    parsed = await acomplete_validated(llm, [], CondenseMultiResult)
    assert parsed.standalone == '환불 기간' and len(parsed.variants) == 2
    assert llm.calls == [CondenseMultiResult]
    # 스키마 dict에 required·maxItems가 실제로 들어가는지 (guided 강제력의 실체)
    schema = CondenseMultiResult.model_json_schema()
    assert 'standalone' in schema['required']
    assert schema['properties']['variants']['maxItems'] == 2
