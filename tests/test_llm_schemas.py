"""구조화 출력 공용 헬퍼 계약 테스트 (#43) — rag/llm_schemas.acomplete_validated.

실패 모드 4갈래를 고정한다: (a) 스키마 거부→무스키마 재시도 (b) 재시도 응답 비JSON→None
(c) 스키마 위반(enum 밖 등)→None (d) 호출 자체 실패→예외 전파(재시도 없음 — 블로킹 콜이라
타임아웃 재시도는 대기만 배가된다). 구 방어 파서(_as_bool)가 pydantic lax 파싱으로
대체됐음도 여기서 회귀 고정한다.
"""
import pytest

from rag.llm_schemas import CondenseMultiResult, RouteDecision, acomplete_validated


class _StubLlm:
    """응답 시퀀스를 순서대로 돌려주는 최소 스텁 — 호출 기록으로 재시도 여부를 검증한다."""

    def __init__(self, responses: list, reject_schema: bool = False):
        self._responses = iter(responses)
        self.reject_schema = reject_schema
        self.calls: list[dict | None] = []          # 호출별 extra_body

    async def acomplete(self, messages, extra_body=None):
        self.calls.append(extra_body)
        if self.reject_schema and extra_body and 'structured_outputs' in extra_body:
            exc = Exception('unknown field structured_outputs')
            exc.status_code = 400
            raise exc
        item = next(self._responses)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.mark.asyncio
async def test_정상_스키마_경로():
    llm = _StubLlm(['{"safe": true, "intent": "KNOWLEDGE"}'])
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision == RouteDecision(safe=True, intent='KNOWLEDGE')
    assert len(llm.calls) == 1 and 'structured_outputs' in llm.calls[0]   # 스키마가 실렸다


@pytest.mark.asyncio
async def test_스키마_거부면_무스키마로_한_번_재시도():
    llm = _StubLlm(['{"safe": false, "intent": "OTHER", "reason": "x"}'], reject_schema=True)
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision.safe is False and decision.reason == 'x'
    assert len(llm.calls) == 2                       # 스키마 시도 + 무스키마 재시도
    assert llm.calls[1] is None                      # 재시도엔 extra_body 없음


@pytest.mark.asyncio
async def test_재시도_응답의_펜스와_부연은_방어된다():
    llm = _StubLlm(['판단 결과입니다:\n```json\n{"safe": true, "intent": "OTHER"}\n```'],
                   reject_schema=True)
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision.intent == 'OTHER'


@pytest.mark.asyncio
async def test_비JSON_응답은_None():
    llm = _StubLlm(['중괄호가 전혀 없는 자유 서술'], reject_schema=True)
    assert await acomplete_validated(llm, [], RouteDecision) is None


@pytest.mark.asyncio
async def test_스키마_위반은_None_부분복구_없음():
    # enum 밖 intent — 소문자 이탈은 폴백으로 (Literal 검증이 곧 화이트리스트)
    llm = _StubLlm(['{"safe": true, "intent": "knowledge"}'])
    assert await acomplete_validated(llm, [], RouteDecision) is None
    # 필수 필드 누락 — 기본값을 안 준 이유(required 유지 = guided 강제력)의 계약 고정
    llm = _StubLlm(['{"intent": "KNOWLEDGE"}'])
    assert await acomplete_validated(llm, [], RouteDecision) is None


@pytest.mark.asyncio
async def test_호출_자체_실패는_재시도_없이_전파():
    llm = _StubLlm([TimeoutError('서버 무응답')])
    with pytest.raises(TimeoutError):
        await acomplete_validated(llm, [], RouteDecision)
    assert len(llm.calls) == 1                       # 타임아웃은 재시도하지 않는다 (대기 배가 방지)


@pytest.mark.asyncio
async def test_pydantic_lax_bool이_구_as_bool을_대체():
    # 무스키마 폴백 응답이 bool을 문자열로 줘도 흡수 — 구 _as_bool(bool("false")==True 함정)의 대체 증명
    llm = _StubLlm(['{"safe": "false", "intent": "OTHER"}'], reject_schema=True)
    decision = await acomplete_validated(llm, [], RouteDecision)
    assert decision.safe is False


@pytest.mark.asyncio
async def test_condense_멀티_스키마():
    llm = _StubLlm(['{"standalone": "환불 기간", "variants": ["환불 처리 기한", "반품 기간"]}'])
    parsed = await acomplete_validated(llm, [], CondenseMultiResult)
    assert parsed.standalone == '환불 기간' and len(parsed.variants) == 2
    # 스키마 dict에 required·maxItems가 실제로 들어가는지 (guided 강제력의 실체)
    schema = llm.calls[0]['structured_outputs']['json']
    assert 'standalone' in schema['required']
    assert schema['properties']['variants']['maxItems'] == 2
