"""LLM 구조화 출력 — 스키마(Pydantic)와 강제 호출 헬퍼 (#43).

프롬프트로 형식을 간청하고 방어 파서로 수습하던 세 지점(가드+인텐트, condense 단일,
condense 멀티)을 vLLM structured_outputs의 json 스키마 강제로 바꾼다. 모델 정의가 곧
스키마(model_json_schema)이자 검증(model_validate_json) — 정의·검증이 한 곳이다.

공개 표면: RouteDecision · CondenseResult · CondenseMultiResult · acomplete_validated ·
is_schema_rejected(#61에서 승격 — eval 하네스도 같은 판정을 쓴다, 사유는 그 docstring).
나머지는 내부 헬퍼(_).

원칙 (rag/citation_tail.py #56과 동일): guided decoding은 확률을 낮추는 최적화지 신뢰의
근거가 아니다 — 스키마가 걸렸든(정상) 무스키마 재시도로 떨어졌든(fail-open) 검증은 항상 돈다.

경계: 이 모듈은 "구조를 신뢰할 수 있는가"까지만 책임진다. 실패 시 무엇으로 대체할지는
호출부마다 달라(가드=fail-open RouteDecision, condense=[query]) 여기서 정하지 않는다 —
호출부가 None(구조 신뢰 불가)이나 예외(호출 자체 실패)를 받아 자기 폴백으로 채우고,
그 대입 지점에서 kms.schema_fallback을 기록한다(폴백이 정상 판정과 트레이스에서 구분되게 —
이 관측이 이 이슈의 원래 문제의식이다).

스키마 필드에 기본값을 주지 말 것: 기본값이 있으면 json 스키마에서 required가 빠져
guided 경로에서 모델이 필드를 생략해도 합법이 된다 — 강제력이 조용히 약화된다.
누락은 검증 실패 → 폴백으로 가는 게 맞고, 그 편이 kms.schema_invalid로 관측까지 된다.
(이 원칙은 구조를 결정하는 필드 — safe·intent·standalone — 에 한한다. reason·variants는
원래 없을 수 있는 값이라 옵셔널이 맞다.)

배치가 schemas/가 아니라 rag/인 이유: schemas/는 FastAPI 요청·응답(API 경계) 계약이고,
이건 LLM에 거는 생성 제약이라는 내부 계약이다 — build_citation_grammar(rag/prompts)와 같은 축.
"""
import logging
from functools import cache
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from rag import otel
from rag.llm import LlmClient

logger = logging.getLogger(__name__)


class RouteDecision(BaseModel):
    """입력 가드레일 + 인텐트 분류 통합 결과 (구 rag/guardrail.py의 dataclass를 대체).

    intent의 Literal 검증이 곧 화이트리스트다 — 무스키마 폴백 경로에서 'knowledge'(소문자)
    같은 이탈은 검증 실패 → 호출부 fail-open으로 떨어지고 kms.schema_invalid로 관측된다.
    """
    safe: bool
    intent: Literal['KNOWLEDGE', 'OTHER']
    reason: str | None = None   # safe=false일 때 차단 사유 — 옵셔널은 의도(정상 턴엔 없음)


class CondenseResult(BaseModel):
    """condense_query(단일) 출력 — 검색 가능한 독립 질문 한 문장."""
    standalone: str


class CondenseMultiResult(BaseModel):
    """condense_to_queries(멀티, #5) 출력. standalone은 저장·캐시 키·리랭크 기준,
    variants는 검색 전용 어휘 변형. 필드 이름이 구 '몇 번째 줄' 계약을 대체한다 —
    라벨·머리말이 끼어들 자리가 구조적으로 없다.

    maxItems=2는 스키마에만 싣고(guided 경로 강제) 검증에는 안 건다(json_schema_extra) —
    max_length로 걸면 무스키마 폴백 응답이 변형 3개를 줬을 때 멀쩡한 standalone까지
    통째로 폴백된다(과잉). 초과분은 후처리의 [:2] 슬라이스가 자른다.
    """
    standalone: str
    variants: list[str] = Field(default_factory=list, json_schema_extra={'maxItems': 2})


@cache
def _schema_extra_body(model_cls: type[BaseModel]) -> dict:
    """vLLM v0.12+ structured_outputs 형식. 세 스키마 모두 런타임 값이 없는 정적이라 캐시 —
    요청마다 조립하는 build_citation_grammar(후보 의존)와 다른 점."""
    return {"structured_outputs": {"json": model_cls.model_json_schema()}}


def is_schema_rejected(exc: Exception) -> bool:
    """서버가 extra_body 파라미터 자체를 거부한 경우만 좁게 (400/422 — 구버전 vLLM).

    타임아웃·5xx는 재시도해도 같은 결과인데 블로킹 호출이라 대기만 배가 된다 —
    #56 astream(스트리밍, 첫 토큰 전 전부 재시도)과 의도적으로 다른 지점.
    status_code 덕 타이핑: openai.APIStatusError 계열이 이 속성을 갖고, 테스트 fake도 흉내낸다.

    공개 심볼인 이유(#61): 소비처가 둘이다 — 여기(structured_outputs)와
    eval/generation.py(출처 꼬리 문법). 둘 다 acomplete에 extra_body를 실어 보내고
    같은 근거로 같은 폭을 쓴다. 규칙을 복제하면 위 rationale은 한 곳에만 남고
    판정은 두 곳에서 갈릴 수 있다. 판정 자체는 스키마에 한정되지 않으므로 이름도
    extra_body 기준으로 읽는다.
    """
    return getattr(exc, 'status_code', None) in (400, 422)


def _extract_json_slice(raw: str) -> str:
    """무스키마 재시도 응답의 ```json 펜스·부연 텍스트 방어 — 구 guardrail._extract_json의 승격.
    첫 '{'부터 마지막 '}'까지. 스키마 강제 응답(순수 JSON)엔 항등이라 양쪽에 걸어도 무해.
    '{'가 없으면 ValueError — 호출자(acomplete_validated)의 검증 실패 처리로 합류."""
    return raw[raw.index('{'):raw.rindex('}') + 1]


async def acomplete_validated(llm: LlmClient, messages: list[dict],
                              model_cls: type[BaseModel], *, span=None) -> BaseModel | None:
    """스키마 강제 acomplete + 검증. 반환 3갈래:

      모델 인스턴스  정상 (스키마 경로든 재시도 경로든 검증 통과)
      None          응답은 받았으나 구조를 신뢰할 수 없음 → 호출부 폴백 (kms.schema_invalid 기록됨)
      예외 전파      호출 자체 실패(타임아웃·서버 다운) → 호출부의 기존 except가 폴백

    호출부는 None과 예외를 같은 폴백으로 합류시키고 그 지점에서 kms.schema_fallback을 기록할 것.
    """
    try:
        raw = await llm.acomplete(messages, extra_body=_schema_extra_body(model_cls))
    except Exception as exc:
        if not is_schema_rejected(exc):
            raise
        logger.warning('structured_outputs 미지원 서버 — 스키마 없이 재시도 (%s)', model_cls.__name__)
        if span is not None:
            otel.set_attrs(span, {'kms.schema_retry': True})
        raw = await llm.acomplete(messages)   # 여기서도 실패하면 그대로 전파 (이중 안전망 없음)

    try:
        return model_cls.model_validate_json(_extract_json_slice(raw or ''))
    except (ValidationError, ValueError):
        # ValidationError는 ValueError의 서브클래스지만 의도를 드러내려 병기.
        # 부분·퍼지 복구는 하지 않는다 — 그럴듯한 복구는 실패를 지표에서 숨긴다(#56 원칙).
        logger.warning('구조화 출력 검증 실패 (%s) raw=%r', model_cls.__name__, (raw or '')[:200])
        if span is not None:
            otel.set_attrs(span, {'kms.schema_invalid': True})
        return None
