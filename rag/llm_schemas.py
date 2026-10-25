"""LLM 구조화 출력 — 응답 스키마(Pydantic)와 스키마 강제 호출 + 검증 헬퍼.

모델 정의가 곧 JSON 스키마이자 검증기다. 스키마를 서버 문법으로 바꾸는 일은 공급자 구현(rag/llm)이 하고,
여기는 `schema=`로 의도만 넘긴다. 스키마 강제는 확률을 낮추는 최적화일 뿐이라 검증은 항상 돈다.

계약은 두 갈래 — 검증을 통과한 모델 인스턴스이거나 예외다. 기본값을 추측해 진행하지 않는다
(가드 폴백은 입력 검사를 끈 채 통과시키는 것이라 안전 측이 아니다).

구조를 결정하는 필드(safe·intent·standalone)에 기본값을 주지 말 것 — 기본값이 있으면 스키마의 required에서
빠져 모델이 필드를 생략해도 합법이 된다. reason·variants는 원래 없을 수 있어 옵셔널이다.
"""

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from rag import otel
from rag.llm import LlmClient

logger = logging.getLogger(__name__)


class LlmJudgmentFailed(RuntimeError):
    """서버는 응답했지만 구조화 출력이 검증을 통과하지 못했다 — 호출 자체의 실패(타임아웃 등)와 구분한다.

    routers/kms.py가 503으로 매핑한다(재시도로 풀릴 수 있는 일시적 실패).
    """


class RouteDecision(BaseModel):
    """입력 가드레일 + 인텐트 분류 통합 결과 (구 rag/guardrail.py의 dataclass를 대체).

    intent의 Literal 검증이 곧 화이트리스트다 — 'knowledge'(소문자) 같은 이탈은 검증 실패로
    끝나고 kms.schema_invalid로 관측된다.

    RETRY(#59)·ATTACHMENT(#63)는 표면 패턴 인식용 **전이 인텐트**다 — RETRY는 재요청
    발화("다시"/"이어서"), ATTACHMENT는 첨부 문서 자체를 대상으로 하는 요청("이 문서
    요약해줘"). prepare()의 디스패처가 상태(직전 턴/첨부 유무)를 보고 즉시 해소한다.
    RETRY는 저장 계층에 절대 도달하지 않고, ATTACHMENT는 라우팅에선 소멸하되
    저장 라벨(messages.intent='ATTACHMENT')로는 남는다 — 첨부만 근거인 턴이
    KNOWLEDGE로 집계되면 stats의 KB 커버리지 지표가 오염되기 때문(#63 리뷰).
    """

    safe: bool
    intent: Literal['KNOWLEDGE', 'OTHER', 'RETRY', 'ATTACHMENT']
    reason: str | None = None  # safe=false일 때 차단 사유 — 옵셔널은 의도(정상 턴엔 없음)


class ReuseJudgment(BaseModel):
    """캐시 재사용 판정(#113) 출력 — 임계 아래 대역 후보를 재사용해도 되는가.

    reason은 관측용(Phoenix 스팬·로그) — 판정 근거가 남아야 오판 사례를 역추적할 수 있다.
    검증 실측은 rag/prompt_texts.py의 CACHE_REUSE_JUDGE_SYSTEM 주석 참조.
    """

    same_answer: bool
    reason: str


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


def _extract_json_slice(raw: str) -> str:
    """첫 '{'부터 마지막 '}'까지 — 펜스·부연 텍스트 방어. '{'가 없으면 ValueError(검증 실패로 합류)."""
    return raw[raw.index('{') : raw.rindex('}') + 1]


async def acomplete_validated(
    llm: LlmClient, messages: list[dict], model_cls: type[BaseModel], *, span=None
) -> BaseModel:
    """스키마 강제 호출 + 검증. 검증 실패는 LlmJudgmentFailed(kms.schema_invalid 기록),
    호출 자체 실패(타임아웃·스키마 거부)는 그대로 전파한다. 호출부는 아무것도 잡지 않는다.
    """
    if span is not None:
        otel.set_attrs(span, {otel.LLM_MODEL: getattr(llm, 'model_name', None)})  # 보조 호출이 어느 모델이었나
    raw = await llm.acomplete(messages, schema=model_cls)

    try:
        return model_cls.model_validate_json(_extract_json_slice(raw or ''))
    except (ValidationError, ValueError) as exc:
        # 부분·퍼지 복구는 하지 않는다 — 그럴듯한 복구는 실패를 지표에서 숨긴다.
        logger.warning('구조화 출력 검증 실패 (%s) raw=%r', model_cls.__name__, (raw or '')[:200])
        if span is not None:
            otel.set_attrs(span, {'kms.schema_invalid': True})
        raise LlmJudgmentFailed(f'{model_cls.__name__} 구조화 출력 검증 실패') from exc
