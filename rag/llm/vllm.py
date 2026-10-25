"""사내 vLLM 구현 — thinking 토글·JSON 강제·출처 꼬리 강제를 vLLM 확장 인자(extra_body)로 싣는다.

요청 바디는 tests/test_llm_providers.py가 고정한다.
"""

from functools import cache

from pydantic import BaseModel

from rag.llm.base import CitationTail
from rag.llm.openai_compat import OpenAiCompatClient


class VllmClient(OpenAiCompatClient):
    def _constraint_params(self, schema: type[BaseModel] | None, citations: CitationTail | None) -> dict:
        # thinking 토글은 매 요청에 실린다 — 덮어쓰면 유실되므로 제약과 같은 extra_body에 병합한다.
        # vLLM(Qwen3)은 on/off만 있어 공통 설정 reasoning_effort에서 파생한다
        thinking = self.reasoning_effort not in (None, "none")
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if schema is not None:
            extra_body |= _schema_body(schema)
        if citations is not None:
            extra_body |= _citation_body(citations)
        return {"extra_body": extra_body}


@cache
def _schema_body(model_cls: type[BaseModel]) -> dict:
    """JSON 스키마 강제 — vLLM structured_outputs 형식. 스키마는 정적이라 캐시."""
    return {"structured_outputs": {"json": model_cls.model_json_schema()}}


def _citation_body(tail: CitationTail) -> dict:
    """출처 꼬리 안쪽을 유효 후보 번호의 JSON 정수 배열로 강제한다(structural_tag — 본문은 자유, 꼬리만 제약).

    ⚠ 정규식으로 되돌리지 말 것 — 앞에 "아무 문자" 구간이 오는 정규식은 서버가 받아들이고도 강제하지 않는다
    (꼬리가 답변 끝에 오는 구조와 정규식 가드 디코딩의 충돌이라 백엔드를 바꿔도 같다).
    강제 여부의 반증: 후보가 1개인 문맥에 enum=[2]를 주고 꼬리에 2가 나오는지 본다 — 강제되면 1로 나온다.

    후보 0개면 `enum: []` 대신 검증된 `maxItems: 0`으로 빈 배열만 허용한다. minItems가 없어 후보가 있어도
    빈 배열(거절)을 낼 수 있다. 서버가 이 인자를 거부하면(400) 그대로 에러다.
    """
    schema = (
        {"type": "array", "items": {"type": "integer", "enum": list(range(1, tail.count + 1))}}
        if tail.count
        else {"type": "array", "maxItems": 0}
    )
    return {
        "response_format": {
            "type": "structural_tag",
            "triggers": [tail.start],
            "structures": [{"begin": tail.start, "schema": schema, "end": tail.end}],
        }
    }
