"""사내 vLLM 구현 — OpenAI 호환 API + vLLM 확장 인자 (#287).

vLLM 전용인 것은 전부 이 파일에 있다: thinking 토글(chat_template_kwargs), JSON 강제(structured_outputs),
출처 꼬리 강제(structural_tag). 셋 다 OpenAI SDK의 extra_body로 실려 요청 바디에 병합된다.
요청 바디는 #287 전(단일 모듈 시절의 클라이언트·llm_schemas·prompts가 만들던 것)과 같다(tests/test_llm_providers.py가 고정).
"""
from functools import cache

from pydantic import BaseModel

from config import settings
from rag.llm.base import CitationTail
from rag.llm.openai_compat import OpenAiCompatClient


class VllmClient(OpenAiCompatClient):

    def _constraint_params(self, schema: type[BaseModel] | None,
                           citations: CitationTail | None) -> dict:
        # thinking 토글은 매 요청에 실린다 — 덮어쓰면 유실되므로 제약과 같은 extra_body에 병합한다.
        # thinking 토글은 공통 설정 llm_reasoning_effort에서 파생한다(#288 결정 5) — vLLM(Qwen3)은 on/off만 있다
        thinking = settings.llm_reasoning_effort not in (None, "none")
        extra_body = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if schema is not None:
            extra_body |= _schema_body(schema)
        if citations is not None:
            extra_body |= _citation_body(citations)
        return {"extra_body": extra_body}


@cache
def _schema_body(model_cls: type[BaseModel]) -> dict:
    """JSON 스키마 강제 — vLLM v0.12+ structured_outputs 형식. 스키마는 런타임 값이 없는 정적이라 캐시."""
    return {"structured_outputs": {"json": model_cls.model_json_schema()}}


def _citation_body(tail: CitationTail) -> dict:
    """출처 꼬리 안쪽을 유효 후보 번호로 강제한다 (#56, #65 재구현).

    vLLM의 `response_format` + `structural_tag`다. `triggers`에 든 문자열이 생성 도중 나타나면 그 지점부터
    `end`까지 `schema`를 강제한다 — **본문은 자유, 꼬리만 제약**이 필요한 우리 경우에 정확히 맞는 기능이다.
    그래서 꼬리 안쪽이 JSON 정수 배열이 된다(형식 상수는 rag/prompt_texts.py의 TAIL_EXAMPLE_* —
    프롬프트 예시와 같은 정의점).

    ⚠ **정규식으로 되돌리지 말 것.** #56의 원래 구현은 `[\\s\\S]*««1|2»»` 형태의 정규식이었고
    vLLM 0.24.0에서 **강제되지 않았다**(2026-08-20 실측, #61에서 발견 → #65). 파라미터는
    받아들여지고 예외도 안 나는데 제약이 안 걸린다. 사다리 실측:

        ««(?:1)?»»              강제됨   ← 제약이 문자열 시작 위치
        ««(?:1)?»»[\\s\\S]*      강제됨   ← 제약이 앞, 수량자는 뒤
        [\\s\\S]*««(?:1)?»»      무효    ← 옛 구현의 형태
        [^«]*««(?:1)?»»         무효
        [\\s\\S]{0,200}««…»»     무효    ← 유계로 바꿔도 마찬가지
        + / .* / [^«»]*          무효

    규칙: 앞에 "아무 문자" 구간이 오면 뒤쪽 제약이 무효다(유한·무한 무관). 이건 서버 버전
    문제가 아니라 "꼬리가 답변 **끝**에 온다"는 구조와 정규식 가드 디코딩의 충돌이라 재현성이
    있다 — 백엔드 3종(xgrammar·guidance·outlines)과 GBNF까지 전부 같은 결과였다.

    structural_tag가 강제된다는 근거(반증 테스트): 후보가 1개뿐인 문맥에 enum=[2]를 주면
    모델이 3/3 `[2]`를 쓴다 — 문맥상 말이 안 되는 값을 강제로 쓴 것이다. 스트리밍
    (astream, 운영 경로)에서도 강제된다. 옛 정규식은 같은 문항에서 5/5 위반했다.

    후보 0개면 `maxItems: 0`으로 빈 배열만 허용한다. `enum: []`을 쓰지 않은 이유: 선택지 0개 enum을
    백엔드가 어떻게 컴파일하는지 미확인이라, 검증된 형태를 쓴다.

    거절 답변도 꼬리는 붙는다 — 빈 배열로. 후보가 있어도 모델이 빈 배열을 낼 수 있다(minItems 미지정) —
    답할 수 없는 질문에 후보 4개를 주고 확인, 2/2 `[]`. 즉 강제는 "없는 번호를 못 쓰게" 할 뿐 거절의
    자유를 빼앗지 않는다.

    서버가 이 인자를 거부하면(400) 그대로 에러다 — #287에서 제약을 빼고 재시도하던 방어를 지웠다
    (사내 서버는 지원하는 버전이라 그 경로가 쓰이지 않았다, 사용자 결정 8).
    """
    schema = ({"type": "array", "items": {"type": "integer", "enum": list(range(1, tail.count + 1))}}
              if tail.count else {"type": "array", "maxItems": 0})
    return {"response_format": {
        "type": "structural_tag",
        "triggers": [tail.start],
        "structures": [{"begin": tail.start, "schema": schema, "end": tail.end}],
    }}
