"""상용 벤더(Gemini·OpenAI·Anthropic) 구현 — LiteLLM이 벤더 문법을 번역하고, 여기서는 의도 → LiteLLM 인자만 한다.

벤더별로 갈리는 것은 `_VENDORS` 표 한 곳에 둔다.
- schema=    JSON 모드만 요청한다(스키마 강제 아님 — 신뢰 근거는 호출부의 Pydantic 검증).
- citations= 디코딩 강제를 하지 않는다(세 벤더에 꼬리만 제약하는 기능이 없다). 프롬프트로 유도하고
             rag/citation_tail 파서가 범위 밖 번호를 버린다 — 조용한 무시가 아니라 파서 검증으로 충족한다.
- 캐싱       표시는 Anthropic에만 붙인다(표시가 없으면 캐싱이 안 된다). OpenAI·Gemini는 자동이고, Gemini에 표시를
             보내면 LiteLLM이 과금되는 명시적 컨텍스트 캐싱으로 번역한다. 적중 여부는 on_usage의 cached_tokens로 본다.
- temperature·reasoning_effort  설정값을 그대로 보낸다 — 벤더가 거부하는 조합은 400으로 드러나게 둔다.
"""

from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from config import settings
from rag.llm.base import CitationTail, FinishReason, LlmUsage

# 벤더표 — LiteLLM 모델 접두 · 캐시 표시 필요 여부. 키·모델명은 공통 설정 하나다.
_VENDORS: dict[str, dict] = {
    "gemini": {"prefix": "gemini", "cache_mark": False},
    "openai": {"prefix": "openai", "cache_mark": False},
    "anthropic": {"prefix": "anthropic", "cache_mark": True},
}

_FINISH: dict[str, FinishReason] = {
    "stop": "stop",
    "length": "length",
}  # LiteLLM은 벤더 종료 사유를 OpenAI 어휘로 맞춰 준다

CACHE_MARK = {"type": "ephemeral"}


def with_cache_mark(messages: list[dict]) -> list[dict]:
    """system 메시지를 내용 블록으로 바꿔 캐시 표시를 붙인다. 호출부 메시지는 바꾸지 않는다."""
    out = []
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            out.append(
                {"role": "system", "content": [{"type": "text", "text": m["content"], "cache_control": CACHE_MARK}]}
            )
        else:
            out.append(m)
    return out


def _usage_of(obj) -> LlmUsage | None:
    """LiteLLM 응답·청크의 usage → LlmUsage. 캐시 적중 토큰은 OpenAI 모양(prompt_tokens_details.cached_tokens)이다."""
    usage = getattr(obj, "usage", None)
    if not usage:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    return LlmUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        cached_tokens=cached or 0,
        cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


class LiteLlmClient:
    def __init__(
        self, vendor: str, *, model: str | None = None, api_key: str | None = None, reasoning_effort: str | None = ""
    ):
        if vendor not in _VENDORS:
            raise ValueError(f"LiteLLM 벤더가 아니다: {vendor!r} (가능: {sorted(_VENDORS)})")
        self.vendor = vendor
        self._model = f"{_VENDORS[vendor]['prefix']}/{model or settings.llm_model}"
        self._api_key = api_key or settings.llm_api_key
        self._reasoning_effort = reasoning_effort  # ""면 요청 시점의 설정을 따른다(OpenAiCompatClient와 같은 규칙)

    @property
    def model_name(self) -> str:
        """요청에 실리는 모델 이름(LiteLLM 접두 포함) — 스팬 llm.model_name 기록용."""
        return self._model

    @property
    def reasoning_effort(self) -> str | None:
        return settings.llm_reasoning_effort if self._reasoning_effort == "" else self._reasoning_effort

    def _messages(self, messages: list[dict]) -> list[dict]:
        return with_cache_mark(messages) if _VENDORS[self.vendor]["cache_mark"] else messages

    def _request_kwargs(self, schema: type[BaseModel] | None, citations: CitationTail | None) -> dict:
        kwargs: dict = {
            "model": self._model,
            "api_key": self._api_key,
            "max_tokens": settings.generation_reserve_tokens,  # 생성 몫 예약 + 폭주 방지 — vLLM과 같은 값
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}  # 스키마 강제 아님, 검증은 Pydantic
        # citations는 보내지 않는다(모듈 docstring) — 꼬리는 프롬프트 + 파서
        return kwargs

    async def acomplete(
        self,
        messages: list[dict],
        *,
        schema: type[BaseModel] | None = None,
        citations: CitationTail | None = None,
        on_usage: Callable[[LlmUsage], None] | None = None,
    ) -> str:
        import litellm  # 지연 import — 무거운 패키지라 vLLM만 쓰는 프로세스가 비용을 안 내게

        response = await litellm.acompletion(
            messages=self._messages(messages), stream=False, **self._request_kwargs(schema, citations)
        )
        if on_usage and (u := _usage_of(response)):
            on_usage(u)
        return response.choices[0].message.content

    async def astream(
        self,
        messages: list[dict],
        *,
        schema: type[BaseModel] | None = None,
        citations: CitationTail | None = None,
        on_finish_reason: Callable[[FinishReason], None] | None = None,
        on_usage: Callable[[LlmUsage], None] | None = None,
    ) -> AsyncIterator[str]:
        import litellm

        kwargs = self._request_kwargs(schema, citations)
        # LiteLLM은 include_usage가 켜졌을 때만 스트림 끝에 usage 청크를 만든다. 벤더가 몰라도 LiteLLM이 자체 처리한다.
        kwargs["stream_options"] = {"include_usage": True}
        stream = await litellm.acompletion(messages=self._messages(messages), stream=True, **kwargs)
        async for chunk in stream:
            choice = chunk.choices[0] if chunk.choices else None
            if choice is not None:
                text = getattr(choice.delta, "content", None)
                if text:
                    yield text
                if choice.finish_reason and on_finish_reason:
                    on_finish_reason(_FINISH.get(choice.finish_reason, "other"))
            if on_usage and (u := _usage_of(chunk)):
                on_usage(u)
