"""상용 벤더 구현 — LiteLLM 경유 (#288). Gemini·OpenAI(GPT)·Anthropic(Claude)을 파일 하나로.

LiteLLM이 하는 일: OpenAI 모양 호출 하나를 벤더 문법(요청·응답·스트리밍 청크·종료 사유·usage)으로 번역한다.
우리가 하는 일은 "의도 → LiteLLM 인자" 한 단계뿐이다(아래 _request_kwargs). 벤더별로 갈리는 것은 전부
`_VENDORS` 표 한 곳에 둔다.

의도 처리 (#288 결정):
- schema=   → JSON 모드(`response_format={"type": "json_object"}`)만 요청한다(결정 3 = b). 스키마 강제가 아니라
              형식 유도이고, 신뢰 근거는 호출부의 Pydantic 검증(rag/llm_schemas)이다. 검증 실패율을 측정해 문제면
              벤더별 스키마 다듬기(a)로 올린다. Anthropic은 JSON 모드가 없어 LiteLLM이 도구 호출로 흉내 낸다 —
              실제 수용 여부는 키 확보 후 확인.
- citations= → **디코딩 강제를 하지 않는다**(결정 2 = a). 세 벤더 어디에도 "본문은 자유, 꼬리만 제약"(vLLM
              structural_tag)이 없다. 꼬리 형식은 프롬프트 규칙으로 유도하고 rag/citation_tail 파서가 범위 밖 번호를
              버린다 — 8월 비교에서 제약 없는 Claude의 인용 정확도가 제약 걸린 14B보다 높았다(0.994 vs 0.969).
              base.py의 "조용히 무시 금지" 원칙과의 관계: 무시가 아니라 **파서 검증으로 충족**한다는 합의이고,
              벤더별 꼬리 위반율은 #288 측정 항목이다.
- 캐싱     → prefix 캐싱은 세 벤더 모두 쓰되 **표시는 Anthropic에만** 붙인다(결정 4, 10/24 수정). Anthropic은
              `cache_control: ephemeral` 표시가 없으면 캐싱이 안 된다. OpenAI·Gemini(암시적 캐싱, 2.5 Flash 최소 2,048토큰)는
              자동이라 아무것도 안 보낸다 — Gemini에 표시를 보내면 LiteLLM이 **명시적 컨텍스트 캐싱**(캐시 객체 생성,
              저장 시간당 과금, 무료 티어 한도 0 → 429, 최신 모델 미지원)으로 번역해 버린다(라이브 실측). 최소 길이 미달이면
              조용히 캐시가 안 될 뿐 에러는 없다. 적중 여부는 on_usage의 cached_tokens로 본다.
- temperature·reasoning_effort → 설정값을 그대로 보낸다(결정 5). 벤더가 거부하는 조합(GPT-5 추론 모델의 temperature≠1,
              Claude 4.7+의 temperature<1)은 400으로 바로 드러나게 둔다 — 조용히 빼지 않는다. 표는 config.py 주석.
"""
from collections.abc import AsyncIterator, Callable

from pydantic import BaseModel

from config import settings
from rag.llm.base import CitationTail, FinishReason, LlmUsage

# 벤더표 — LiteLLM 모델 접두 · 캐시 표시 필요 여부(모듈 docstring '캐싱'). 키·모델명은 공통 설정 하나다(결정 6).
_VENDORS: dict[str, dict] = {
    "gemini":    {"prefix": "gemini",    "cache_mark": False},   # 암시적 캐싱 자동. 표시하면 명시적 캐싱으로 번역돼 과금·429
    "openai":    {"prefix": "openai",    "cache_mark": False},   # 자동 캐싱. 표시는 무시되지만 보낼 이유가 없다
    "anthropic": {"prefix": "anthropic", "cache_mark": True},    # 표시가 없으면 캐싱 안 됨
}

_FINISH: dict[str, FinishReason] = {"stop": "stop", "length": "length"}   # LiteLLM은 벤더 종료 사유를 OpenAI 어휘로 맞춰 준다

CACHE_MARK = {"type": "ephemeral"}


def with_cache_mark(messages: list[dict]) -> list[dict]:
    """system 메시지를 내용 블록으로 바꿔 캐시 표시를 붙인다. 나머지는 그대로(얕은 복사 — 호출부 메시지를 안 바꾼다)."""
    out = []
    for m in messages:
        if m.get("role") == "system" and isinstance(m.get("content"), str):
            out.append({"role": "system",
                        "content": [{"type": "text", "text": m["content"], "cache_control": CACHE_MARK}]})
        else:
            out.append(m)
    return out


def _usage_of(obj) -> LlmUsage | None:
    """LiteLLM 응답·청크의 usage → LlmUsage. 캐시 적중 토큰은 OpenAI 모양(prompt_tokens_details.cached_tokens)."""
    usage = getattr(obj, "usage", None)
    if not usage:
        return None
    details = getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", None) if details else None
    return LlmUsage(prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    cached_tokens=cached or 0,
                    cache_write_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0)


class LiteLlmClient:
    def __init__(self, vendor: str, *, model: str | None = None, api_key: str | None = None,
                 reasoning_effort: str | None = ""):
        if vendor not in _VENDORS:
            raise ValueError(f"LiteLLM 벤더가 아니다: {vendor!r} (가능: {sorted(_VENDORS)})")
        self.vendor = vendor
        self._model = f"{_VENDORS[vendor]['prefix']}/{model or settings.llm_model}"
        self._api_key = api_key or settings.llm_api_key
        self._reasoning_effort = reasoning_effort   # ""면 요청 시점의 설정을 따른다(OpenAiCompatClient와 같은 규칙, #297)

    @property
    def model_name(self) -> str:
        """요청에 실리는 모델 이름(LiteLLM 접두 포함) — 스팬 llm.model_name 기록용(#297)."""
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
            "max_tokens": settings.generation_reserve_tokens,   # 생성 몫 예약 + 폭주 방지 (F100) — vLLM과 같은 값
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        if self.reasoning_effort is not None:
            kwargs["reasoning_effort"] = self.reasoning_effort
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}   # 결정 3 = b — 스키마 강제 아님, 검증은 Pydantic
        # citations는 보내지 않는다(모듈 docstring) — 꼬리는 프롬프트 + 파서
        return kwargs

    async def acomplete(self, messages: list[dict], *, schema: type[BaseModel] | None = None,
                        citations: CitationTail | None = None,
                        on_usage: Callable[[LlmUsage], None] | None = None) -> str:
        import litellm   # 지연 import — 118MB 패키지라 vLLM만 쓰는 프로세스(워커 등)가 비용을 안 내게
        response = await litellm.acompletion(messages=self._messages(messages), stream=False,
                                             **self._request_kwargs(schema, citations))
        if on_usage and (u := _usage_of(response)):
            on_usage(u)
        return response.choices[0].message.content

    async def astream(self, messages: list[dict], *, schema: type[BaseModel] | None = None,
                      citations: CitationTail | None = None,
                      on_finish_reason: Callable[[FinishReason], None] | None = None,
                      on_usage: Callable[[LlmUsage], None] | None = None) -> AsyncIterator[str]:
        import litellm
        kwargs = self._request_kwargs(schema, citations)
        # LiteLLM은 include_usage가 켜졌을 때만 스트림 끝에 usage 청크를 만든다(streaming_handler.send_stream_usage) — 벤더 무관.
        # 벤더가 이 인자를 몰라도 LiteLLM이 자체 처리하므로 거부되지 않는다(리뷰 지적 1, 소스 확인).
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
