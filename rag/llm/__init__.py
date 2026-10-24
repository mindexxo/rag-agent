"""LLM 공급자 — **정의점** (#287). 어떤 벤더를 쓸지는 `create_llm()` 한 곳에서만 정해진다.

    from rag.llm import LlmClient, CitationTail, create_llm

    llm = create_llm()                                   # settings.llm_provider로 고른 구현
    raw = await llm.acomplete(messages, schema=RouteDecision)

구성:
- base.py           공통 계약(LlmClient·CitationTail·FinishReason·ConstraintUnsupported)
- openai_compat.py  OpenAI SDK 공통 바탕(접속·스트리밍·종료 사유)
- vllm.py           사내 vLLM(확장 인자로 JSON·꼬리 강제)
- litellm.py        상용 벤더 3종(gemini·openai·anthropic) — LiteLLM이 벤더 문법 번역, 캐싱 표시·usage 포함

벤더를 추가하려면: 구현 파일 하나 + 아래 `_PROVIDERS`에 한 줄 + config.py `llm_provider`의 Literal에 이름 하나.
호출부는 바뀌지 않는다.
"""
from collections.abc import Callable

from config import settings
from rag.llm.base import CitationTail, ConstraintUnsupported, FinishReason, LlmClient, LlmUsage
from rag.llm.litellm import LiteLlmClient
from rag.llm.vllm import VllmClient

__all__ = ["CitationTail", "ConstraintUnsupported", "FinishReason", "LlmClient", "LlmUsage", "create_llm"]

_PROVIDERS: dict[str, "Callable[[], LlmClient]"] = {
    "vllm": VllmClient,
    "gemini": lambda: LiteLlmClient("gemini"),
    "openai": lambda: LiteLlmClient("openai"),
    "anthropic": lambda: LiteLlmClient("anthropic"),
}


def create_llm(provider: str | None = None) -> LlmClient:
    """설정(또는 인자)의 공급자 구현을 만든다. 모르는 이름은 즉시 실패 — 엉뚱한 기본값으로 붙지 않게."""
    name = provider or settings.llm_provider
    try:
        return _PROVIDERS[name]()
    except KeyError:
        raise ValueError(f"알 수 없는 LLM 공급자: {name!r} (가능: {sorted(_PROVIDERS)})") from None
