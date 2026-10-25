"""LLM 공급자의 **정의점** — 어떤 벤더를 쓸지는 `create_llm()` 한 곳에서만 정해진다.

벤더 추가: 구현 파일 하나 + `_PROVIDERS`에 한 줄 + config.py `llm_provider`의 Literal에 이름 하나.
"""

from collections.abc import Callable

from config import settings
from rag.llm.base import CitationTail, ConstraintUnsupported, FinishReason, LlmClient, LlmUsage
from rag.llm.litellm import LiteLlmClient
from rag.llm.vllm import VllmClient

__all__ = [
    "CitationTail",
    "ConstraintUnsupported",
    "FinishReason",
    "LlmClient",
    "LlmUsage",
    "aux_overridden",
    "create_aux_llm",
    "create_llm",
]

_PROVIDERS: dict[str, "Callable[..., LlmClient]"] = {
    "vllm": VllmClient,
    "gemini": lambda **kw: LiteLlmClient("gemini", **kw),
    "openai": lambda **kw: LiteLlmClient("openai", **kw),
    "anthropic": lambda **kw: LiteLlmClient("anthropic", **kw),
}


def create_llm(
    provider: str | None = None, *, model: str | None = None, reasoning_effort: str | None = ""
) -> LlmClient:
    """공급자 구현을 만든다. 모르는 이름은 즉시 실패한다.
    model·reasoning_effort를 안 주면 생성 설정(llm_model·llm_reasoning_effort)을 따른다 — ""는 '설정을 따름'."""
    name = provider or settings.llm_provider
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"알 수 없는 LLM 공급자: {name!r} (가능: {sorted(_PROVIDERS)})") from None
    return factory(model=model, reasoning_effort=reasoning_effort)


def aux_overridden() -> bool:
    """보조 호출 설정이 하나라도 있는가 — 없으면 보조 호출도 생성 클라이언트를 그대로 쓴다."""
    return bool(settings.llm_aux_model or settings.llm_aux_reasoning_effort)


def create_aux_llm() -> LlmClient:
    """보조 호출(의도 분류·질의 재작성·캐시 재사용 판정)용 클라이언트. 공급자·주소·키·온도는 같고 모델·생각 수준만
    llm_aux_*로 바꾼다 — 비어 있는 쪽은 생성 값을 따른다."""
    return create_llm(model=settings.llm_aux_model or None, reasoning_effort=settings.llm_aux_reasoning_effort)
