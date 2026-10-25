"""LLM 공급자 — **정의점** (#287). 어떤 벤더를 쓸지는 `create_llm()` 한 곳에서만 정해진다.

    from rag.llm import LlmClient, CitationTail, create_llm

    llm = create_llm()                                   # settings.llm_provider로 고른 구현
    aux = create_aux_llm()                               # 보조 호출(분류·재작성·캐시 판정)용 — 모델·생각 수준만 다를 수 있다(#297)
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
    """설정(또는 인자)의 공급자 구현을 만든다. 모르는 이름은 즉시 실패 — 엉뚱한 기본값으로 붙지 않게.
    model·reasoning_effort를 안 주면 요청에 생성 설정(llm_model·llm_reasoning_effort)이 실린다 — ""는 '설정을 따름'."""
    name = provider or settings.llm_provider
    try:
        factory = _PROVIDERS[name]
    except KeyError:
        raise ValueError(f"알 수 없는 LLM 공급자: {name!r} (가능: {sorted(_PROVIDERS)})") from None
    return factory(model=model, reasoning_effort=reasoning_effort)


def aux_overridden() -> bool:
    """보조 호출 설정(#297)이 하나라도 있는가 — 없으면 보조 호출도 생성 클라이언트를 그대로 쓴다(요청이 이전과 같다)."""
    return bool(settings.llm_aux_model or settings.llm_aux_reasoning_effort)


def create_aux_llm() -> LlmClient:
    """보조 호출(의도 분류·질의 재작성·캐시 재사용 판정)용 클라이언트(#297). 같은 공급자·주소·키·온도에 모델·생각 수준만
    llm_aux_*로 바꾼다 — 비어 있는 쪽은 생성 값을 따른다."""
    return create_llm(model=settings.llm_aux_model or None, reasoning_effort=settings.llm_aux_reasoning_effort)
