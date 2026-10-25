"""OpenAI SDK 공통 바탕 — OpenAI 호환 API를 쓰는 구현(vLLM, 이후 GPT·Gemini 호환 창구)이 공유한다 (#287 결정 3).

접속·스트리밍·종료 사유 처리처럼 까다롭고 벤더 무관한 코드는 여기 한 벌만 둔다. 구현마다 다른 것은
**의도(schema·citations)를 요청 인자로 바꾸는 방법** 하나다 — 하위 클래스가 `_constraint_params`를 고친다.
바탕은 제약을 모르므로, 하위 클래스가 고치지 않았는데 제약이 오면 ConstraintUnsupported다(base.py의 원칙).
"""
from collections.abc import AsyncIterator, Callable

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel

from config import settings
from rag.llm.base import CitationTail, ConstraintUnsupported, FinishReason, LlmUsage

# 벤더 종료 사유 → 공통 어휘. 모르는 값은 'other'(content_filter·abort 등).
_FINISH_REASONS: dict[str, FinishReason] = {"stop": "stop", "length": "length"}


def normalize_finish_reason(raw: str) -> FinishReason:
    return _FINISH_REASONS.get(raw, "other")


class OpenAiCompatClient:
    """OpenAI SDK 위의 공통 구현. 단독으로 쓰면 제약 없는 평범한 OpenAI 호환 호출이다."""

    def __init__(self, *, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 reasoning_effort: str | None = ""):
        self._model = model or settings.llm_model
        # ""면 요청 시점의 settings.llm_reasoning_effort를 따른다(생성 클라이언트·기본). 보조 클라이언트(#297)만 값을 넘긴다
        self._reasoning_effort = reasoning_effort
        self._client = AsyncOpenAI(
            base_url=base_url or settings.llm_base_url,
            api_key=api_key or settings.llm_api_key,
            # read=300: 토큰 간격이 5분 넘으면 stall로 보고 끊음(정상 스트리밍은 토큰 <1s 간격이라 무영향).
            # 기본 read=600(10분)은 서버 멈춤 시 자원을 너무 오래 물어서 단축.
            timeout=httpx.Timeout(300.0, connect=5.0),
        )

    @property
    def model_name(self) -> str:
        """요청에 실리는 모델 이름 — 스팬 llm.model_name 기록용(#297)."""
        return self._model

    @property
    def reasoning_effort(self) -> str | None:
        """이 클라이언트의 생각 수준 — 생성 클라이언트는 설정을 요청 시점에 읽는다(테스트가 설정을 바꿔 끼울 수 있게)."""
        return settings.llm_reasoning_effort if self._reasoning_effort == "" else self._reasoning_effort

    # ── 하위 클래스가 고치는 곳 ─────────────────────────────

    def _constraint_params(self, schema: type[BaseModel] | None,
                           citations: CitationTail | None) -> dict:
        """의도 → 요청 인자. 바탕은 어떤 제약도 강제할 줄 모른다."""
        if schema is not None or citations is not None:
            raise ConstraintUnsupported(f"{type(self).__name__}: schema·citations 제약을 강제할 수 없다")
        return {}

    # ── 공통 ───────────────────────────────────────────────

    def _request_kwargs(self, schema: type[BaseModel] | None, citations: CitationTail | None) -> dict:
        """chat.completions.create 인자. temperature는 설정됐을 때만 보낸다(None = 서버 기본값)."""
        kwargs = {
            "model": self._model,
            "max_tokens": settings.generation_reserve_tokens,   # 생성 몫 예약 + 폭주 방지 (F100)
            **self._constraint_params(schema, citations),
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        return kwargs

    async def acomplete(self, messages: list[dict], *, schema: type[BaseModel] | None = None,
                        citations: CitationTail | None = None,
                        on_usage: Callable[[LlmUsage], None] | None = None) -> str:
        # on_usage는 받되 부르지 않는다 — usage를 받으려면 요청 인자(stream_options)가 바뀌는데 vLLM 요청 바디는
        # #287이 고정했고 사내 서버는 비용 검증 대상이 아니다. 상용 구현(rag/llm/litellm.py)만 올린다.
        response = await self._client.chat.completions.create(
            messages=messages, stream=False, **self._request_kwargs(schema, citations))
        return response.choices[0].message.content

    async def astream(self, messages: list[dict], *, schema: type[BaseModel] | None = None,
                      citations: CitationTail | None = None,
                      on_finish_reason: Callable[[FinishReason], None] | None = None,
                      on_usage: Callable[[LlmUsage], None] | None = None) -> AsyncIterator[str]:
        """스트리밍 호출. on_usage는 받되 부르지 않는다(acomplete와 같은 이유).

        on_finish_reason(#129): 마지막 청크는 delta 없이 finish_reason만 실려 와서 예전 `if delta:` 가드가
        통째로 삼켰다 — 토큰 상한에 잘려도(length) 호출부가 알 길이 없었다. 반환이 AsyncIterator[str]라
        값을 실을 통로가 없어 콜백으로 뺐다. 콜백은 호출마다 새로 받으므로 공용 싱글톤이어도 동시 요청 간
        값이 섞이지 않는다(인스턴스 속성 방식을 피한 이유).
        """
        stream = await self._client.chat.completions.create(
            messages=messages, stream=True, **self._request_kwargs(schema, citations))
        try:
            async for chunk in stream:
                choice = chunk.choices[0]
                if choice.delta.content:
                    yield choice.delta.content
                if choice.finish_reason and on_finish_reason:
                    on_finish_reason(normalize_finish_reason(choice.finish_reason))
        finally:
            # 정상완료·예외(타임아웃 등)·소비 중단 어느 경우든 연결을 닫아 버려진 생성이 서버 자원을
            # 물지 않게 한다 (P2 llm.py:44). close 실패가 본 결과를 가리지 않게 삼킨다.
            try:
                await stream.close()
            except Exception:
                pass
