"""LLM 클라이언트.

Ollama / vLLM의 OpenAI 호환 API를 래핑.
RagService는 이 클라이언트만 바라보며, 실제 서버가 무엇인지 알 필요 없음.
URL/모델명은 config에서 주입 → 환경 변수만 바꾸면 Ollama ↔ vLLM 전환 가능.
"""
from collections.abc import AsyncIterator

import httpx
from openai import AsyncOpenAI
from config import settings

class LlmClient:
    def __init__(self):
        self._client = AsyncOpenAI(
            base_url=settings.vllm_base_url,
            api_key="EMPTY",
            # read=300: 토큰 간격이 5분 넘으면 stall로 보고 끊음(정상 스트리밍은 토큰 <1s 간격이라 무영향).
            # 기본 read=600(10분)은 vLLM 멈춤 시 GPU를 너무 오래 물어서 단축.
            timeout=httpx.Timeout(300.0, connect=5.0),
        )

    def _base_kwargs(self) -> dict:
        """공통 호출 인자. temperature는 설정됐을 때만 전달 (None = 서버 기본값)."""
        kwargs = {
            "model": settings.vllm_model,
            "max_tokens": settings.generation_reserve_tokens,   # 생성 몫 예약 + 폭주 방지 (F100)
            "extra_body": {"chat_template_kwargs": {"enable_thinking": settings.llm_enable_thinking}},
        }
        if settings.llm_temperature is not None:
            kwargs["temperature"] = settings.llm_temperature
        return kwargs

    async def acomplete(self, messages: list[dict]) -> str:
        """비스트리밍 호출. 응답 전체를 문자열로 반환.

        condense처럼 결과를 바로 파싱해야 할 때 사용.
        """
        response = await self._client.chat.completions.create(
            messages=messages,
            stream=False,
            **self._base_kwargs(),
        )
        return response.choices[0].message.content

    async def astream(self, messages: list[dict]) -> AsyncIterator[str]:
        """스트리밍 호출. vLLM이 토큰을 만들어내는 족족 yield."""
        stream = await self._client.chat.completions.create(
            messages=messages,
            stream=True,
            **self._base_kwargs(),
        )
        try:
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        finally:
            # 정상완료·예외(타임아웃 등)·소비 중단 어느 경우든 vLLM 연결을 닫아
            # 버려진 생성이 GPU를 물지 않게 한다 (P2 llm.py:44).
            # close 실패가 본 결과(정상 생성분)를 가리지 않게 삼킨다 — cleanup은 결과를 깨면 안 됨.
            try:
                await stream.close()
            except Exception:
                pass