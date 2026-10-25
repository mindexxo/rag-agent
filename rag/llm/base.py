"""LLM 공통 계약의 **정의점** — 호출부는 의도(schema·citations)만 말하고, 벤더 문법으로 바꾸는 것은 각 구현의 일이다.

⚠ 제약을 강제하지 못하는 구현은 조용히 무시하지 말고 ConstraintUnsupported를 낸다 — 서버가 제약 인자를
받고도 에러 없이 강제하지 않는 경우가 있으니, 새 벤더·서버 버전은 제약이 없으면 반드시 어기게 되는 요청으로 직접 확인한다.

0층이라 꼬리 표식도 값으로 받고(CitationTail.start/end), 지표·추적 기록은 호출부 몫이다.
"""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

# 생성 종료 사유 — 벤더 어휘를 이 셋으로 맞춘다(지표 kms_llm_finish_reason_total의 라벨). 'length' = 출력 토큰 상한에 잘림.
FinishReason = Literal["stop", "length", "other"]


@dataclass(frozen=True)
class CitationTail:
    """답변 끝 출처 꼬리 제약 — "start와 end 사이에는 1..count 정수 배열만 온다".

    count=0이면 빈 배열만 허용한다. 후보 수를 세는 규칙은 호출부(rag/prompts.build_citation_tail)가 정한다.
    """

    count: int
    start: str
    end: str


@dataclass(frozen=True)
class LlmUsage:
    """호출 하나의 토큰 사용량. cached_tokens가 0이면 프롬프트 캐싱이 안 먹은 것이다."""

    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0  # 캐시에서 읽은 입력 토큰
    cache_write_tokens: int = 0  # 캐시에 새로 쓴 입력 토큰(Anthropic)


class ConstraintUnsupported(Exception):
    """구현이 요청받은 제약(schema·citations)을 강제할 수 없다."""


class LlmClient(Protocol):
    """LLM 구현의 계약. 인스턴스는 rag.llm.create_llm()이 만든다."""

    @property
    def model_name(self) -> str:
        """요청에 실리는 모델 이름 — 스팬 llm.model_name 기록용."""
        ...

    async def acomplete(
        self,
        messages: list[dict],
        *,
        schema: type[BaseModel] | None = None,
        citations: CitationTail | None = None,
        on_usage: Callable[[LlmUsage], None] | None = None,
    ) -> str:
        """usage는 on_usage로 올린다(구현이 못 주면 안 부른다)."""
        ...

    def astream(
        self,
        messages: list[dict],
        *,
        schema: type[BaseModel] | None = None,
        citations: CitationTail | None = None,
        on_finish_reason: Callable[[FinishReason], None] | None = None,
        on_usage: Callable[[LlmUsage], None] | None = None,
    ) -> AsyncIterator[str]:
        """텍스트 조각을 내보낸다. 종료 사유·usage는 콜백으로 올린다(반환 타입에 실을 자리가 없다)."""
        ...
