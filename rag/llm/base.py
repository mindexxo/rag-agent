"""LLM 공통 계약 — **정의점** (#287). 호출부는 이 모듈의 이름만 안다.

호출부(서비스·가드·질의 재작성·캐시 판정·eval)는 LLM에게 **무엇을 원하는지**만 말한다.

    schema=RouteDecision          → "이 스키마의 JSON으로 답해"
    citations=CitationTail(...)   → "꼬리 표식 사이에는 1~N 정수 배열만 써"

이 의도를 서버가 알아듣는 문법(vLLM structured_outputs·structural_tag, OpenAI response_format,
Anthropic 구조화 출력 …)으로 바꾸는 것은 **각 구현의 일**이다. 벤더 문법이 호출부로 새어 나오면
벤더를 하나 추가할 때마다 호출부가 바뀐다 — #287 전에는 vLLM 딕셔너리를 호출부가 직접 조립했다.

⚠ **제약을 지원하지 못하는 구현은 조용히 무시하지 말고 ConstraintUnsupported를 낸다.**
서버가 제약 인자를 받아 놓고 에러 없이 강제하지 않는 일이 실제로 있었다(#61 — vLLM이 정규식 제약을
받아들이고 무시했다). Anthropic의 OpenAI 호환 창구도 response_format을 조용히 무시한다(공식 문서).
그래서 새 벤더나 서버 버전을 붙일 때는 제약이 **실제로 강제되는지 직접 확인**한다 — 제약이 없으면
반드시 어기게 되는 요청(후보 1개인 문맥에 2번만 허용 등)을 보내 결과를 보는 방식(#65의 반증 테스트).

0층(leaf)이다 — config 말고는 rag/ 모듈을 import하지 않는다. 그래서 꼬리 표식(rag/prompt_texts)도
값으로 받는다(CitationTail.start/end). 지표·추적 기록은 호출부 몫이고, 여기서는 값만 올려보낸다.
"""
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

# 생성 종료 사유 — 벤더 어휘를 이 셋으로 맞춘다(#287 결정 5). 지표 kms_llm_finish_reason_total의 라벨이 된다.
# 'length' = 출력 토큰 상한에 잘림(#129가 잡으려던 것). 벤더별 원래 값은 여기서 버린다 — 대시보드가 벤더를 몰라야 한다.
FinishReason = Literal["stop", "length", "other"]


@dataclass(frozen=True)
class CitationTail:
    """답변 끝 출처 꼬리 제약 — "start와 end 사이에는 1..count 정수 배열만 온다".

    count=0이면 빈 배열만 허용한다(후보 0개 — 운영은 즉시 경로라 도달하지 않고 eval에서 도달).
    후보 수를 세는 규칙(출처 다음 첨부)은 우리 도메인 규칙이라 호출부가 정한다(rag/prompts.build_citation_tail).
    """
    count: int
    start: str
    end: str


@dataclass(frozen=True)
class LlmUsage:
    """호출 하나의 토큰 사용량 — 비용 검증용(#288). cached_tokens가 0이면 프롬프트 캐싱이 안 먹은 것이다."""
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int = 0        # 캐시에서 읽은 입력 토큰(입력 단가의 0.1배)
    cache_write_tokens: int = 0   # 캐시에 새로 쓴 입력 토큰(Anthropic 1.25배)


class ConstraintUnsupported(Exception):
    """구현이 요청받은 제약(schema·citations)을 강제할 수 없다 — 조용히 무시하지 않으려고 낸다."""


class LlmClient(Protocol):
    """LLM이 갖춰야 할 모양. 인스턴스는 rag.llm.create_llm()이 설정을 보고 만든다."""

    @property
    def model_name(self) -> str:
        """요청에 실리는 모델 이름 — 스팬 llm.model_name 기록용(#297)."""
        ...

    async def acomplete(self, messages: list[dict], *, schema: type[BaseModel] | None = None,
                        citations: CitationTail | None = None,
                        on_usage: Callable[[LlmUsage], None] | None = None) -> str:
        """비스트리밍 — 응답 전체를 문자열로. usage는 on_usage로 올린다(구현이 못 주면 안 부른다)."""
        ...

    def astream(self, messages: list[dict], *, schema: type[BaseModel] | None = None,
                citations: CitationTail | None = None,
                on_finish_reason: Callable[[FinishReason], None] | None = None,
                on_usage: Callable[[LlmUsage], None] | None = None) -> AsyncIterator[str]:
        """스트리밍 — 텍스트 조각을 만드는 족족 내보낸다. 종료 사유·usage는 콜백으로 올린다(#129·#288 — 0층이라 기록은 호출부 몫)."""
        ...
