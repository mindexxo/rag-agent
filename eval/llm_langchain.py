"""LlmClient(rag/llm 계약)를 RAGAS judge로 쓰기 위한 langchain 어댑터 (#103 → #290 일반화).

원래 Claude CLI 전용 어댑터(#103, 옛 파일명 claude_langchain)였는데 실제로 부르는 건 `client.acomplete(messages)` 하나라,
설정 공급자(create_llm — vLLM·gemini·openai·anthropic)와 Claude CLI(eval/claude_client.py)를 같은 어댑터로
감싼다(#290 결정 2). 분기는 eval/ragas_eval.py의 _judge에만 있다.

부재단정 judge(eval/absence_judge.py)는 acomplete 시그니처만 맞으면 되어 어댑터가 필요 없지만,
RAGAS는 judge에게 langchain BaseLanguageModel 인터페이스(agenerate_prompt)를 요구한다 — 그래서
BaseChatModel을 상속한 얇은 어댑터가 필요.

RAGAS evaluate()는 내부 이벤트 루프에서 항상 async 경로(agenerate_text)만 타므로
_agenerate만 네이티브로 구현하면 된다 — asyncio.run 같은 루프 브리징이 불필요하고,
acomplete가 이미 async(HTTP 또는 create_subprocess_exec)라 RAGAS 루프에 자연 귀속된다.
_generate(동기)는 RAGAS가 안 부르므로 스텁이다.

**주의(#103 각주감)**:
- temperature는 클라이언트 몫 — 설정 공급자는 LLM_TEMPERATURE, Claude CLI는 CLI 기본 온도로 돈다.
  커스텀 모델에 temperature 속성을 두지 않아 RAGAS의 강제 주입이 자동 무시된다(ragas_eval은 bypass_temperature도 켠다).
- 제약 의도(schema)를 안 넘긴다 — RAGAS의 JSON 복구 재시도(FixOutputFormat, 프롬프트당 최대 3회)가
  발동할 수 있다. Claude CLI는 서브프로세스라 콜=프로세스 — RunConfig(max_workers)를 낮춘다(ragas_eval.py).
"""

from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult


class LlmClientChatModel(BaseChatModel):
    """LlmClient(acomplete)를 감싼 최소 langchain chat model — RAGAS judge 전용."""

    client: Any  # rag/llm의 LlmClient 계약(Protocol이라 pydantic isinstance 검사 불가 — Any로 둔다)
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "llm-client"

    async def _agenerate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        # BaseMessage(system/human/ai) → {"role","content"} — LlmClient.acomplete가 받는 형태.
        dict_messages = convert_to_openai_messages(messages)
        text = await self.client.acomplete(dict_messages)  # 제약 의도 없음 = 스키마 강제 불가
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        # RAGAS evaluate()는 async만 탄다 — 동기 경로는 호출되지 않는다(추상 요건 충족용 스텁).
        raise NotImplementedError("RAGAS judge는 async(_agenerate)만 사용한다")
