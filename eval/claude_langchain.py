"""ClaudeCliClient를 RAGAS judge로 쓰기 위한 langchain 어댑터 (#103).

부재단정 judge(eval/absence_judge.py)는 acomplete 시그니처만 맞으면 되어 어댑터가 필요
없지만(ClaudeCliClient를 그대로 주입), RAGAS는 judge에게 langchain BaseLanguageModel
인터페이스(agenerate_prompt)를 요구한다 — 그래서 BaseChatModel을 상속한 얇은 어댑터가 필요.

RAGAS evaluate()는 내부 이벤트 루프에서 항상 async 경로(agenerate_text)만 타므로
_agenerate만 네이티브로 구현하면 된다 — asyncio.run 같은 루프 브리징이 불필요하고,
ClaudeCliClient.acomplete가 이미 create_subprocess_exec 기반이라 RAGAS 루프에 자연 귀속된다.
_generate(동기)는 RAGAS가 안 부르므로 스텁이다.

**주의(#103 각주감)**:
- temperature 미노출 — claude judge는 CLI 기본 온도로 돈다(vLLM judge의 0.2와 다름).
  커스텀 모델에 temperature 속성을 두지 않아 RAGAS의 강제 주입이 자동 무시된다.
- guided decoding 없음 — RAGAS의 JSON 복구 재시도(FixOutputFormat, 프롬프트당 최대 3회)가
  vLLM보다 자주 발동할 수 있다. 콜 수·소요가 크게 늘 수 있어 스모크 규모로만 쓴다.
- 서브프로세스라 콜=프로세스 — RunConfig(max_workers)를 낮춰야 한다(ragas_eval.py에서 조정).
"""
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.outputs import ChatGeneration, ChatResult

from eval.claude_client import ClaudeCliClient


class ClaudeCliChatModel(BaseChatModel):
    """claude -p 헤드리스를 감싼 최소 langchain chat model — RAGAS judge 전용."""

    client: ClaudeCliClient
    model_config = {"arbitrary_types_allowed": True}   # ClaudeCliClient는 pydantic 모델이 아님

    @property
    def _llm_type(self) -> str:
        return "claude-cli"

    async def _agenerate(self, messages: list[BaseMessage], stop=None,
                         run_manager=None, **kwargs: Any) -> ChatResult:
        # BaseMessage(system/human/ai) → {"role","content"} — ClaudeCliClient가 받는 형태.
        dict_messages = convert_to_openai_messages(messages)
        text = await self.client.acomplete(dict_messages)   # extra_body 없음 = 스키마 강제 불가
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _generate(self, messages: list[BaseMessage], stop=None,
                  run_manager=None, **kwargs: Any) -> ChatResult:
        # RAGAS evaluate()는 async만 탄다 — 동기 경로는 호출되지 않는다(추상 요건 충족용 스텁).
        raise NotImplementedError("RAGAS judge는 async(_agenerate)만 사용한다")
