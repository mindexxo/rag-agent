"""오정보 2단계 판정(#214) — 가짜 LLM으로 결합 규칙만 고정한다. 실제 판정 품질은 eval/misinfo_judge.selftest."""
import json

import pytest

from eval.misinfo_judge import FIXTURES, confirm_violations
from rag.llm_schemas import LlmJudgmentFailed


class _FakeLlm:
    def __init__(self, labels: dict[str, str], fail: set[str] = frozenset()):
        self.labels, self.fail, self.calls = labels, fail, 0

    async def acomplete(self, messages, extra_body=None):
        self.calls += 1
        phrase = messages[-1]["content"].split("\n", 1)[0].removeprefix("금지 주장: ")
        if phrase in self.fail:
            raise LlmJudgmentFailed("judge down")
        return json.dumps({"label": self.labels[phrase], "reason": "fake"}, ensure_ascii=False)


@pytest.mark.asyncio
async def test_빈_후보면_LLM을_부르지_않는다():
    llm = _FakeLlm({})
    assert await confirm_violations(llm, "답변", []) == ([], []) and llm.calls == 0


@pytest.mark.asyncio
async def test_부정_정정은_빠지고_단정만_남는다():
    llm = _FakeLlm({"4만원 이상 주문에만 사용": "negated_or_corrected", "근로시간으로 인정되지 않": "asserted"})
    confirmed, audit = await confirm_violations(llm, "답변", ["4만원 이상 주문에만 사용", "근로시간으로 인정되지 않"])
    assert confirmed == ["근로시간으로 인정되지 않"]
    assert [a["label"] for a in audit] == ["negated_or_corrected", "asserted"] and llm.calls == 2


@pytest.mark.asyncio
async def test_판정_실패는_위반_유지():
    llm = _FakeLlm({"a": "negated_or_corrected"}, fail={"a"})
    confirmed, audit = await confirm_violations(llm, "답변", ["a"])
    assert confirmed == ["a"] and audit[0]["error"].startswith("LlmJudgmentFailed")


def test_FIXTURES는_오탐_3_진짜_1():
    assert [f["expected"] for f in FIXTURES].count("asserted") == 1 and len(FIXTURES) == 4
