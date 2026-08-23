"""gold 이력 어댑터 통합 회귀 (#77).

막으려는 버그 클래스: rag/conversation.py의 이력 함수가 **새 속성을 읽기 시작했는데**
eval 픽스처가 그 속성이 없는 객체를 만들어 AttributeError로 죽는 것. 실제로 두 번 났다
(#59 status 도입, #72 failed 분기 추가) — 두 번 다 사람이 우연히 발견했고 그 사이
생성축·멀티턴검색축이 정지해 있었다.

**어댑터 단위 검증만으로는 못 잡는다.** 크래시는 어댑터가 아니라 소비 함수 안쪽
(_history_content)에서 나기 때문이다. 그래서 실제 소비 경로 셋을 다 태운다 —
build_prior_turns · condense_query · condense_to_queries. 셋 다 _history_content를 공유한다.

**gold 파일을 읽지 않는다.** 사용자가 gold를 상시 편집 중이라(라벨 감사·케이스 추가)
무관한 데이터 변경에 이 테스트가 흔들린다. 대신 gold의 **계약**(role·content 두 키,
status 없음 — 2026-08 시점 608행 전수)을 인라인으로 고정한다. gold가 status를 갖게 되면
이 상수도 함께 갱신할 것.

DB·LLM 없음 — Message는 세션 없이 transient로만 쓰고, LLM은 conftest의 FakeLlm을 직접
인스턴스화한다(monkeypatch 픽스처가 필요 없는 순수 함수 호출이라).
"""
import pytest

from eval._gold_history import DEFAULT_STATUS, messages_from_conversation
from rag.conversation import build_prior_turns, condense_query, condense_to_queries
from rag.models import Message
from rag.prompt_texts import CANCELLED_TURN_EMPTY, CANCELLED_TURN_SUFFIX, FAILED_TURN_EMPTY
from tests.conftest import FakeLlm

# gold_set_v2.jsonl의 multi_turn 이력과 같은 모양 — role·content 두 키뿐이다.
_GOLD_TURN = [
    {'role': 'user', 'content': '단순변심 반품은 며칠까지 되나요?'},
    {'role': 'assistant', 'content': '상품 수령일로부터 14일 이내에 신청하시면 됩니다.'},
]


class TestAdapter:
    def test_진짜_Message를_돌려준다(self):
        """SimpleNamespace류 부분 객체로 퇴행하면 여기서 걸린다 — 그게 #77의 원인이었다."""
        assert all(isinstance(m, Message) for m in messages_from_conversation(_GOLD_TURN))

    def test_status가_없으면_운영_기본값으로_채운다(self):
        """transient Message는 컬럼 기본값이 안 걸려 None으로 남는다(flush 시점 기본값).
        None을 방치하면 비교가 `!= 'done'`으로 바뀌는 날 전 이력이 비정상 턴이 된다."""
        assert messages_from_conversation(_GOLD_TURN)[1].status == DEFAULT_STATUS

    def test_gold가_실은_status는_그대로_통과한다(self):
        """어댑터를 안 고치고도 gold가 취소·실패 턴을 표현할 수 있어야 한다."""
        out = messages_from_conversation(
            [{'role': 'assistant', 'content': '', 'status': 'cancelled'}])
        assert out[0].status == 'cancelled'

    def test_이력_없음은_빈_리스트(self):
        assert messages_from_conversation(None) == []
        assert messages_from_conversation([]) == []


class TestConsumerPaths:
    """#77의 실제 크래시 경로. 어댑터가 아니라 이 함수들 안쪽에서 터졌다."""

    def test_build_prior_turns를_통과한다(self):
        turns = build_prior_turns(messages_from_conversation(_GOLD_TURN), 2000)
        assert turns == [{'q': _GOLD_TURN[0]['content'], 'a': _GOLD_TURN[1]['content']}]

    @pytest.mark.asyncio
    async def test_condense_query를_통과한다(self):
        out = await condense_query(FakeLlm(), '그럼 교환도 그래요?',
                                   messages_from_conversation(_GOLD_TURN))
        assert isinstance(out, str) and out

    @pytest.mark.asyncio
    async def test_condense_to_queries를_통과한다(self):
        """멀티쿼리 경로. 지금은 _condense_call이 multi 여부와 무관하게 이력 조립을 공유하므로
        위 단일 경로와 같은 코드를 타지만, 갈라지면 이 테스트만 남는다 — 소비처가 둘인 건 사실이다."""
        out = await condense_to_queries(FakeLlm(), '그럼 교환도 그래요?',
                                        messages_from_conversation(_GOLD_TURN))
        assert isinstance(out, list) and out


class TestStatusRendering:
    """#72가 추가한 분기가 조용히 바뀌는 것을 잡는다. 문구는 상수로 비교한다."""

    def _rendered(self, status: str, content: str) -> str:
        msgs = messages_from_conversation(
            [{'role': 'user', 'content': '질문'},
             {'role': 'assistant', 'content': content, 'status': status}])
        return build_prior_turns(msgs, 2000)[0]['a']

    def test_취소_턴은_부분답변에_표식을_붙인다(self):
        assert self._rendered('cancelled', '여기까지') == '여기까지' + CANCELLED_TURN_SUFFIX

    def test_취소_턴이_빈_답이면_대체_문구(self):
        assert self._rendered('cancelled', '') == CANCELLED_TURN_EMPTY

    def test_실패_턴은_대체_문구(self):
        assert self._rendered('failed', '') == FAILED_TURN_EMPTY

    def test_정상_턴은_본문_그대로(self):
        assert self._rendered(DEFAULT_STATUS, '정상 답변') == '정상 답변'


class TestSmokeSample:
    """스모크셋이 타입을 누락하지 않는지 (#77).

    옛 구현(rows[:n])은 gold의 테넌트 블록 순서 때문에 SMOKE<61에서 multi_turn을 한 건도
    포함하지 않았다 — #77이 고친 경로가 바로 그것이라, 빨리 확인하려고 SMOKE를 쓰면
    이 버그를 못 잡았다. 그 함정을 되돌리지 못하게 고정한다.
    """

    def test_작은_스모크에도_모든_타입이_들어온다(self):
        from eval.generation import GEN_TYPES, _smoke_sample

        rows = ([{'type': 'single_fact'}] * 20 + [{'type': 'paraphrase'}] * 20
                + [{'type': 'rare_lexical'}] * 10 + [{'type': 'multi_doc'}] * 10
                + [{'type': 'multi_turn'}] * 15)          # gold의 테넌트 블록 순서와 같은 모양
        for n in (1, 5, 10):
            got = {r['type'] for r in _smoke_sample(rows, n)}
            assert got == GEN_TYPES, f'SMOKE={n}에서 누락: {GEN_TYPES - got}'

    def test_원래_순서를_유지한다(self):
        """결과 파일을 회차 간 비교하기 쉬우려면 순서가 흔들리지 않아야 한다."""
        from eval.generation import _smoke_sample

        rows = [{'type': 'a', 'i': 0}, {'type': 'b', 'i': 1}, {'type': 'a', 'i': 2}]
        assert [r['i'] for r in _smoke_sample(rows, 4)] == [0, 1, 2]
