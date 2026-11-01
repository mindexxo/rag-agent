"""TurnStatus 계약 — 정의(rag/models.TurnStatus)와 파생 집합(rag/turn/turn_state)의 관계가 조용히 어긋나지 않게 한다."""

import json

from fastapi.encoders import jsonable_encoder

from rag.models import TurnStatus
from rag.turn.turn_state import HISTORY_ISOLATED, TERMINAL, UNANSWERED


class TestStrEnumCompat:
    """StrEnum이 기존 문자열과 호환된다 — tests·eval의 문자열 시딩이 이 성질에 기댄다."""

    def test_문자열_비교와_해시가_동일하다(self):
        assert TurnStatus.CANCELLED == 'cancelled'
        assert hash(TurnStatus.CANCELLED) == hash('cancelled')
        assert 'failed' in UNANSWERED  # 문자열 멤버십도 그대로

    def test_와이어_직렬화는_값_그대로다(self):
        """SSE done.finish_reason·FE 응답 status — 소문자 문자열 그대로 나간다."""
        assert json.dumps(TurnStatus.DONE) == '"done"'
        assert jsonable_encoder(TurnStatus.BLOCKED) == 'blocked'

    def test_다섯_값이_전부다(self):
        """새 상태를 추가하면 이 테스트가 깨진다 — 그때 확인할 곳:
        파생 집합(turn_state) · eval/gold_history.py 계약 · FE의 status 판별 로직."""
        assert {s.value for s in TurnStatus} == {'generating', 'done', 'cancelled', 'failed', 'blocked'}


class TestDerivedSets:
    """두 파생 집합은 소비처가 다르다(이력 조립 vs RETRY 판정) — 뭉치면 그 독립성이 사라진다."""

    def test_격리와_미응답은_교집합이_없다(self):
        assert set(HISTORY_ISOLATED) & set(UNANSWERED) == set()

    def test_TERMINAL은_전체에서_generating만_뺀_것이다(self):
        """finish_reason 어휘의 근거 — 계산형이라 상태 추가 시 자동으로 따라온다.

        구체 값으로도 고정한다 — 와이어(FE) 계약이 이 네 값을 전제한다.
        (5값 완전성 테스트가 있는 한 두 단언은 동치지만, 실패 메시지가 다른 이야기를 한다.)
        """
        assert set(TERMINAL) == set(TurnStatus) - {TurnStatus.GENERATING}
        assert set(TERMINAL) == {TurnStatus.DONE, TurnStatus.CANCELLED, TurnStatus.FAILED, TurnStatus.BLOCKED}

    def test_blocked는_미응답이_아니다(self):
        """차단 결정을 '다시'로 뒤집으면 안 된다 — RETRY 대상에서 제외."""
        assert TurnStatus.BLOCKED not in UNANSWERED
        assert TurnStatus.BLOCKED in HISTORY_ISOLATED


class TestWriteValidation:
    def test_다섯_종_밖_값은_즉시_거부된다(self):
        """finalize_turn 진입부의 TurnStatus(status) — 오타('canceled')가 조용히 저장돼
        RETRY가 무시되는 버그 클래스를 저장 전에 끊는다."""
        import pytest

        with pytest.raises(ValueError):
            TurnStatus('canceled')  # l 하나 빠진 오타
        assert TurnStatus('cancelled') is TurnStatus.CANCELLED  # 정상 값은 멤버로
