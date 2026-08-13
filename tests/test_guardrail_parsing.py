"""가드레일 응답 파싱 단위 테스트 — _as_bool / _extract_json.

_as_bool은 리팩토링에서 고친 버그(bool("false")==True 함정)를 고정하는 테스트.
"""
import pytest

from rag.guardrail import _as_bool, _extract_json


class TestAsBool:
    def test_문자열_false류는_False(self):
        for v in ('false', 'FALSE', 'False', ' no ', '0', 'f', 'N'):
            assert _as_bool(v) is False, v

    def test_문자열_true류는_True(self):
        for v in ('true', 'yes', '1', '아무거나'):
            assert _as_bool(v) is True, v

    def test_bool은_그대로(self):
        assert _as_bool(True) is True
        assert _as_bool(False) is False

    def test_None은_default(self):
        assert _as_bool(None) is True            # 기본 fail-open
        assert _as_bool(None, default=False) is False

    def test_JSON_숫자_0과_1(self):
        # LLM이 "safe": 0 처럼 숫자로 주는 건 실제로 흔함
        assert _as_bool(0) is False
        assert _as_bool(1) is True


class TestExtractJson:
    def test_정상_JSON(self):
        assert _extract_json('{"safe": true}') == {'safe': True}

    def test_앞뒤_잡텍스트_무시(self):
        raw = '판단 결과입니다: {"safe": false, "reason": "x"} 이상입니다.'
        assert _extract_json(raw) == {'safe': False, 'reason': 'x'}

    def test_닫는_중괄호_직후_텍스트(self):
        # '}' 바로 뒤가 공백이 아니면 슬라이스 경계(+1)가 정확해야 파싱됨 — off-by-one 뮤테이션 방지
        assert _extract_json('{"safe": true}바로뒤텍스트') == {'safe': True}

    def test_마크다운_펜스(self):
        raw = '```json\n{"safe": true, "intent": "OTHER"}\n```'
        assert _extract_json(raw) == {'safe': True, 'intent': 'OTHER'}

    def test_중첩_객체(self):
        # rindex(마지막 })가 스펙 — index(첫 })로 바뀌면 중첩 JSON이 잘려 조용히 fail-open 퇴화
        raw = '{"safe": true, "meta": {"k": 1}}'
        assert _extract_json(raw) == {'safe': True, 'meta': {'k': 1}}

    def test_JSON_없으면_ValueError(self):
        # classify_and_guard의 fail-open except가 이 계약(ValueError)에 의존한다
        with pytest.raises(ValueError):
            _extract_json('중괄호가 전혀 없는 응답')
