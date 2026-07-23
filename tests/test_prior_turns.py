"""멀티턴 이력 조립 단위 테스트 — conversation.build_prior_turns.

user→assistant 쌍만 턴으로 인정, 토큰 예산은 최신 턴 우선(최소 1턴 보장).
"""
from types import SimpleNamespace

from rag.conversation import build_prior_turns
from rag.tokens import estimate_tokens


def _msg(role: str, content: str):
    return SimpleNamespace(role=role, content=content)


def test_정상_쌍_페어링():
    msgs = [_msg('user', 'q1'), _msg('assistant', 'a1'), _msg('user', 'q2'), _msg('assistant', 'a2')]
    assert build_prior_turns(msgs, budget_tokens=1000) == [{'q': 'q1', 'a': 'a1'}, {'q': 'q2', 'a': 'a2'}]


def test_답변_없는_질문은_제외():
    msgs = [_msg('user', 'q1'), _msg('assistant', 'a1'), _msg('user', 'q2')]  # q2는 미답변
    assert build_prior_turns(msgs, budget_tokens=1000) == [{'q': 'q1', 'a': 'a1'}]


def test_순서_꼬임_assistant가_먼저면_버려짐():
    # 리뷰에서 지적된 created_at 동률 정렬 꼬임 시나리오 — 페어링이 조용히 턴을 누락하는 현재 동작 고정
    msgs = [_msg('assistant', 'a1'), _msg('user', 'q1')]
    assert build_prior_turns(msgs, budget_tokens=1000) == []


def test_연속_user는_마지막_질문만_페어링():
    msgs = [_msg('user', 'q1'), _msg('user', 'q2'), _msg('assistant', 'a')]
    assert build_prior_turns(msgs, budget_tokens=1000) == [{'q': 'q2', 'a': 'a'}]


def test_예산은_최신_턴_우선_시간순_유지():
    # 턴마다 내용을 다르게(길이는 동일 → 토큰 비용 동일) — '어느' 턴을 골랐는지까지 단언
    qs = [f'질문{i}' + '가' * 10 for i in (1, 2, 3)]
    as_ = [f'답변{i}' + '나' * 10 for i in (1, 2, 3)]
    per_turn = estimate_tokens(qs[0]) + estimate_tokens(as_[0])
    msgs = []
    for q, a in zip(qs, as_):
        msgs += [_msg('user', q), _msg('assistant', a)]
    result = build_prior_turns(msgs, budget_tokens=per_turn * 2)
    assert result == [{'q': qs[1], 'a': as_[1]}, {'q': qs[2], 'a': as_[2]}]  # 최신 2턴, 시간순


def test_빈_답변도_턴으로_페어링됨_현재동작():
    # generating 자리표시(content='')가 이력에 섞이면 빈 턴이 프롬프트에 들어간다 —
    # 현재 동작 문서화. 자리표시를 걸러야 한다면 load_recent_messages 쪽 필터가 정답 (통합에서 확인)
    msgs = [_msg('user', 'q'), _msg('assistant', '')]
    assert build_prior_turns(msgs, budget_tokens=100) == [{'q': 'q', 'a': ''}]


def test_예산_초과여도_최소_1턴_보장():
    msgs = [_msg('user', '아주 긴 질문' * 50), _msg('assistant', '아주 긴 답변' * 50)]
    assert len(build_prior_turns(msgs, budget_tokens=1)) == 1


def test_빈_질문은_턴에서_제외됨_현재동작():
    # 빈 답변은 페어링되고(위) 빈 질문은 버려지는 비대칭 — truthiness 판정의 현재 동작 고정
    msgs = [_msg('user', ''), _msg('assistant', 'a')]
    assert build_prior_turns(msgs, budget_tokens=100) == []


def test_빈_입력():
    assert build_prior_turns([], budget_tokens=100) == []
