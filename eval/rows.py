"""골드 파일 행의 모양 — 정본(eval/gold_set_v2.jsonl)과 eval/gold_private가 같은 스키마를 쓴다.

외부 JSON 경계라 TypedDict로 적는다. 런타임 검증은 하지 않는다 — 파일 검증은 eval/validate_gold_v2.py가 맡는다.
"""

from typing import NotRequired, TypedDict


class ConversationTurn(TypedDict):
    """multi_turn 문항의 앞선 대화 한 턴."""

    role: str
    content: str


class ExpectedChunk(TypedDict):
    """정답 근거 — 문서 파일명과 그 문서 청크에 포함돼야 하는 원문 발췌."""

    filename: str
    snippet: str
    heading_path: NotRequired[list[str]]  # 정본에는 없다 — 있으면 같은 발췌가 여럿일 때 이 헤딩의 청크를 고른다


class GoldRow(TypedDict):
    """골드 한 문항. expected_docs 원소가 list면 '이 중 하나'인 대체 출처 묶음이다."""

    id: str
    query: str
    type: str
    has_evidence: bool
    conversation: list[ConversationTurn] | None  # multi_turn 계열만 값이 있다
    expected_docs: list[str | list[str]]
    expected_chunks: list[ExpectedChunk]
    expected_points: list[str]
    safety_tags: list[str]
    expected_answer: NotRequired[str]
    must_not_contain: NotRequired[list[str]]
    note: NotRequired[str]
