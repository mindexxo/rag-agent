"""운영 리포트 응답 스키마 (지표 MVP — Alli형: 사용량·답변률·지식 갭)."""
from datetime import datetime

from pydantic import BaseModel


class DailyCount(BaseModel):
    date: str            # 'YYYY-MM-DD'
    questions: int


class TopDocument(BaseModel):
    filename: str
    citations: int       # 답변 인용 누적 (기간 내)


class StatsSummary(BaseModel):
    """운영자가 보고 행동할 수 있는 것만 (2026-08-07 정리 — 제외 사유는 routers/stats.py docstring).

    활성 사용자·응답 지연·차단/실패는 응답에서 제거 — 원천 컬럼은 계속 저장되므로 필요 시 SQL로.
    """
    period_days: int
    questions: int               # 기간 내 질문 수 (user 메시지 — OTHER 포함 전체 사용량)
    knowledge_done: int          # 완료된 지식 질문 답변 수 (intent='KNOWLEDGE') — 답변률의 분모
    refusals: int                # "확인할 수 없습니다" 거절 답변 수
    refusal_rate: float          # 거절률 = refusals / knowledge_done — 잡담(OTHER)이 분모에 안 섞인 지식 커버리지
    daily: list[DailyCount]      # 질문 없는 날도 0으로 채워짐 (기간 내 연속)
    top_documents: list[TopDocument]


class UnansweredItem(BaseModel):
    question: str        # 거절당한 사용자 질문 원문 — FAQ 보강 재료 (지식 갭)
    asked_at: datetime
