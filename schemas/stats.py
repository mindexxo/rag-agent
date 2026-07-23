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
    period_days: int
    questions: int               # 기간 내 질문 수 (user 메시지)
    active_users: int            # 질문한 고유 사용자 수 (X-User-Id 기준)
    refusals: int                # "확인할 수 없습니다" 거절 답변 수
    refusal_rate: float          # 거절률 = refusals / 완료된 답변 (지식 커버리지 신호)
    blocked: int                 # 가드레일 차단
    failed: int                  # 생성 실패
    avg_latency_ms: int | None   # 평균 응답 생성 시간 (씨앗 — 데이터 쌓이면 채워짐)
    p95_latency_ms: int | None
    daily: list[DailyCount]
    top_documents: list[TopDocument]


class UnansweredItem(BaseModel):
    question: str        # 거절당한 사용자 질문 원문 — FAQ 보강 재료 (지식 갭)
    asked_at: datetime
