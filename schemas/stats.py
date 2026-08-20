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
    # 근거없음(ungrounded) = 인용한 문서가 0건인 답변 (#61). 옛 이름은 refusals/refusal_rate로,
    # 판정이 "거절 문구 부분일치"였다 — 프롬프트 문구에 묶여 규칙 완화 시 검출률이 붕괴했다.
    # 이제 판정은 messages.cited_docs가 비었는지이고, 문구와 무관하다. 거절뿐 아니라
    # "근거 없이 답한" 경우도 포함하므로 옛 '거절률'보다 넓은 개념이다.
    ungrounded: int              # 근거 미확인 답변 수
    ungrounded_rate: float       # 근거미확인율 = ungrounded / knowledge_done — 잡담(OTHER)이 분모에 안 섞인 지식 커버리지
    daily: list[DailyCount]      # 질문 없는 날도 0으로 채워짐 (기간 내 연속)
    top_documents: list[TopDocument]


class UnansweredItem(BaseModel):
    question: str        # 근거를 못 댄 답변의 사용자 질문 원문 — FAQ 보강 재료 (지식 갭, #61)
    asked_at: datetime
