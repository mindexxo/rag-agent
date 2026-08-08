from datetime import datetime

from pydantic import BaseModel, Field

from schemas.kms import SourceCitation


class ConversationSummary(BaseModel):
    conversation_id: int
    title: str | None = None     # 첫 질문 앞 80자 (없으면 FE가 '대화 #id' 폴백)
    updated_at: datetime         # 최근 사용 시각(last_used_at) — 목록 "n분 전" 표시용 (#10)


class ConversationListResponse(BaseModel):
    """목록 응답 (#10, breaking — 기존 배열 → 객체. FE 합의됨)."""
    items: list[ConversationSummary]
    has_more: bool               # offset+limit 뒤에 더 있는가 (limit+1 조회로 판정)


class ConversationTitleUpdate(BaseModel):
    """PATCH 바디 — 제목 변경만 허용."""
    title: str = Field(min_length=1, max_length=80)   # 80자 = 자동 제목(첫 질문 앞 80자)과 동일 상한


class ConversationMessage(BaseModel):
    role: str
    content: str
    status: str = "done"          # generating|done|failed|blocked (FE 재접속 시 진행상태 판별)
    sources: list[SourceCitation] | None = None
    attachments: list[str] | None = None  # 이 턴에 첨부된 파일명 목록 (본문 텍스트는 내려주지 않음)
    cited_docs: list[str] | None = None   # 실인용 파일명 (저장 시 확정) — FE 각주 필터가 본문 재파싱 없이 사용
