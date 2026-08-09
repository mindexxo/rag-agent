from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from schemas.kms import SourceCitation

# 👎 사유 태그 (#8) — 수리 경로별 고정 슬러그. 검증은 이 Literal 하나 (4종에 테이블은 과설계)
FeedbackTag = Literal['wrong_info', 'wrong_source', 'outdated_doc', 'insufficient']


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
    message_id: int               # 피드백 PATCH 대상 식별 (#8) — assistant 메시지 id를 FE가 알아야 버튼이 동작
    role: str
    content: str
    status: str = "done"          # generating|done|failed|blocked (FE 재접속 시 진행상태 판별)
    sources: list[SourceCitation] | None = None
    attachments: list[str] | None = None  # 이 턴에 첨부된 파일명 목록 (본문 텍스트는 내려주지 않음)
    cited_docs: list[str] | None = None   # 실인용 파일명 (저장 시 확정) — FE 각주 필터가 본문 재파싱 없이 사용
    feedback: bool | None = None          # assistant: 👍/👎 현재 상태 — FE가 히스토리 재진입 시 버튼 상태 복원 (#8)
    feedback_tag: FeedbackTag | None = None


class MessageFeedbackUpdate(BaseModel):
    """PATCH /kms/messages/{id}/feedback 바디 (#8) — 멱등 set: FE가 원하는 최종 상태를 보낸다.

    feedback: true=👍, false=👎, null=취소(태그도 함께 NULL).
    tag: 👎일 때만 허용, 옵셔널 (안 골라도 👎는 기록 — 필수로 만들면 👎 자체가 줄어드는 UX 함정).
    """
    feedback: bool | None
    tag: FeedbackTag | None = None

    @model_validator(mode='after')
    def _tag_only_with_downvote(self):
        if self.tag is not None and self.feedback is not False:
            raise ValueError('tag는 feedback=false(👎)일 때만 보낼 수 있습니다')
        return self


class MessageFeedbackState(BaseModel):
    """피드백 PATCH 응답 — 저장된 최종 상태 에코 (FE 상태 동기화용)."""
    message_id: int
    feedback: bool | None
    feedback_tag: FeedbackTag | None
