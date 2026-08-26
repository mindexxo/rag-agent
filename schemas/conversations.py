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
    # 검색(q) 매칭 발췌 (#28) — 내용에서 걸린 대화만. 제목에서만 걸렸거나 q 미전송이면 None
    # (제목은 이미 title로 보이므로 같은 내용을 중복해 내려주지 않는다).
    # 하이라이트는 FE가 이 문자열 안에서 처리한다 — 서버는 마크업 없는 평문만 준다.
    snippet: str | None = None


class ConversationListResponse(BaseModel):
    """목록 응답 (#10, breaking — 기존 배열 → 객체. FE 합의됨)."""
    items: list[ConversationSummary]
    has_more: bool               # offset+len(items) < total (#28부터 total에서 파생)
    total: int = 0               # 필터 적용 후 전체 대화 수 (#28) — q 없으면 내 대화 총 개수


class ConversationTitleUpdate(BaseModel):
    """PATCH 바디 — 제목 변경만 허용."""
    title: str = Field(min_length=1, max_length=80)   # 80자 = 자동 제목(첫 질문 앞 80자)과 동일 상한


class ConversationMessage(BaseModel):
    message_id: int               # 피드백 PATCH 대상 식별 (#8) — assistant 메시지 id를 FE가 알아야 버튼이 동작
    role: str
    content: str
    # 값 어휘는 rag/models.TurnStatus. 타입은 str 유지 — TurnStatus로 바꾸면 응답 검증이 되어
    # DB의 낡은 비정상 값에서 GET 전체가 500이 된다(전엔 무해하게 echo). 쓰기는 finalize_turn이
    # 이미 막으므로 읽기 검증은 이득 없이 실패 모드만 추가한다 (#85 리뷰).
    status: str = "done"          # FE 재접속 시 진행상태 판별
    # 실제 인용된 출처 객체만 (#56) — SSE done.citations와 같은 이름·의미 (매핑 두 벌 금지).
    # 구 계약의 sources(검색 후보 전체)+cited_docs(파일명)는 FE가 필터를 들고 있어야 했다 — 서버가 확정해 보낸다.
    citations: list[SourceCitation] | None = None
    attachments: list[str] | None = None  # 이 턴에 첨부된 파일명 목록 (본문 텍스트는 내려주지 않음)
    feedback: bool | None = None          # assistant: 👍/👎 현재 상태 — FE가 히스토리 재진입 시 버튼 상태 복원 (#8)
    feedback_tag: FeedbackTag | None = None
    feedback_text: str | None = None


class MessageFeedbackUpdate(BaseModel):
    """PATCH /kms/messages/{id}/feedback 바디 (#8) — 멱등 set: FE가 원하는 최종 상태를 보낸다.

    feedback: true=👍, false=👎, null=취소(태그·텍스트도 함께 NULL).
    tag/text: 👎일 때만 허용, 둘 다 옵셔널 (안 적어도 👎는 기록 — 필수로 만들면 👎 자체가
    줄어드는 UX 함정). tag=집계·라우팅 축, text=태그가 못 잡는 사유 발굴 축.
    """
    feedback: bool | None
    tag: FeedbackTag | None = None
    text: str | None = Field(None, max_length=500)   # 자유 서술 — FE는 고객 정보 미기입 안내 필수

    @model_validator(mode='after')
    def _detail_only_with_downvote(self):
        if (self.tag is not None or self.text is not None) and self.feedback is not False:
            raise ValueError('tag·text는 feedback=false(👎)일 때만 보낼 수 있습니다')
        return self


class MessageFeedbackState(BaseModel):
    """피드백 PATCH 응답 — 저장된 최종 상태 에코 (FE 상태 동기화용)."""
    message_id: int
    feedback: bool | None
    feedback_tag: FeedbackTag | None
    feedback_text: str | None
