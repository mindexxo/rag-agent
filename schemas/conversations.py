from pydantic import BaseModel

from schemas.kms import SourceCitation


class ConversationSummary(BaseModel):
    conversation_id: int
    title: str | None = None     # 첫 질문 앞 80자 (없으면 FE가 '대화 #id' 폴백)

class ConversationMessage(BaseModel):
    role: str
    content: str
    status: str = "done"          # generating|done|failed|blocked (FE 재접속 시 진행상태 판별)
    sources: list[SourceCitation] | None = None
    attachments: list[str] | None = None  # 이 턴에 첨부된 파일명 목록 (본문 텍스트는 내려주지 않음)
    cited_docs: list[str] | None = None   # 실인용 파일명 (저장 시 확정) — FE 각주 필터가 본문 재파싱 없이 사용
