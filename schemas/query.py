"""죽은 레거시 — 참조처 없음, 새 코드에서 쓰지 마라(실제 질의 모델은 schemas/kms.py의 KmsQueryRequest)."""

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
