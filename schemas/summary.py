from pydantic import BaseModel

class SummaryRequest(BaseModel):
    text: str

class SummaryResponse(BaseModel):
    summary: list[str]

class Segment(BaseModel):
    start: float
    end: float
    text: str
    speaker: str

class SttResponse(BaseModel):
    segments: list[Segment]
    language: str
    duration: float
    num_speakers: int
    full_text: str
    summary: SummaryResponse
