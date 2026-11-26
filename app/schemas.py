from pydantic import BaseModel, Field
from typing import Optional

class IngestRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2_000_000)
    source: str = 'pasted-text.txt'
    title: str = 'Untitled document'
    section: Optional[str] = None

class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=4000)
    top_k: int = Field(default=12, ge=1, le=30)
    source: Optional[str] = Field(default=None, max_length=512)

class Citation(BaseModel):
    id: str
    source: str
    title: str
    section: Optional[str] = None
    position: int
    page: Optional[int] = None
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    timings_ms: dict[str, float]
    usage: dict[str, int | float | str]
    pipeline: list[str]
