from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    filters: dict | None = None


class CitationSchema(BaseModel):
    path: str
    start_line: int
    end_line: int
    chunk_id: str = ""


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationSchema]
    question: str
    retrieved_chunk_count: int = 0


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
