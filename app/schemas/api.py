"""Request/response shapes of the HTTP API (validated by Pydantic)."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class DeleteOut(BaseModel):
    doc_id: str
    deleted_chunks: int


class UploadOut(BaseModel):
    doc_id: str
    filename: str
    status: Literal["indexed", "duplicate"]
    num_chunks: int


class AskIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    question: str = Field(min_length=1, max_length=1000)
    doc_ids: list[str] | None = Field(default=None, max_length=50)
    session_id: str | None = Field(default=None, max_length=64)


class SourceOut(BaseModel):
    label: str          # "S1", "S2", ... matches the [S#] citations in the answer
    doc_id: str
    chunk_id: str
    filename: str
    page: int | None
    score: float
    text: str           # the excerpt the LLM saw
    cited: bool         # True if the answer actually cites this source


class AskOut(BaseModel):
    answer: str
    session_id: str
    search_query: str   # what was actually searched (differs for follow-ups)
    grounded: bool      # False when no evidence was found and the LLM was not called
    sources: list[SourceOut]
    invalid_citations: list[int]
    retrieval_ms: float
    generation_ms: float