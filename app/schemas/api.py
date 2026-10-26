"""Request/response shapes of the HTTP API (validated by Pydantic)."""
from pydantic import BaseModel


class DocumentOut(BaseModel):
    doc_id: str
    filename: str
    num_chunks: int


class DeleteOut(BaseModel):
    doc_id: str
    deleted_chunks: int