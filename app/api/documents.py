"""Endpoints for managing the indexed documents."""
import logging

from fastapi import APIRouter, Depends, HTTPException

from app.schemas.api import DeleteOut, DocumentOut
from app.vectorstore.chroma_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=list[DocumentOut])
def list_documents(store: VectorStore = Depends(get_vector_store)) -> list[DocumentOut]:
    """List every indexed document with its chunk count."""
    return [
        DocumentOut(doc_id=d.doc_id, filename=d.filename, num_chunks=d.num_chunks)
        for d in store.list_documents()
    ]


@router.delete("/{doc_id}", response_model=DeleteOut)
def delete_document(
    doc_id: str, store: VectorStore = Depends(get_vector_store)
) -> DeleteOut:
    """Remove a document and all of its chunks from the index."""
    deleted = store.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return DeleteOut(doc_id=doc_id, deleted_chunks=deleted)