"""Endpoints for uploading, listing and deleting documents."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Response, UploadFile

from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder, get_embedder
from app.ingestion.upload import remove_stored_files, store_upload
from app.schemas.api import DeleteOut, DocumentOut, UploadOut
from app.vectorstore.chroma_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=UploadOut, status_code=201)
def upload_document(
    response: Response,
    file: UploadFile,
    store: VectorStore = Depends(get_vector_store),
    embedder: Embedder = Depends(get_embedder),
    settings: Settings = Depends(get_settings),
) -> UploadOut:
    """Upload and index one document (PDF, DOCX, TXT or MD)."""
    result = store_upload(
        file.file,
        file.filename,
        settings.upload_dir,
        store,
        embedder,
        settings.chunk_size,
        settings.chunk_overlap,
    )
    if result.status == "duplicate":
        response.status_code = 200  # nothing new was created
    return UploadOut(
        doc_id=result.doc_id,
        filename=result.filename,
        status=result.status,
        num_chunks=result.num_chunks,
    )


@router.get("", response_model=list[DocumentOut])
def list_documents(store: VectorStore = Depends(get_vector_store)) -> list[DocumentOut]:
    """List every indexed document with its chunk count."""
    return [
        DocumentOut(doc_id=d.doc_id, filename=d.filename, num_chunks=d.num_chunks)
        for d in store.list_documents()
    ]


@router.delete("/{doc_id}", response_model=DeleteOut)
def delete_document(
    doc_id: str,
    store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> DeleteOut:
    """Remove a document: its indexed chunks and its stored original file."""
    deleted = store.delete_document(doc_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    remove_stored_files(settings.upload_dir, doc_id)
    return DeleteOut(doc_id=doc_id, deleted_chunks=deleted)