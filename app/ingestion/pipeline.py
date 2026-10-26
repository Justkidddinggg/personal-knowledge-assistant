"""The complete ingestion pipeline: parse -> dedupe -> chunk -> embed -> store."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from app.ingestion.chunking import chunk_document
from app.ingestion.loader import load_document
from app.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class SupportsEmbedDocuments(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    filename: str
    status: Literal["indexed", "duplicate"]
    num_chunks: int


def ingest_file(
    path: Path,
    store: VectorStore,
    embedder: SupportsEmbedDocuments,
    chunk_size: int,
    chunk_overlap: int,
    original_name: str | None = None,
) -> IngestResult:
    doc = load_document(path, original_name)

    # Duplicate check happens BEFORE the expensive embedding step.
    if store.has_document(doc.doc_id):
        logger.info("Skipping %s: identical content already indexed", doc.filename)
        return IngestResult(doc.doc_id, doc.filename, "duplicate", 0)

    chunks = chunk_document(doc, chunk_size, chunk_overlap)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store.add_chunks(chunks, embeddings)
    return IngestResult(doc.doc_id, doc.filename, "indexed", len(chunks))