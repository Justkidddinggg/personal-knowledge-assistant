"""Persistent vector storage backed by ChromaDB."""
import logging
from functools import lru_cache
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.schemas.document import Chunk, DocumentInfo, SearchResult

logger = logging.getLogger(__name__)

COLLECTION_NAME = "knowledge_base"
_ADD_BATCH_SIZE = 500


class EmbeddingModelMismatchError(RuntimeError):
    """The stored vectors were made by a different embedding model."""


def _to_metadata(chunk: Chunk) -> dict[str, str | int]:
    meta: dict[str, str | int] = {
        "doc_id": chunk.doc_id,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
    }
    if chunk.page is not None:  # Chroma rejects None values, so omit the key
        meta["page"] = chunk.page
    return meta


class VectorStore:
    def __init__(self, persist_dir: Path, embedding_model: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self._collection = self._client.get_collection(COLLECTION_NAME)
        except Exception:  # "not found" exception type differs across versions
            self._collection = self._client.create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine", "embedding_model": embedding_model},
            )

        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise EmbeddingModelMismatchError(
                f"Index was built with '{stored_model}' but the app is configured "
                f"for '{embedding_model}'. Re-index your documents or restore the "
                "original EMBEDDING_MODEL."
            )

    # ---- writes -------------------------------------------------------
    def add_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        for start in range(0, len(chunks), _ADD_BATCH_SIZE):
            end = start + _ADD_BATCH_SIZE
            batch = chunks[start:end]
            # upsert (not add): re-running a half-finished ingest is harmless
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings[start:end].tolist(),
                documents=[c.text for c in batch],
                metadatas=[_to_metadata(c) for c in batch],
            )
        logger.info("Stored %d chunk(s)", len(chunks))

    def delete_document(self, doc_id: str) -> int:
        """Delete every chunk of a document. Returns how many were removed."""
        ids = self._collection.get(where={"doc_id": doc_id}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("Deleted %d chunk(s) of document %s", len(ids), doc_id[:12])
        return len(ids)

    # ---- reads --------------------------------------------------------
    def has_document(self, doc_id: str) -> bool:
        found = self._collection.get(where={"doc_id": doc_id}, limit=1, include=[])
        return bool(found["ids"])

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> list[DocumentInfo]:
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        grouped: dict[str, DocumentInfo] = {}
        for meta in metadatas:
            doc_id = str(meta["doc_id"])
            current = grouped.get(doc_id)
            grouped[doc_id] = DocumentInfo(
                doc_id=doc_id,
                filename=str(meta["filename"]),
                num_chunks=(current.num_chunks if current else 0) + 1,
            )
        return sorted(grouped.values(), key=lambda d: d.filename.lower())

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        doc_ids: list[str] | None = None,
    ) -> list[SearchResult]:
        """Top-K similarity search, optionally restricted to some documents."""
        if self.count() == 0 or doc_ids == []:
            return []
        where = {"doc_id": {"$in": doc_ids}} if doc_ids is not None else None
        raw = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results: list[SearchResult] = []
        for chunk_id, text, meta, distance in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    doc_id=str(meta["doc_id"]),
                    filename=str(meta["filename"]),
                    page=int(meta["page"]) if "page" in meta else None,
                    text=text,
                    score=1.0 - float(distance),
                )
            )
        return results


@lru_cache
def get_vector_store() -> VectorStore:
    s = get_settings()
    return VectorStore(s.chroma_dir, s.embedding_model)