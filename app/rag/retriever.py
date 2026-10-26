"""Query-side pipeline: embed the question and fetch the best chunks."""
import logging
from typing import Protocol

import numpy as np

from app.schemas.document import SearchResult
from app.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)


class SupportsEmbedQuery(Protocol):
    def embed_query(self, text: str) -> np.ndarray: ...


class Retriever:
    def __init__(
        self,
        embedder: SupportsEmbedQuery,
        store: VectorStore,
        top_k: int,
        min_score: float,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(
        self, question: str, doc_ids: list[str] | None = None
    ) -> list[SearchResult]:
        """Top-K chunks, minus any that score below the relevance threshold."""
        query_vec = self._embedder.embed_query(question)
        results = self._store.query(query_vec, self._top_k, doc_ids)
        kept = [r for r in results if r.score >= self._min_score]
        logger.info(
            "Retrieved %d chunk(s), kept %d above score %.2f",
            len(results), len(kept), self._min_score,
        )
        return kept