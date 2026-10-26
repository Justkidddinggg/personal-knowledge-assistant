"""Thin wrapper around a Sentence Transformers model."""
import logging
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self.max_seq_length: int = self._model.max_seq_length

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed many texts. Returns an array of shape (len(texts), dim)."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,  # cosine similarity == dot product
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single question. Kept separate from embed_documents because
        some models (e.g. BGE) need a special prefix for queries only."""
        return self.embed_documents([text])[0]


@lru_cache
def get_embedder() -> Embedder:
    """Load the model once per process; loading takes seconds, querying ms."""
    return Embedder(get_settings().embedding_model)