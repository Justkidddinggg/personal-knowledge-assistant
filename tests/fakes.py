import zlib

import numpy as np


class FakeEmbedder:
    """Deterministic bag-of-words embedder for tests (no model download)."""

    DIM = 64

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.DIM, dtype=np.float32)
        for word in text.lower().split():
            v[zlib.crc32(word.encode()) % self.DIM] += 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm else v