from pathlib import Path

import pytest

from app.vectorstore.chroma_store import VectorStore


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedding_model="fake")