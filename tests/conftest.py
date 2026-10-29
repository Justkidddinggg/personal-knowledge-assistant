from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.embeddings.embedder import get_embedder
from app.main import app
from app.vectorstore.chroma_store import VectorStore, get_vector_store
from tests.fakes import FakeEmbedder


@pytest.fixture
def store(tmp_path: Path) -> VectorStore:
    return VectorStore(tmp_path / "chroma", embedding_model="fake")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
        chunk_size=200,
        chunk_overlap=40,
    )


@pytest.fixture
def client(store: VectorStore, settings: Settings) -> Iterator[TestClient]:
    """API client wired to a temporary database, a fake embedder and temp folders."""
    app.dependency_overrides[get_vector_store] = lambda: store
    app.dependency_overrides[get_embedder] = lambda: FakeEmbedder()
    app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(app)
    app.dependency_overrides.clear()