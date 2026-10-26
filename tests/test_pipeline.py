from pathlib import Path

from app.ingestion.pipeline import ingest_file
from app.vectorstore.chroma_store import VectorStore
from tests.fakes import FakeEmbedder

TEXT = " ".join(f"This is sentence number {i} about topic {i % 7}." for i in range(60))


def _write(path: Path, text: str = TEXT) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_indexes_document(tmp_path: Path, store: VectorStore) -> None:
    result = ingest_file(
        _write(tmp_path / "a.txt"), store, FakeEmbedder(), chunk_size=200, chunk_overlap=40
    )
    assert result.status == "indexed"
    assert result.num_chunks > 1
    assert store.count() == result.num_chunks


def test_same_file_twice_is_duplicate(tmp_path: Path, store: VectorStore) -> None:
    f = _write(tmp_path / "a.txt")
    first = ingest_file(f, store, FakeEmbedder(), chunk_size=200, chunk_overlap=40)
    second = ingest_file(f, store, FakeEmbedder(), chunk_size=200, chunk_overlap=40)
    assert second.status == "duplicate"
    assert store.count() == first.num_chunks


def test_renamed_copy_is_duplicate(tmp_path: Path, store: VectorStore) -> None:
    a, b = _write(tmp_path / "a.txt"), _write(tmp_path / "b.txt")
    ingest_file(a, store, FakeEmbedder(), chunk_size=200, chunk_overlap=40)
    result = ingest_file(b, store, FakeEmbedder(), chunk_size=200, chunk_overlap=40)
    assert result.status == "duplicate"