import io
from pathlib import Path

import pytest

from app.ingestion.exceptions import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.ingestion.pipeline import IngestResult
from app.ingestion.upload import (
    remove_stored_files,
    safe_display_name,
    save_stream,
    store_upload,
)
from app.vectorstore.chroma_store import VectorStore
from tests.fakes import FakeEmbedder

TEXT = " ".join(f"This is sentence number {i} about topic {i % 7}." for i in range(60)).encode()


@pytest.fixture
def upload_dir(tmp_path: Path) -> Path:
    return tmp_path / "uploads"


def upload(store: VectorStore, upload_dir: Path, data: bytes, name: str) -> IngestResult:
    return store_upload(
        io.BytesIO(data), name, upload_dir, store, FakeEmbedder(),
        chunk_size=200, chunk_overlap=40,
    )


def files_in(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.glob("*"))  # includes hidden temp files


def test_upload_is_indexed_and_original_kept(store: VectorStore, upload_dir: Path) -> None:
    result = upload(store, upload_dir, TEXT, "note.txt")
    assert result.status == "indexed"
    assert store.has_document(result.doc_id)
    assert files_in(upload_dir) == [f"{result.doc_id}.txt"]


def test_duplicate_leaves_single_copy(store: VectorStore, upload_dir: Path) -> None:
    first = upload(store, upload_dir, TEXT, "a.txt")
    second = upload(store, upload_dir, TEXT, "b.txt")
    assert second.status == "duplicate"
    assert files_in(upload_dir) == [f"{first.doc_id}.txt"]


def test_unsupported_extension_rejected_before_writing(
    store: VectorStore, upload_dir: Path
) -> None:
    with pytest.raises(UnsupportedFileTypeError):
        upload(store, upload_dir, b"x", "image.png")
    assert files_in(upload_dir) == []


def test_empty_document_leaves_no_temp_file(store: VectorStore, upload_dir: Path) -> None:
    with pytest.raises(EmptyDocumentError):
        upload(store, upload_dir, b"   \n  ", "empty.txt")
    assert files_in(upload_dir) == []


def test_directory_parts_in_name_are_stripped(store: VectorStore, upload_dir: Path) -> None:
    result = upload(store, upload_dir, TEXT, "../../evil.txt")
    assert result.filename == "evil.txt"
    assert files_in(upload_dir) == [f"{result.doc_id}.txt"]
    assert safe_display_name("..\\..\\evil.txt") == "evil.txt"


def test_save_stream_rejects_oversize(tmp_path: Path) -> None:
    with pytest.raises(FileTooLargeError):
        save_stream(io.BytesIO(b"x" * 10), tmp_path / "out.bin", max_bytes=5)


def test_remove_stored_files_ignores_glob_characters(upload_dir: Path) -> None:
    upload_dir.mkdir()
    (upload_dir / "keep.txt").write_text("a")
    (upload_dir / "abc.txt").write_text("b")
    assert remove_stored_files(upload_dir, "*") == 0  # must not match everything
    assert files_in(upload_dir) == ["abc.txt", "keep.txt"]
    assert remove_stored_files(upload_dir, "abc") == 1
    assert files_in(upload_dir) == ["keep.txt"]