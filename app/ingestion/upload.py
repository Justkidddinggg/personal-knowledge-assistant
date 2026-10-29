"""Receive an uploaded file safely, index it, and keep the original copy."""
import glob
import logging
import uuid
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from app.ingestion.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.ingestion.loader import MAX_FILE_SIZE_MB
from app.ingestion.parsers import PARSERS
from app.ingestion.pipeline import IngestResult, SupportsEmbedDocuments, ingest_file
from app.vectorstore.chroma_store import VectorStore

logger = logging.getLogger(__name__)

_READ_BLOCK = 1024 * 1024  # 1 MB


def safe_display_name(raw: str | None) -> str:
    """Drop any directory part: '../../x.txt' -> 'x.txt' (handles / and \\)."""
    return PureWindowsPath(raw or "").name or "unnamed"


def save_stream(source: BinaryIO, dest: Path, max_bytes: int) -> None:
    """Copy an upload to disk in blocks, aborting as soon as it is too large."""
    written = 0
    with dest.open("wb") as out:
        while block := source.read(_READ_BLOCK):
            written += len(block)
            if written > max_bytes:
                raise FileTooLargeError("File is larger than the allowed limit")
            out.write(block)


def store_upload(
    source: BinaryIO,
    original_name: str | None,
    upload_dir: Path,
    store: VectorStore,
    embedder: SupportsEmbedDocuments,
    chunk_size: int,
    chunk_overlap: int,
) -> IngestResult:
    """Validate -> save to a temp file -> index -> keep the original as <doc_id>.<ext>."""
    name = safe_display_name(original_name)
    suffix = Path(name).suffix.lower()
    if suffix not in PARSERS:  # reject before touching the disk
        supported = ", ".join(sorted(PARSERS))
        raise UnsupportedFileTypeError(
            f"'{suffix or 'no extension'}' is not supported. Use: {supported}"
        )

    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f".tmp-{uuid.uuid4().hex}{suffix}"  # never the user's name
    try:
        save_stream(source, temp_path, MAX_FILE_SIZE_MB * 1024 * 1024)
        result = ingest_file(
            temp_path, store, embedder, chunk_size, chunk_overlap, original_name=name
        )
        if result.status == "indexed":
            temp_path.replace(upload_dir / f"{result.doc_id}{suffix}")
        return result
    finally:
        temp_path.unlink(missing_ok=True)  # no-op after a successful move


def remove_stored_files(upload_dir: Path, doc_id: str) -> int:
    """Delete the saved original(s) of a document. Returns how many were removed."""
    removed = 0
    # glob.escape: a doc_id like "*" must match literally, never "everything".
    for path in upload_dir.glob(f"{glob.escape(doc_id)}.*"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed