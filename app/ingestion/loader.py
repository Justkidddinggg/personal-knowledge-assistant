"""Entry point of the ingestion stage: validate -> hash -> parse -> clean."""
import logging
from pathlib import Path

from app.ingestion.cleaning import clean_text
from app.ingestion.exceptions import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from app.ingestion.hashing import hash_file
from app.ingestion.parsers import PARSERS
from app.schemas.document import ParsedDocument, ParsedPage

logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 50


def load_document(path: Path, original_name: str | None = None) -> ParsedDocument:
    """Parse one file into a ParsedDocument with cleaned, page-tagged text."""
    name = original_name or path.name
    suffix = Path(name).suffix.lower()

    if suffix not in PARSERS:
        supported = ", ".join(sorted(PARSERS))
        raise UnsupportedFileTypeError(f"'{suffix}' is not supported. Use: {supported}")

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise FileTooLargeError(f"{size_mb:.1f} MB exceeds {MAX_FILE_SIZE_MB} MB limit")

    file_type, parser = PARSERS[suffix]
    raw_pages = parser(path)

    cleaned = [
        ParsedPage(text=clean_text(p.text), page=p.page) for p in raw_pages
    ]
    cleaned = [p for p in cleaned if p.text]  # drop blank pages

    if not cleaned:
        raise EmptyDocumentError(
            f"No text found in '{name}'. It may be a scanned document (image-only); "
            "OCR is not supported yet."
        )

    doc = ParsedDocument(
        doc_id=hash_file(path),
        filename=name,
        file_type=file_type,
        pages=cleaned,
    )
    logger.info("Loaded %s: %d page(s), %d chars", name, len(doc.pages), doc.total_chars)
    return doc