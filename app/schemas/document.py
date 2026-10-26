"""Data structures shared across the ingestion pipeline."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ParsedPage:
    """A unit of extracted text. `page` is None for formats without pages."""

    text: str
    page: int | None = None


@dataclass
class ParsedDocument:
    doc_id: str          # SHA-256 of file content (used for duplicate detection)
    filename: str
    file_type: str       # "pdf" | "docx" | "txt" | "md"
    pages: list[ParsedPage] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(len(p.text) for p in self.pages)

@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document, carrying the metadata for citations."""

    chunk_id: str        # unique, deterministic: "<doc_id[:16]>_<index>"
    doc_id: str
    filename: str
    page: int | None
    chunk_index: int     # position within the whole document
    text: str



@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by similarity search. `score` is cosine similarity."""

    chunk_id: str
    doc_id: str
    filename: str
    page: int | None
    text: str
    score: float


@dataclass(frozen=True)
class DocumentInfo:
    doc_id: str
    filename: str
    num_chunks: int