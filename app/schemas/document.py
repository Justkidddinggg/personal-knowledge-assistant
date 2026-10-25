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