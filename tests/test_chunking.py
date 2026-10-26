import pytest

from app.ingestion.chunking import chunk_document, split_text
from app.schemas.document import ParsedDocument, ParsedPage


def _sentences(n: int) -> str:
    return " ".join(f"Sentence number {i}." for i in range(n))


def test_short_text_is_one_chunk() -> None:
    assert split_text("Hello world.", chunk_size=100, overlap=20) == ["Hello world."]


def test_chunks_respect_size_limit() -> None:
    chunks = split_text(_sentences(200), chunk_size=100, overlap=30)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_consecutive_chunks_overlap() -> None:
    chunks = split_text(_sentences(200), chunk_size=100, overlap=30)
    assert chunks[0][-8:] in chunks[1]


def test_text_without_separators_is_hard_cut() -> None:
    chunks = split_text("x" * 1000, chunk_size=100, overlap=10)
    assert all(len(c) <= 100 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        split_text("anything", chunk_size=100, overlap=100)


def test_chunk_metadata_and_unique_ids() -> None:
    doc = ParsedDocument(
        doc_id="a" * 64,
        filename="book.pdf",
        file_type="pdf",
        pages=[
            ParsedPage(text=_sentences(60), page=1),
            ParsedPage(text=_sentences(60), page=2),
        ],
    )
    chunks = chunk_document(doc, chunk_size=200, overlap=40)
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.filename == "book.pdf" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)