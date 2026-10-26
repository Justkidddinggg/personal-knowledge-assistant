"""Split page text into overlapping chunks, preferring natural boundaries."""
import logging

from app.schemas.document import Chunk, ParsedDocument

logger = logging.getLogger(__name__)

# Try the largest natural boundary first, fall back to smaller ones.
SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split_keep(text: str, sep: str) -> list[str]:
    """Split on `sep` but keep the separator at the end of each part."""
    parts = text.split(sep)
    pieces = [p + sep for p in parts[:-1]] + [parts[-1]]
    return [p for p in pieces if p]


def _split_to_pieces(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively break text into pieces that are each <= chunk_size."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:  # no natural boundary left: hard cut
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, rest = separators[0], separators[1:]
    parts = _split_keep(text, sep)
    if len(parts) == 1:  # separator not present; try a finer one
        return _split_to_pieces(text, rest, chunk_size)

    pieces: list[str] = []
    for part in parts:
        if len(part) > chunk_size:
            pieces.extend(_split_to_pieces(part, rest, chunk_size))
        else:
            pieces.append(part)
    return pieces


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    """Greedily pack pieces into chunks, carrying a tail over as overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for piece in pieces:
        if current and current_len + len(piece) > chunk_size:
            chunks.append("".join(current).strip())
            # Drop pieces from the front until the leftover is a small enough
            # overlap AND the new piece fits.
            while current and (
                current_len > overlap or current_len + len(piece) > chunk_size
            ):
                current_len -= len(current[0])
                current.pop(0)
        current.append(piece)
        current_len += len(piece)

    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    pieces = _split_to_pieces(text, SEPARATORS, chunk_size)
    return _merge_pieces(pieces, chunk_size, overlap)


def chunk_document(doc: ParsedDocument, chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunk each page separately so every chunk keeps its page number."""
    chunks: list[Chunk] = []
    index = 0
    for page in doc.pages:
        for text in split_text(page.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id[:16]}_{index:04d}",
                    doc_id=doc.doc_id,
                    filename=doc.filename,
                    page=page.page,
                    chunk_index=index,
                    text=text,
                )
            )
            index += 1
    logger.info("Chunked %s into %d chunk(s)", doc.filename, len(chunks))
    return chunks