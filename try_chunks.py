import sys
from pathlib import Path

from app.config import get_settings
from app.ingestion.chunking import chunk_document
from app.ingestion.loader import load_document

s = get_settings()
doc = load_document(Path(sys.argv[1]))
chunks = chunk_document(doc, s.chunk_size, s.chunk_overlap)

lengths = [len(c.text) for c in chunks]
print(f"{doc.filename}: {len(doc.pages)} pages -> {len(chunks)} chunks")
print(f"chunk length: min={min(lengths)} avg={sum(lengths)//len(lengths)} max={max(lengths)}")
for c in chunks[:3]:
    print(f"\n--- {c.chunk_id} | page={c.page} | {len(c.text)} chars ---")
    print(c.text)