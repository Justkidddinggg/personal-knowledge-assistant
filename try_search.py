import sys
from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.ingestion.chunking import chunk_document
from app.ingestion.loader import load_document

s = get_settings()
doc = load_document(Path(sys.argv[1]))
chunks = chunk_document(doc, s.chunk_size, s.chunk_overlap)

embedder = get_embedder()
vectors = embedder.embed_documents([c.text for c in chunks])  # done once

query = sys.argv[2]
scores = vectors @ embedder.embed_query(query)
top = scores.argsort()[::-1][: s.top_k]

print(f"Query: {query}\n")
for i in top:
    c = chunks[i]
    print(f"[{scores[i]:.3f}] page={c.page} {c.chunk_id}")
    print(c.text[:300].replace("\n", " "), "\n")