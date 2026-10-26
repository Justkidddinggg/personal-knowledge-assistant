import sys
import time
from pathlib import Path

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.ingestion.pipeline import ingest_file
from app.vectorstore.chroma_store import get_vector_store

s = get_settings()
store = get_vector_store()
cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
t0 = time.perf_counter()

if cmd == "ingest":
    r = ingest_file(Path(sys.argv[2]), store, get_embedder(), s.chunk_size, s.chunk_overlap)
    print(f"{r.status}: {r.filename} ({r.num_chunks} chunks) doc_id={r.doc_id[:12]}")
elif cmd == "search":
    results = store.query(get_embedder().embed_query(sys.argv[2]), s.top_k)
    for r in results:
        print(f"[{r.score:.3f}] {r.filename} page={r.page} {r.chunk_id}")
        print(r.text[:200].replace("\n", " "), "\n")
elif cmd == "delete":
    print(f"deleted {store.delete_document(sys.argv[2])} chunks")
elif cmd == "list":
    for d in store.list_documents():
        print(f"{d.doc_id}  {d.filename}  ({d.num_chunks} chunks)")
    print(f"total chunks: {store.count()}")

print(f"\n[{cmd}] took {time.perf_counter() - t0:.2f}s")