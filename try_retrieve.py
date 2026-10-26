import sys

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.vectorstore.chroma_store import get_vector_store

s = get_settings()
question = sys.argv[1]
results = get_vector_store().query(get_embedder().embed_query(question), s.top_k)

print(f"Question: {question}")
print(f"Threshold: {s.min_retrieval_score}\n")
for r in results:
    mark = "KEEP" if r.score >= s.min_retrieval_score else "DROP"
    print(f"{mark}  [{r.score:.3f}] page={r.page}  {r.text[:80]!r}")