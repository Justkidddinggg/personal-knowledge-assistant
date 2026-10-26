import sys

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.rag.answering import answer_question
from app.rag.llm import get_llm
from app.rag.retriever import Retriever
from app.vectorstore.chroma_store import get_vector_store

s = get_settings()
retriever = Retriever(get_embedder(), get_vector_store(), s.top_k, s.min_retrieval_score)
out = answer_question(sys.argv[1], retriever, get_llm())

print(out.answer)
print("\nSources:")
for i, r in enumerate(out.sources, start=1):
    mark = "*" if i in out.cited else " "
    print(f" {mark}[{i}] {r.filename} page={r.page} score={r.score:.3f}")
print(f"\ngrounded={out.grounded} cited={out.cited} invalid={out.invalid_citations}")
print(f"retrieval={out.retrieval_ms:.0f}ms generation={out.generation_ms:.0f}ms")