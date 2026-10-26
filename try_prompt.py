import sys

from app.config import get_settings
from app.embeddings.embedder import get_embedder
from app.rag.prompts import build_messages
from app.rag.retriever import Retriever
from app.vectorstore.chroma_store import get_vector_store

s = get_settings()
retriever = Retriever(get_embedder(), get_vector_store(), s.top_k, s.min_retrieval_score)
question = sys.argv[1]
results = retriever.retrieve(question)

if not results:
    print("No chunk passed the score threshold: the LLM would not be called.")
else:
    for m in build_messages(question, results):
        print(f"=== {m['role'].upper()} ===\n{m['content']}\n")
    user_len = len(build_messages(question, results)[1]["content"])
    print(f"(user message length: {user_len} characters)")