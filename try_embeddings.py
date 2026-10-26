import time

from app.embeddings.embedder import get_embedder

t0 = time.perf_counter()
embedder = get_embedder()
print(f"model loaded in {time.perf_counter() - t0:.1f}s | max_seq_length={embedder.max_seq_length}")

query = "How do I reset my password?"
texts = [
    "To change your login credentials, open Settings and choose Security.",
    "The cafeteria serves lunch from 11 am to 2 pm.",
    "Passwords must be at least 12 characters long.",
    "Our quarterly revenue grew by 8 percent.",
]

doc_vecs = embedder.embed_documents(texts)
q_vec = embedder.embed_query(query)
print(f"vector shape: {doc_vecs.shape}")

scores = doc_vecs @ q_vec  # normalized vectors -> dot product = cosine similarity
print(f"\nQuery: {query}")
for score, text in sorted(zip(scores, texts), reverse=True):
    print(f"{score:.3f}  {text}")