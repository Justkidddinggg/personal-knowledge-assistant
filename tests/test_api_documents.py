from fastapi.testclient import TestClient

from app.schemas.document import Chunk
from app.vectorstore.chroma_store import VectorStore
from tests.fakes import FakeEmbedder


def add_doc(store: VectorStore, doc_id: str, filename: str, n_chunks: int) -> None:
    chunks = [
        Chunk(f"{doc_id}_{i:04d}", doc_id, filename, 1, i, f"text {i}")
        for i in range(n_chunks)
    ]
    store.add_chunks(chunks, FakeEmbedder().embed_documents([c.text for c in chunks]))


def test_list_documents_empty(client: TestClient) -> None:
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_counts(client: TestClient, store: VectorStore) -> None:
    add_doc(store, "d1", "a.txt", 3)
    add_doc(store, "d2", "b.txt", 1)
    body = client.get("/documents").json()
    assert body == [
        {"doc_id": "d1", "filename": "a.txt", "num_chunks": 3},
        {"doc_id": "d2", "filename": "b.txt", "num_chunks": 1},
    ]


def test_delete_document(client: TestClient, store: VectorStore) -> None:
    add_doc(store, "d1", "a.txt", 2)
    response = client.delete("/documents/d1")
    assert response.status_code == 200
    assert response.json() == {"doc_id": "d1", "deleted_chunks": 2}
    assert client.get("/documents").json() == []


def test_delete_unknown_document_is_404(client: TestClient) -> None:
    assert client.delete("/documents/nope").status_code == 404