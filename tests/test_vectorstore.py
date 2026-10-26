from pathlib import Path

import pytest

from app.schemas.document import Chunk
from app.vectorstore.chroma_store import EmbeddingModelMismatchError, VectorStore
from tests.fakes import FakeEmbedder

embedder = FakeEmbedder()


def make_chunk(doc_id: str, index: int, text: str, page: int | None = 1) -> Chunk:
    return Chunk(
        chunk_id=f"{doc_id}_{index:04d}",
        doc_id=doc_id,
        filename=f"{doc_id}.txt",
        page=page,
        chunk_index=index,
        text=text,
    )


def add(store: VectorStore, chunks: list[Chunk]) -> None:
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))


def test_query_returns_best_match_with_metadata(store: VectorStore) -> None:
    add(store, [
        make_chunk("d1", 0, "apple apple apple fruit"),
        make_chunk("d1", 1, "stock market crash"),
    ])
    results = store.query(embedder.embed_query("apple"), top_k=2)
    assert results[0].chunk_id == "d1_0000"
    assert results[0].filename == "d1.txt"
    assert results[0].page == 1
    assert results[0].score > results[1].score


def test_page_none_round_trips(store: VectorStore) -> None:
    add(store, [make_chunk("d1", 0, "apple fruit", page=None)])
    results = store.query(embedder.embed_query("apple"), top_k=1)
    assert results[0].page is None


def test_delete_document_removes_all_chunks(store: VectorStore) -> None:
    add(store, [make_chunk("d1", 0, "apple"), make_chunk("d1", 1, "banana")])
    assert store.has_document("d1")
    assert store.delete_document("d1") == 2
    assert not store.has_document("d1")
    assert store.count() == 0


def test_filter_by_doc_ids(store: VectorStore) -> None:
    add(store, [make_chunk("d1", 0, "apple fruit"), make_chunk("d2", 0, "apple pie")])
    results = store.query(embedder.embed_query("apple"), top_k=5, doc_ids=["d2"])
    assert {r.doc_id for r in results} == {"d2"}
    assert store.query(embedder.embed_query("apple"), top_k=5, doc_ids=[]) == []


def test_embedding_model_mismatch_is_rejected(tmp_path: Path) -> None:
    VectorStore(tmp_path / "db", embedding_model="model-a")
    with pytest.raises(EmbeddingModelMismatchError):
        VectorStore(tmp_path / "db", embedding_model="model-b")


def test_data_persists_across_instances(tmp_path: Path) -> None:
    first = VectorStore(tmp_path / "db", embedding_model="fake")
    add(first, [make_chunk("d1", 0, "apple fruit")])
    second = VectorStore(tmp_path / "db", embedding_model="fake")
    assert second.has_document("d1")