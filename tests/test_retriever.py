from app.rag.retriever import Retriever
from app.schemas.document import SearchResult
from tests.fakes import FakeEmbedder


def make_result(n: int, score: float) -> SearchResult:
    return SearchResult(f"c{n}", "d1", "paper.pdf", 1, "some text", score)


class StubStore:
    """Pretends to be the vector store and returns fixed results."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    def query(self, query_embedding, top_k, doc_ids=None):
        return self.results


def test_retriever_filters_low_scores() -> None:
    store = StubStore([make_result(1, 0.6), make_result(2, 0.1)])
    retriever = Retriever(FakeEmbedder(), store, top_k=5, min_score=0.2)  # type: ignore[arg-type]
    kept = retriever.retrieve("question")
    assert [r.chunk_id for r in kept] == ["c1"]