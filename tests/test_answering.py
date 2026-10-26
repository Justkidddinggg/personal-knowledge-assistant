import pytest

from app.rag.answering import answer_question, extract_citations
from app.rag.llm import LLMError
from app.rag.prompts import NO_EVIDENCE_ANSWER
from app.schemas.document import SearchResult


def make_result(n: int) -> SearchResult:
    return SearchResult(f"c{n}", "d1", "paper.pdf", n, f"text {n}", 0.5)


class FakeLLM:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls += 1
        return self.reply


class FailingLLM:
    def chat(self, messages: list[dict[str, str]]) -> str:
        raise LLMError("server down")


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_doc_ids: list[str] | None = None

    def retrieve(self, question: str, doc_ids: list[str] | None = None):
        self.last_doc_ids = doc_ids
        return self.results


def test_extract_citations_handles_formats() -> None:
    valid, invalid = extract_citations("A [S1] B [S2][S3] C [S1, S2] D [S9]", num_sources=3)
    assert valid == [1, 2, 3]
    assert invalid == [9]


def test_document_own_references_are_not_citations() -> None:
    # Papers contain their own "[139]" references. The model copied them in a
    # real run; they must never be mistaken for our [S#] source labels.
    valid, invalid = extract_citations("Pretraining helps [S1][139].", num_sources=5)
    assert valid == [1]
    assert invalid == []


def test_no_evidence_skips_llm() -> None:
    llm = FakeLLM("should never be used")
    out = answer_question("anything", FakeRetriever([]), llm)
    assert out.answer == NO_EVIDENCE_ANSWER
    assert out.grounded is False
    assert llm.calls == 0


def test_answer_reports_sources_and_invalid_citations() -> None:
    llm = FakeLLM("Answer: it works [S1] and also [S5].")
    out = answer_question("q", FakeRetriever([make_result(1), make_result(2)]), llm)
    assert out.grounded is True
    assert len(out.sources) == 2
    assert out.cited == [1]
    assert out.invalid_citations == [5]
    assert out.retrieval_ms >= 0 and out.generation_ms >= 0
    assert llm.calls == 1


def test_llm_error_propagates() -> None:
    with pytest.raises(LLMError):
        answer_question("q", FakeRetriever([make_result(1)]), FailingLLM())


def test_doc_ids_are_passed_to_retriever() -> None:
    retriever = FakeRetriever([make_result(1)])
    answer_question("q", retriever, FakeLLM("ok [S1]"), doc_ids=["d1"])
    assert retriever.last_doc_ids == ["d1"]