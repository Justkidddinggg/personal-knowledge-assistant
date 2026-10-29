from fastapi.testclient import TestClient

from app.api.ask import get_retriever, get_session_store
from app.main import app
from app.rag.llm import LLMError, get_llm
from app.rag.sessions import SessionStore
from app.schemas.document import SearchResult


def result(n: int) -> SearchResult:
    return SearchResult(f"c{n}", "d1", "paper.pdf", n, f"excerpt {n}", 0.5)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.queries: list[str] = []
        self.last_doc_ids: list[str] | None = None

    def retrieve(self, question: str, doc_ids: list[str] | None = None):
        self.queries.append(question)
        self.last_doc_ids = doc_ids
        return self.results


class FakeLLM:
    """Returns the scripted replies in order (the last one repeats)."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return reply


class DownLLM:
    def chat(self, messages: list[dict[str, str]]) -> str:
        raise LLMError("secret internal detail at http://localhost:11434")


def wire(retriever, llm) -> None:
    """Swap in fakes. The `client` fixture clears all overrides after each test."""
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm
    sessions = SessionStore()
    app.dependency_overrides[get_session_store] = lambda: sessions


def test_ask_returns_answer_with_sources(client: TestClient) -> None:
    wire(FakeRetriever([result(1), result(2)]), FakeLLM("Answer: it works [S2]."))
    response = client.post("/ask", json={"question": "How does it work?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True
    assert body["session_id"]
    assert body["search_query"] == "How does it work?"
    assert [s["label"] for s in body["sources"]] == ["S1", "S2"]
    assert [s["cited"] for s in body["sources"]] == [False, True]
    assert body["sources"][1]["filename"] == "paper.pdf"
    assert body["sources"][1]["page"] == 2
    assert body["invalid_citations"] == []


def test_ask_without_evidence_does_not_call_llm(client: TestClient) -> None:
    llm = FakeLLM("never used")
    wire(FakeRetriever([]), llm)
    body = client.post("/ask", json={"question": "Anything?"}).json()
    assert body["grounded"] is False
    assert body["sources"] == []
    assert llm.calls == 0


def test_ask_reports_invalid_citations(client: TestClient) -> None:
    wire(FakeRetriever([result(1)]), FakeLLM("Answer: see [S1] and [S9]."))
    body = client.post("/ask", json={"question": "q"}).json()
    assert body["invalid_citations"] == [9]


def test_llm_down_is_503_without_leaking_details(client: TestClient) -> None:
    wire(FakeRetriever([result(1)]), DownLLM())
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert "secret" not in response.text
    assert "localhost" not in response.text


def test_blank_question_is_422(client: TestClient) -> None:
    wire(FakeRetriever([]), FakeLLM("x"))
    assert client.post("/ask", json={"question": "   "}).status_code == 422


def test_too_long_question_is_422(client: TestClient) -> None:
    wire(FakeRetriever([]), FakeLLM("x"))
    assert client.post("/ask", json={"question": "x" * 1001}).status_code == 422


def test_doc_ids_are_forwarded(client: TestClient) -> None:
    retriever = FakeRetriever([result(1)])
    wire(retriever, FakeLLM("ok [S1]"))
    client.post("/ask", json={"question": "q", "doc_ids": ["d1"]})
    assert retriever.last_doc_ids == ["d1"]


def test_follow_up_searches_with_rewritten_question(client: TestClient) -> None:
    retriever = FakeRetriever([result(1)])
    # replies in order: 1st answer, condense rewrite, 2nd answer
    llm = FakeLLM("First answer [S1].", "What loss does LLMTime use?", "Second answer [S1].")
    wire(retriever, llm)

    first = client.post("/ask", json={"question": "What loss does MOIRAI use?"}).json()
    assert first["search_query"] == "What loss does MOIRAI use?"
    assert llm.calls == 1  # no history yet -> no rewrite call

    second = client.post(
        "/ask", json={"question": "And the second one?", "session_id": first["session_id"]}
    ).json()
    assert second["session_id"] == first["session_id"]
    assert second["search_query"] == "What loss does LLMTime use?"
    assert retriever.queries == ["What loss does MOIRAI use?", "What loss does LLMTime use?"]
    assert llm.calls == 3


def test_unknown_session_id_starts_a_fresh_session(client: TestClient) -> None:
    wire(FakeRetriever([result(1)]), FakeLLM("ok [S1]"))
    body = client.post("/ask", json={"question": "q", "session_id": "made-up"}).json()
    assert body["session_id"] != "made-up"
    assert body["search_query"] == "q"


def test_refusals_are_not_added_to_history(client: TestClient) -> None:
    llm = FakeLLM("never used")
    wire(FakeRetriever([]), llm)  # nothing retrieved -> refusal
    first = client.post("/ask", json={"question": "q1"}).json()
    client.post("/ask", json={"question": "q2", "session_id": first["session_id"]})
    assert llm.calls == 0  # no history -> no rewrite call either