"""Orchestrates the query pipeline: retrieve -> prompt -> generate -> verify."""
import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.rag.llm import ChatLLM
from app.rag.prompts import NO_EVIDENCE_ANSWER, build_messages
from app.schemas.document import SearchResult

logger = logging.getLogger(__name__)

_CITATION = re.compile(r"\[(S\d+(?:\s*,\s*S\d+)*)\]")

class SupportsRetrieve(Protocol):
    def retrieve(
        self, question: str, doc_ids: list[str] | None = None
    ) -> list[SearchResult]: ...


@dataclass(frozen=True)
class RAGAnswer:
    answer: str
    sources: list[SearchResult]   # numbered 1..n in the order the LLM saw them
    cited: list[int]              # valid source numbers the LLM cited
    invalid_citations: list[int]  # numbers cited that do not exist
    grounded: bool                # False when we refused for lack of evidence
    retrieval_ms: float
    generation_ms: float


def extract_citations(text: str, num_sources: int) -> tuple[list[int], list[int]]:
    """Return (valid, invalid) citation numbers found in the LLM's answer."""
    found: set[int] = set()
    for match in _CITATION.finditer(text):
        found.update(int(n) for n in re.findall(r"\d+", match.group(1)))
    valid = sorted(n for n in found if 1 <= n <= num_sources)
    invalid = sorted(n for n in found if not 1 <= n <= num_sources)
    return valid, invalid


def answer_question(
    question: str,
    retriever: SupportsRetrieve,
    llm: ChatLLM,
    doc_ids: list[str] | None = None,
    search_query: str | None = None,
) -> RAGAnswer:
    """Retrieve with `search_query` (defaults to the question), answer `question`."""
    t0 = time.perf_counter()
    sources = retriever.retrieve(search_query or question, doc_ids)
    t1 = time.perf_counter()
    retrieval_ms = (t1 - t0) * 1000

    if not sources:  # no evidence: do not even ask the LLM
        return RAGAnswer(NO_EVIDENCE_ANSWER, [], [], [], False, retrieval_ms, 0.0)

    text = llm.chat(build_messages(question, sources))
    generation_ms = (time.perf_counter() - t1) * 1000

    cited, invalid = extract_citations(text, len(sources))
    if invalid:
        logger.warning("LLM cited non-existent source(s): %s", invalid)
    if not cited:
        logger.warning("LLM answer contains no valid citations")

    return RAGAnswer(text, sources, cited, invalid, True, retrieval_ms, generation_ms)