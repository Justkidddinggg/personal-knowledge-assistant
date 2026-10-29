"""Endpoint for asking questions about the indexed documents."""
import logging
from functools import lru_cache

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.embeddings.embedder import Embedder, get_embedder
from app.rag.answering import answer_question
from app.rag.condense import condense_question
from app.rag.llm import get_llm
from app.rag.retriever import Retriever
from app.rag.sessions import SessionStore, Turn
from app.schemas.api import AskIn, AskOut, SourceOut
from app.vectorstore.chroma_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@lru_cache
def get_session_store() -> SessionStore:
    return SessionStore()


def get_retriever(
    embedder: Embedder = Depends(get_embedder),
    store: VectorStore = Depends(get_vector_store),
    settings: Settings = Depends(get_settings),
) -> Retriever:
    return Retriever(embedder, store, settings.top_k, settings.min_retrieval_score)


@router.post("/ask", response_model=AskOut)
def ask(
    body: AskIn,
    retriever: Retriever = Depends(get_retriever),
    llm=Depends(get_llm),
    sessions: SessionStore = Depends(get_session_store),
) -> AskOut:
    """Answer a question using only the indexed documents, with citations.

    Pass the returned `session_id` back to ask follow-up questions.
    """
    session_id, history = sessions.start_or_continue(body.session_id, body.doc_ids)

    # Search with a standalone version of the question; answer the original one.
    search_query = condense_question(body.question, history, llm)
    result = answer_question(
        body.question, retriever, llm, body.doc_ids, search_query=search_query
    )

    if result.grounded:  # refusals add nothing useful to the context
        sessions.add_turn(session_id, Turn(body.question, result.answer))

    sources = [
        SourceOut(
            label=f"S{i}",
            doc_id=r.doc_id,
            chunk_id=r.chunk_id,
            filename=r.filename,
            page=r.page,
            score=round(r.score, 3),
            text=r.text,
            cited=i in result.cited,
        )
        for i, r in enumerate(result.sources, start=1)
    ]
    return AskOut(
        answer=result.answer,
        session_id=session_id,
        search_query=search_query,
        grounded=result.grounded,
        sources=sources,
        invalid_citations=result.invalid_citations,
        retrieval_ms=round(result.retrieval_ms, 1),
        generation_ms=round(result.generation_ms, 1),
    )