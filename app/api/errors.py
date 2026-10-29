"""One place that decides which HTTP status each domain error maps to."""
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.ingestion.exceptions import (
    CorruptDocumentError,
    EmptyDocumentError,
    FileTooLargeError,
    IngestionError,
    UnsupportedFileTypeError,
)
from app.rag.llm import LLMError

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: dict[type[Exception], int] = {
    UnsupportedFileTypeError: 415,
    FileTooLargeError: 413,
    EmptyDocumentError: 422,
    CorruptDocumentError: 422,
    IngestionError: 400,  # fallback for any other ingestion problem
    LLMError: 503,
}

# For these errors the real message may reveal internals, so users get a generic
# one and the full detail goes to the log only.
_PUBLIC_MESSAGE: dict[type[Exception], str] = {
    LLMError: "The language model is unavailable right now. Please try again shortly.",
}


def _make_handler(status_code: int, public_message: str | None = None):
    async def handler(request: Request, exc: Exception) -> JSONResponse:
        if public_message:
            logger.error("%s: %s", type(exc).__name__, exc)
        return JSONResponse(
            status_code=status_code,
            content={"detail": public_message or str(exc)},
        )

    return handler


def register_exception_handlers(app: FastAPI) -> None:
    for exc_type, status_code in _STATUS_BY_ERROR.items():
        app.add_exception_handler(
            exc_type, _make_handler(status_code, _PUBLIC_MESSAGE.get(exc_type))
        )