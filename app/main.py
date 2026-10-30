import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.ask import router as ask_router
from app.api.documents import router as documents_router
from app.api.errors import register_exception_handlers
from app.config import get_settings
from app.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
settings.ensure_dirs()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # explicit list, never "*"
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)
register_exception_handlers(app)
app.include_router(documents_router)
app.include_router(ask_router)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used for testing and later for Docker."""
    return {"status": "ok", "app": settings.app_name}