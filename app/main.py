import logging

from fastapi import FastAPI

from app.config import get_settings
from app.logging_config import setup_logging

settings = get_settings()
setup_logging(settings.log_level)
settings.ensure_dirs()
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used for testing and later for Docker."""
    return {"status": "ok", "app": settings.app_name}