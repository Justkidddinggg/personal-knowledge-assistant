"""Central application configuration.

All tunable values live here and are loaded from environment variables
(or a local .env file). Never hardcode secrets elsewhere in the codebase.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Personal Knowledge Assistant"
    log_level: str = "INFO"

    # Storage
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    chroma_dir: Path = BASE_DIR / "data" / "chroma"

    # RAG parameters (tuned in Phase 8)
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM (provider-agnostic; used in Phase 5)
    llm_provider: str = "openai"
    llm_api_key: str = ""
    llm_model: str = ""

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (parsed once per process)."""
    return Settings()