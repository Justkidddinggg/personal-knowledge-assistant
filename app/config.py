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

    # Web frontend origins allowed to call the API from a browser (CORS)
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # RAG parameters (tuned in Phase 8)
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5
    min_retrieval_score: float = 0.3  # starting point from 2 test queries; calibrate in Phase 8

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM: any OpenAI-compatible endpoint (Ollama by default)
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.1

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (parsed once per process)."""
    return Settings()