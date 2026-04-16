from __future__ import annotations

from pathlib import Path
from threading import Lock

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    dev_mode: bool = Field(default=False, validation_alias="DEV_MODE")

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    chat_model: str = Field(default="gemini-2.0-flash", validation_alias="CHAT_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", validation_alias="EMBEDDING_MODEL")
    rewrite_model: str = Field(default="gemini-2.0-flash", validation_alias="REWRITE_MODEL")

    database_url: str = Field(default="sqlite:///./data/app.db", validation_alias="DATABASE_URL")
    lancedb_path: Path = Field(default=Path("./data/lancedb"), validation_alias="LANCEDB_PATH")
    prompt_path: Path = Field(default=Path("./prompts/system.md"), validation_alias="PROMPT_PATH")

    rag_enabled: bool = Field(default=False, validation_alias="RAG_ENABLED")
    rag_rewrite_enabled: bool = Field(default=True, validation_alias="RAG_REWRITE_ENABLED")
    retrieval_language: str = Field(default="en", validation_alias="RETRIEVAL_LANGUAGE")
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, validation_alias="CHUNK_OVERLAP")

    rag_rewrite_lang_filter: bool = Field(default=False, validation_alias="RAG_REWRITE_LANG_FILTER")
    fasttext_lid_model_path: Path | None = Field(default=None, validation_alias="FASTTEXT_LID_MODEL_PATH")
    rag_rewrite_min_prob_en: float = Field(default=0.20, validation_alias="RAG_REWRITE_MIN_PROB_EN")
    rag_rewrite_fr_max_prob_creole: float = Field(
        default=0.50,
        validation_alias="RAG_REWRITE_FR_MAX_PROB_CREOLE",
    )
    rag_rewrite_creole_labels: str = Field(default="ht", validation_alias="RAG_REWRITE_CREOLE_LABELS")
    rag_verbose: bool = Field(default=False, validation_alias="RAG_VERBOSE")

    whatsapp_verify_token: str = Field(default="", validation_alias="WHATSAPP_VERIFY_TOKEN")
    whatsapp_app_secret: str = Field(default="", validation_alias="WHATSAPP_APP_SECRET")
    whatsapp_access_token: str = Field(default="", validation_alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_phone_number_id: str = Field(default="", validation_alias="WHATSAPP_PHONE_NUMBER_ID")


_lock = Lock()
_cached_settings: Settings | None = None
_cached_env_mtime: float | None = None


def _dotenv_mtime() -> float:
    path = Path(".env")
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def get_settings() -> Settings:
    """Reload from `.env` when the file's mtime changes (no uvicorn restart needed for most flags)."""
    global _cached_settings, _cached_env_mtime
    mtime = _dotenv_mtime()
    with _lock:
        if _cached_settings is None or mtime != _cached_env_mtime:
            _cached_settings = Settings()
            _cached_env_mtime = mtime
        return _cached_settings


def reset_settings_cache_for_tests() -> None:
    """Clear process-wide settings cache (pytest / isolated runs)."""
    global _cached_settings, _cached_env_mtime
    with _lock:
        _cached_settings = None
        _cached_env_mtime = None
