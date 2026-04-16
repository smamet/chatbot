from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    whatsapp_verify_token: str = Field(default="", validation_alias="WHATSAPP_VERIFY_TOKEN")
    whatsapp_app_secret: str = Field(default="", validation_alias="WHATSAPP_APP_SECRET")
    whatsapp_access_token: str = Field(default="", validation_alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_phone_number_id: str = Field(default="", validation_alias="WHATSAPP_PHONE_NUMBER_ID")


@lru_cache
def get_settings() -> Settings:
    return Settings()
