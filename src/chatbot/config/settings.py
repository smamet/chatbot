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
        populate_by_name=True,
    )

    dev_mode: bool = Field(default=False, validation_alias="DEV_MODE")
    dev_mail_inject_smtp_host: str = Field(
        default="greenmail", validation_alias="DEV_MAIL_INJECT_SMTP_HOST"
    )
    dev_mail_inject_smtp_port: int = Field(
        default=3025, validation_alias="DEV_MAIL_INJECT_SMTP_PORT"
    )
    dev_mailpit_web_url: str = Field(
        default="http://127.0.0.1:8025", validation_alias="DEV_MAILPIT_WEB_URL"
    )
    session_secret: str = Field(default="change-me-session", validation_alias="SESSION_SECRET")

    admin_token: str = Field(default="", validation_alias="ADMIN_TOKEN")
    app_secret_key: str = Field(default="", validation_alias="APP_SECRET_KEY")
    hook_poll_seconds: int = Field(default=5, validation_alias="HOOK_POLL_SECONDS")
    mail_poll_seconds: int = Field(default=60, validation_alias="MAIL_POLL_SECONDS")
    catalog_poll_seconds: int = Field(default=300, validation_alias="CATALOG_POLL_SECONDS")

    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    chat_model: str = Field(default="gemini-2.5-flash", validation_alias="CHAT_MODEL")
    embedding_model: str = Field(default="gemini-embedding-001", validation_alias="EMBEDDING_MODEL")
    embed_retry_max: int = Field(default=5, validation_alias="EMBED_RETRY_MAX")
    embed_retry_base_429_seconds: float = Field(
        default=30.0, validation_alias="EMBED_RETRY_BASE_429_SECONDS"
    )
    embed_retry_base_503_seconds: float = Field(
        default=5.0, validation_alias="EMBED_RETRY_BASE_503_SECONDS"
    )
    attachment_max_bytes: int = Field(default=10_485_760, validation_alias="ATTACHMENT_MAX_BYTES")
    attachment_max_total_bytes: int = Field(
        default=26_214_400, validation_alias="ATTACHMENT_MAX_TOTAL_BYTES"
    )
    rewrite_model: str = Field(default="gemini-2.5-flash", validation_alias="REWRITE_MODEL")
    email_thread_stale_days: int = Field(default=90, validation_alias="EMAIL_THREAD_STALE_DAYS")
    email_thread_subject_similarity: float = Field(
        default=0.85, validation_alias="EMAIL_THREAD_SUBJECT_SIMILARITY"
    )
    email_thread_llm_enabled: bool = Field(default=False, validation_alias="EMAIL_THREAD_LLM_ENABLED")
    email_thread_llm_min_confidence: float = Field(
        default=0.7, validation_alias="EMAIL_THREAD_LLM_MIN_CONFIDENCE"
    )

    database_url: str = Field(
        default="mysql+pymysql://chatbot:chatbot@127.0.0.1:3306/chatbot",
        validation_alias="DATABASE_URL",
    )
    data_root: Path = Field(default=Path("./data"), validation_alias="DATA_ROOT")
    lancedb_root: Path = Field(default=Path("./data/lancedb"), validation_alias="LANCEDB_ROOT")

    rag_enabled: bool = Field(default=True, validation_alias="RAG_ENABLED")
    rag_rewrite_enabled: bool = Field(default=True, validation_alias="RAG_REWRITE_ENABLED")
    retrieval_language: str = Field(default="en", validation_alias="RETRIEVAL_LANGUAGE")
    rag_top_k: int = Field(default=5, validation_alias="RAG_TOP_K")
    chunk_size: int = Field(default=800, validation_alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=100, validation_alias="CHUNK_OVERLAP")
    rag_rewrite_lang_filter: bool = Field(default=True, validation_alias="RAG_REWRITE_LANG_FILTER")
    rag_verbose: bool = Field(default=False, validation_alias="RAG_VERBOSE")
    lancedb_optimize_after_sync: bool = Field(
        default=True, validation_alias="LANCEDB_OPTIMIZE_AFTER_SYNC"
    )
    lancedb_cleanup_older_than_days: int = Field(
        default=1, validation_alias="LANCEDB_CLEANUP_OLDER_THAN_DAYS"
    )

    gemini_pricing_json: str = Field(default="", validation_alias="GEMINI_PRICING_JSON")
    client_billing_input_per_million_usd: float = Field(
        default=1.0, validation_alias="CLIENT_BILLING_INPUT_PER_MILLION_USD"
    )
    client_billing_output_per_million_usd: float = Field(
        default=3.0, validation_alias="CLIENT_BILLING_OUTPUT_PER_MILLION_USD"
    )
    disk_snapshot_enabled: bool = Field(default=True, validation_alias="DISK_SNAPSHOT_ENABLED")

    order_modification_window_hours: int = Field(
        default=6, validation_alias="ORDER_MODIFICATION_WINDOW_HOURS"
    )

    public_base_url: str = Field(default="", validation_alias="PUBLIC_BASE_URL")

    microsoft_mail_client_id: str = Field(default="", validation_alias="MICROSOFT_MAIL_CLIENT_ID")
    microsoft_mail_client_secret: str = Field(
        default="", validation_alias="MICROSOFT_MAIL_CLIENT_SECRET"
    )
    google_mail_client_id: str = Field(default="", validation_alias="GOOGLE_MAIL_CLIENT_ID")
    google_mail_client_secret: str = Field(default="", validation_alias="GOOGLE_MAIL_CLIENT_SECRET")

    whatsapp_verify_token: str = Field(default="", validation_alias="WHATSAPP_VERIFY_TOKEN")
    whatsapp_app_secret: str = Field(default="", validation_alias="WHATSAPP_APP_SECRET")
    whatsapp_access_token: str = Field(default="", validation_alias="WHATSAPP_ACCESS_TOKEN")
    whatsapp_phone_number_id: str = Field(default="", validation_alias="WHATSAPP_PHONE_NUMBER_ID")
    whatsapp_admin_wa_id: str = Field(default="", validation_alias="WHATSAPP_ADMIN_WA_ID")
    messenger_verify_token: str = Field(default="", validation_alias="MESSENGER_VERIFY_TOKEN")
    messenger_page_access_token: str = Field(default="", validation_alias="MESSENGER_PAGE_ACCESS_TOKEN")
    instagram_verify_token: str = Field(default="", validation_alias="INSTAGRAM_VERIFY_TOKEN")
    instagram_access_token: str = Field(default="", validation_alias="INSTAGRAM_ACCESS_TOKEN")
    instagram_ig_user_id: str = Field(default="", validation_alias="INSTAGRAM_IG_USER_ID")

    @property
    def messenger_effective_verify_token(self) -> str:
        return self.messenger_verify_token.strip() or self.whatsapp_verify_token.strip()

    @property
    def instagram_effective_verify_token(self) -> str:
        return self.instagram_verify_token.strip() or self.whatsapp_verify_token.strip()

    @property
    def lancedb_path(self) -> Path:
        return self.lancedb_root


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
    global _cached_settings, _cached_env_mtime
    mtime = _dotenv_mtime()
    with _lock:
        if _cached_settings is None or mtime != _cached_env_mtime:
            _cached_settings = Settings()
            _cached_env_mtime = mtime
        return _cached_settings


def reset_settings_cache_for_tests() -> None:
    global _cached_settings, _cached_env_mtime
    with _lock:
        _cached_settings = None
        _cached_env_mtime = None
