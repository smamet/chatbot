from __future__ import annotations

from pathlib import Path

from evenor.config.settings import Settings


def tenant_docs_dir(settings: Settings, slug: str) -> Path:
    path = settings.data_root / "docs" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def tenant_lancedb_dir(settings: Settings, slug: str) -> Path:
    path = settings.lancedb_root / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def tenant_catalog_dir(settings: Settings, slug: str) -> Path:
    path = settings.data_root / "catalog" / slug
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_catalog_filename(item_code: str) -> str:
    safe = item_code.replace("/", "-").replace("\\", "-").strip()
    return safe or "item"
