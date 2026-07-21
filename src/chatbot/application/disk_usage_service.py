from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from chatbot.adapters.persistence.tenant_paths import (
    tenant_catalog_dir,
    tenant_docs_dir,
    tenant_lancedb_dir,
)
from chatbot.application.quote_pdf_storage import quote_pdf_dir
from chatbot.config.settings import Settings


@dataclass(frozen=True, slots=True)
class DiskCategoryUsage:
    label: str
    bytes: int
    file_count: int


@dataclass(frozen=True, slots=True)
class TenantDiskUsage:
    categories: tuple[DiskCategoryUsage, ...]
    total_bytes: int
    total_files: int


@dataclass(frozen=True, slots=True)
class HostDiskUsage:
    total_bytes: int
    used_bytes: int
    free_bytes: int


def format_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TiB"


def _dir_usage(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    total = 0
    count = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
                count += 1
            except OSError:
                continue
    return total, count


def _category(label: str, path: Path) -> DiskCategoryUsage:
    size, count = _dir_usage(path)
    return DiskCategoryUsage(label=label, bytes=size, file_count=count)


class DiskUsageService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def tenant_usage(self, slug: str) -> TenantDiskUsage:
        settings = self._settings
        categories = (
            _category("Documents", tenant_docs_dir(settings, slug)),
            _category("Catalog", tenant_catalog_dir(settings, slug)),
            _category("Vector index", tenant_lancedb_dir(settings, slug)),
            _category("Attachments", settings.data_root / "attachments" / slug),
            _category("Quotes", quote_pdf_dir(settings, slug)),
            _category("Backups", settings.data_root / "backups" / slug),
        )
        total_bytes = sum(c.bytes for c in categories)
        total_files = sum(c.file_count for c in categories)
        return TenantDiskUsage(
            categories=categories,
            total_bytes=total_bytes,
            total_files=total_files,
        )

    def host_usage(self) -> HostDiskUsage:
        usage = shutil.disk_usage(self._settings.data_root)
        return HostDiskUsage(
            total_bytes=usage.total,
            used_bytes=usage.used,
            free_bytes=usage.free,
        )
