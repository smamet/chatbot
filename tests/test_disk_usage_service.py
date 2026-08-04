from __future__ import annotations

from pathlib import Path

import pytest

from evenor.application.disk_usage_service import DiskUsageService, format_bytes
from evenor.config.settings import Settings


def test_format_bytes() -> None:
    assert format_bytes(0) == "0 B"
    assert format_bytes(1024) == "1.0 KiB"
    assert format_bytes(1536) == "1.5 KiB"


def test_tenant_disk_usage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    slug = "demo"
    docs = data / "docs" / slug
    docs.mkdir(parents=True)
    (docs / "a.md").write_text("hello", encoding="utf-8")
    (docs / "b.md").write_bytes(b"x" * 100)

    settings = Settings(
        data_root=data,
        lancedb_root=data / "lancedb",
        gemini_api_key="k",
        admin_token="a",
        app_secret_key="s",
        session_secret="s",
    )
    svc = DiskUsageService(settings)
    usage = svc.tenant_usage(slug)
    docs_cat = next(c for c in usage.categories if c.label == "Documents")
    assert docs_cat.file_count == 2
    assert docs_cat.bytes == len("hello") + 100
    assert usage.total_bytes == docs_cat.bytes


def test_host_disk_usage(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    settings = Settings(
        data_root=data,
        lancedb_root=data / "lancedb",
        gemini_api_key="k",
        admin_token="a",
        app_secret_key="s",
        session_secret="s",
    )
    host = DiskUsageService(settings).host_usage()
    assert host.total_bytes > 0
    assert host.free_bytes >= 0
