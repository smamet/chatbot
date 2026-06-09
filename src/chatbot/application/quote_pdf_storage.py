from __future__ import annotations

import json
import threading
from pathlib import Path

from chatbot.config.settings import Settings
from chatbot.domain.models.pending_reply import PendingReply

TEST_QUOTE_PDF_TTL_SECONDS = 120


def safe_quote_filename(quote_name: str) -> str:
    return quote_name.replace("/", "-").strip()


def quote_pdf_dir(settings: Settings, tenant_slug: str) -> Path:
    return settings.data_root / "quotes" / tenant_slug


def store_quote_pdf(
    settings: Settings,
    tenant_slug: str,
    quote_name: str,
    pdf_bytes: bytes,
    *,
    ttl_seconds: int | None = None,
) -> Path:
    root = quote_pdf_dir(settings, tenant_slug)
    root.mkdir(parents=True, exist_ok=True)
    safe = safe_quote_filename(quote_name)
    path = root / f"{safe}.pdf"
    path.write_bytes(pdf_bytes)
    if ttl_seconds is not None and ttl_seconds > 0:
        schedule_quote_pdf_deletion(path, delay_seconds=ttl_seconds)
    return path


def schedule_quote_pdf_deletion(path: Path, *, delay_seconds: int) -> None:
    def _delete() -> None:
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass

    timer = threading.Timer(delay_seconds, _delete)
    timer.daemon = True
    timer.start()


def quote_pdf_path(settings: Settings, tenant_slug: str, quote_name: str) -> Path | None:
    safe = safe_quote_filename(quote_name)
    if not safe:
        return None
    path = quote_pdf_dir(settings, tenant_slug) / f"{safe}.pdf"
    return path if path.is_file() else None


def quote_pdf_dashboard_url(tenant_slug: str, quote_name: str) -> str:
    safe = safe_quote_filename(quote_name)
    return f"/dashboard/bots/{tenant_slug}/integrations/erpnext/quotation-pdf/{safe}"


def attachment_entry(*, path: Path, filename: str, mime_type: str = "application/pdf") -> dict[str, str]:
    return {"filename": filename, "path": str(path), "mime_type": mime_type}


def encode_attachments_json(entries: list[dict[str, str]]) -> str:
    return json.dumps(entries)


def parse_attachment_paths(attachments_json: str | None) -> list[Path]:
    if not attachments_json:
        return []
    try:
        raw = json.loads(attachments_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    paths: list[Path] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        path_str = str(row.get("path", "")).strip()
        if path_str:
            paths.append(Path(path_str))
    return paths


def delete_attachment_files(attachments_json: str | None) -> None:
    for path in parse_attachment_paths(attachments_json):
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def cleanup_pending_reply_attachments(reply: PendingReply) -> None:
    delete_attachment_files(reply.attachments_json)
