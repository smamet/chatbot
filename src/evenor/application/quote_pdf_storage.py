from __future__ import annotations

import json
import mimetypes
import re
import threading
import uuid
from pathlib import Path

from evenor.config.settings import Settings
from evenor.domain.models.outbound_attachment import OutboundAttachment
from evenor.domain.models.pending_reply import PendingReply

TEST_QUOTE_PDF_TTL_SECONDS = 120

_ALLOWED_ATTACHMENT_SUFFIXES = frozenset(
    {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".webp",
        ".docx",
        ".xlsx",
        ".csv",
        ".txt",
        ".md",
    }
)


class AttachmentValidationError(ValueError):
    pass


def safe_quote_filename(quote_name: str) -> str:
    return quote_name.replace("/", "-").strip()


def quote_pdf_dir(settings: Settings, tenant_slug: str) -> Path:
    return settings.data_root / "quotes" / tenant_slug


def outbound_attachment_dir(settings: Settings, tenant_slug: str, reply_id: int) -> Path:
    return settings.data_root / "attachments" / tenant_slug / str(reply_id)


def safe_upload_filename(filename: str) -> str:
    name = Path(filename or "attachment").name.strip()
    name = re.sub(r"[^\w.\- ]+", "_", name).strip("._ ")
    return name or "attachment"


def resolve_attachment_mime_type(filename: str, content_type: str | None = None) -> str:
    if content_type:
        base = content_type.split(";", 1)[0].strip().lower()
        if base and base != "application/octet-stream":
            return base
    guessed, _ = mimetypes.guess_type(filename)
    return guessed or "application/octet-stream"


def is_user_attachment_path(
    settings: Settings,
    tenant_slug: str,
    reply_id: int,
    path: Path,
) -> bool:
    root = outbound_attachment_dir(settings, tenant_slug, reply_id).resolve()
    try:
        return path.resolve().is_relative_to(root)
    except (OSError, ValueError, RuntimeError):
        return False


def parse_attachment_entries(attachments_json: str | None) -> list[dict[str, str]]:
    if not attachments_json:
        return []
    try:
        raw = json.loads(attachments_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    entries: list[dict[str, str]] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        path_str = str(row.get("path", "")).strip()
        filename = str(row.get("filename", "")).strip() or "attachment"
        mime_type = str(row.get("mime_type", "application/octet-stream")).strip()
        if path_str:
            entries.append({"filename": filename, "path": path_str, "mime_type": mime_type})
    return entries


def attachment_entries_total_bytes(entries: list[dict[str, str]]) -> int:
    total = 0
    for row in entries:
        path = Path(row.get("path", ""))
        if path.is_file():
            total += path.stat().st_size
    return total


def partition_attachment_entries(
    attachments_json: str | None,
    *,
    settings: Settings,
    tenant_slug: str,
    reply_id: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    manual: list[dict[str, str]] = []
    quote_entries: list[dict[str, str]] = []
    for row in parse_attachment_entries(attachments_json):
        path = Path(row["path"])
        if is_user_attachment_path(settings, tenant_slug, reply_id, path):
            manual.append(row)
        else:
            quote_entries.append(row)
    return manual, quote_entries


def merge_attachment_entries(
    attachments_json: str | None,
    new_entries: list[dict[str, str]],
) -> str:
    existing = parse_attachment_entries(attachments_json)
    return encode_attachments_json(existing + new_entries)


def remove_attachment_entry(attachments_json: str | None, path_str: str) -> str:
    kept = [row for row in parse_attachment_entries(attachments_json) if row.get("path") != path_str]
    return encode_attachments_json(kept)


def validate_outbound_attachment_upload(
    settings: Settings,
    *,
    filename: str,
    data: bytes,
    content_type: str | None,
    existing_json: str | None,
) -> str:
    safe_name = safe_upload_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in _ALLOWED_ATTACHMENT_SUFFIXES:
        raise AttachmentValidationError(f"File type not allowed: {suffix or '(none)'}")
    if len(data) > settings.attachment_max_bytes:
        raise AttachmentValidationError(
            f"File exceeds maximum size ({settings.attachment_max_bytes} bytes)"
        )
    if not data:
        raise AttachmentValidationError("Empty file")
    existing_bytes = attachment_entries_total_bytes(parse_attachment_entries(existing_json))
    if existing_bytes + len(data) > settings.attachment_max_total_bytes:
        raise AttachmentValidationError(
            f"Total attachments exceed maximum ({settings.attachment_max_total_bytes} bytes)"
        )
    return resolve_attachment_mime_type(safe_name, content_type)


def store_outbound_attachment(
    settings: Settings,
    tenant_slug: str,
    reply_id: int,
    filename: str,
    data: bytes,
    *,
    mime_type: str,
) -> dict[str, str]:
    root = outbound_attachment_dir(settings, tenant_slug, reply_id)
    root.mkdir(parents=True, exist_ok=True)
    safe_name = safe_upload_filename(filename)
    stored_name = f"{uuid.uuid4().hex}_{safe_name}"
    path = root / stored_name
    path.write_bytes(data)
    return attachment_entry(path=path, filename=safe_name, mime_type=mime_type)


def attachment_rows_for_ui(
    attachments_json: str | None,
    *,
    settings: Settings,
    tenant_slug: str,
    reply_id: int,
    quote_name: str | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for entry in parse_attachment_entries(attachments_json):
        path = Path(entry["path"])
        size = path.stat().st_size if path.is_file() else 0
        deletable = is_user_attachment_path(settings, tenant_slug, reply_id, path)
        view_url = attachment_view_url_for_entry(
            entry,
            settings=settings,
            tenant_slug=tenant_slug,
            reply_id=reply_id,
            quote_name=quote_name,
        )
        rows.append(
            {
                "filename": entry["filename"],
                "path": entry["path"],
                "mime_type": entry.get("mime_type", "application/octet-stream"),
                "size_bytes": size,
                "deletable": deletable,
                "is_quote_pdf": not deletable,
                "view_url": view_url,
            }
        )
    return rows


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


def quote_pdf_dashboard_url(tenant_slug: str, quote_name: str, *, inline: bool = False) -> str:
    safe = safe_quote_filename(quote_name)
    url = f"/dashboard/bots/{tenant_slug}/integrations/erpnext/quotation-pdf/{safe}"
    if inline:
        return f"{url}?inline=1"
    return url


def validation_attachment_view_url(tenant_slug: str, reply_id: int, path: str) -> str:
    from urllib.parse import quote

    return (
        f"/dashboard/bots/{tenant_slug}/validation/{reply_id}/attachments/file"
        f"?path={quote(path, safe='')}"
    )


def is_quote_pdf_path(settings: Settings, tenant_slug: str, quote_name: str, path: Path) -> bool:
    expected = quote_pdf_path(settings, tenant_slug, quote_name)
    if expected is None:
        return False
    try:
        return path.resolve() == expected.resolve()
    except (OSError, ValueError, RuntimeError):
        return False


def attachment_view_url_for_entry(
    entry: dict[str, str],
    *,
    settings: Settings,
    tenant_slug: str,
    reply_id: int,
    quote_name: str | None,
) -> str | None:
    path = Path(entry["path"])
    if not path.is_file():
        return None
    if is_user_attachment_path(settings, tenant_slug, reply_id, path):
        return validation_attachment_view_url(tenant_slug, reply_id, entry["path"])
    if quote_name and is_quote_pdf_path(settings, tenant_slug, quote_name, path):
        return quote_pdf_dashboard_url(tenant_slug, quote_name, inline=True)
    return None


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


def load_attachments_from_json(attachments_json: str | None) -> list[OutboundAttachment]:
    if not attachments_json:
        return []
    try:
        raw = json.loads(attachments_json)
    except json.JSONDecodeError:
        return []
    if not isinstance(raw, list):
        return []
    attachments: list[OutboundAttachment] = []
    for row in raw:
        if not isinstance(row, dict):
            continue
        path_str = str(row.get("path", "")).strip()
        filename = str(row.get("filename", "")).strip() or "attachment.pdf"
        mime_type = str(row.get("mime_type", "application/pdf")).strip() or "application/pdf"
        if not path_str:
            continue
        path = Path(path_str)
        if not path.is_file():
            continue
        attachments.append(
            OutboundAttachment(
                filename=filename,
                data=path.read_bytes(),
                mime_type=mime_type,
            )
        )
    return attachments


def delete_attachment_files(attachments_json: str | None) -> None:
    for path in parse_attachment_paths(attachments_json):
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            pass


def cleanup_pending_reply_attachments(reply: PendingReply) -> None:
    delete_attachment_files(reply.attachments_json)
