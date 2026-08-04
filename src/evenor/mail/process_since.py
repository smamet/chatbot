from __future__ import annotations

from datetime import UTC, datetime


def process_since_now_iso() -> str:
    return datetime.now(UTC).replace(second=0, microsecond=0).isoformat()


def parse_process_since(config: dict) -> datetime | None:
    raw = str(config.get("process_since", "")).strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def format_for_datetime_local(value: str | datetime | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        dt = parse_process_since({"process_since": value})
        if dt is None:
            return ""
    else:
        dt = value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    local = dt.astimezone()
    return local.strftime("%Y-%m-%dT%H:%M")


def parse_from_form(local_value: str) -> str:
    raw = local_value.strip()
    if not raw:
        return ""
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(UTC).isoformat()


def format_process_since_display(value: str | None) -> str:
    if not value:
        return "—"
    dt = parse_process_since({"process_since": value})
    if dt is None:
        return "—"
    return dt.astimezone().strftime("%Y-%m-%d %H:%M")


def imap_since_date(process_since: datetime) -> str:
    """IMAP SEARCH SINCE date (day granularity, English month abbrev)."""
    dt = process_since.astimezone(UTC)
    return f"{dt.day:02d}-{dt.strftime('%b')}-{dt.year}"
