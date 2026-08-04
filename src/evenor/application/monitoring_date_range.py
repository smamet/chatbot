from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from fastapi import HTTPException

from evenor.adapters.persistence.disk_usage_repository import today_utc

MONITORING_DEFAULT_DAYS = 30
MONITORING_MAX_SPAN_DAYS = 366
MONITORING_PAGE_SIZE = 50


@dataclass(frozen=True, slots=True)
class MonitoringDateRange:
    since: date
    until: date
    usage_days: int
    usage_page: int


def default_monitoring_range() -> MonitoringDateRange:
    until = today_utc()
    since = until - timedelta(days=MONITORING_DEFAULT_DAYS - 1)
    return MonitoringDateRange(
        since=since,
        until=until,
        usage_days=MONITORING_DEFAULT_DAYS,
        usage_page=1,
    )


def parse_monitoring_range(params: Mapping[str, str]) -> MonitoringDateRange:
    today = today_utc()
    raw_from = (params.get("usage_from") or "").strip()
    raw_to = (params.get("usage_to") or "").strip()

    if not raw_from and not raw_to:
        parsed = default_monitoring_range()
        since, until, usage_days = parsed.since, parsed.until, parsed.usage_days
    else:
        if not raw_from or not raw_to:
            raise HTTPException(
                status_code=422,
                detail="Both usage_from and usage_to are required",
            )
        try:
            since = date.fromisoformat(raw_from)
            until = date.fromisoformat(raw_to)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail="Invalid date format (use YYYY-MM-DD)",
            ) from exc
        if since > until:
            raise HTTPException(
                status_code=422,
                detail="usage_from must be on or before usage_to",
            )
        if until > today:
            raise HTTPException(status_code=422, detail="usage_to cannot be in the future")
        usage_days = (until - since).days + 1
        if usage_days > MONITORING_MAX_SPAN_DAYS:
            raise HTTPException(
                status_code=422,
                detail=f"Date range cannot exceed {MONITORING_MAX_SPAN_DAYS} days",
            )

    page_raw = (params.get("usage_page") or "1").strip()
    try:
        usage_page = max(1, int(page_raw))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Invalid usage_page") from exc

    return MonitoringDateRange(
        since=since,
        until=until,
        usage_days=usage_days,
        usage_page=usage_page,
    )


def monitoring_query_string(
    *,
    since: date | None = None,
    until: date | None = None,
    usage_page: int | None = None,
    tab: str | None = None,
) -> str:
    parts: list[str] = []
    if tab:
        parts.append(f"tab={tab}")
    if since is not None:
        parts.append(f"usage_from={since.isoformat()}")
    if until is not None:
        parts.append(f"usage_to={until.isoformat()}")
    if usage_page is not None and usage_page > 1:
        parts.append(f"usage_page={usage_page}")
    return "&".join(parts)
