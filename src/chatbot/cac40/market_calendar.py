"""Euronext / IG FR40 weekend & holiday flatten window (no external calendar API)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, time
from typing import Any
from zoneinfo import ZoneInfo

from dateutil.easter import easter

# Euronext Paris full-day closures (cash). Bastille Day (14 Jul) etc. stay open.
_FIXED_CLOSURES: tuple[tuple[int, int, str], ...] = (
    (1, 1, "New Year"),
    (5, 1, "Labour Day"),
    (12, 25, "Christmas"),
    (12, 26, "Boxing Day"),
)


def euronext_closures(year: int) -> dict[date, str]:
    """Return {date: reason} for Euronext Paris full closures in ``year``."""
    out: dict[date, str] = {}
    for month, day, name in _FIXED_CLOSURES:
        out[date(year, month, day)] = name
    eas = easter(year)
    out[eas - timedelta(days=2)] = "Good Friday"
    out[eas + timedelta(days=1)] = "Easter Monday"
    return out


def is_trading_day(d: date) -> bool:
    """True if weekday and not a Euronext full closure."""
    if d.weekday() >= 5:
        return False
    return d not in euronext_closures(d.year)


def _parse_hhmm(close_hhmm: str) -> time:
    raw = (close_hhmm or "22:00").strip()
    parts = raw.split(":")
    hour = int(parts[0])
    minute = int(parts[1]) if len(parts) > 1 else 0
    return time(hour=hour, minute=minute, second=0)


def flatten_check(
    now: datetime | None = None,
    *,
    close_hhmm: str = "22:00",
    lead_minutes: int = 30,
    tz: str = "Europe/Paris",
) -> dict[str, Any]:
    """
    Detect the pre-close flatten window before a weekend / holiday gap.

    Active when ``now`` is in ``[close - lead, close]`` on a trading day whose
    **next calendar day** is a non-trading day (Sat, or Friday before Monday holiday, etc.).
    """
    zone = ZoneInfo(tz)
    current = now or datetime.now(zone)
    if current.tzinfo is None:
        current = current.replace(tzinfo=zone)
    else:
        current = current.astimezone(zone)

    today = current.date()
    tomorrow = today + timedelta(days=1)
    close_t = _parse_hhmm(close_hhmm)
    close_at = datetime.combine(today, close_t, tzinfo=zone)
    lead = max(0, int(lead_minutes))
    window_start = close_at - timedelta(minutes=lead)

    # Span from tomorrow until the next trading day (weekend and/or holiday gap).
    reasons: list[str] = []
    probe = tomorrow
    next_open = tomorrow
    for _ in range(14):
        if is_trading_day(probe):
            next_open = probe
            break
        if probe.weekday() >= 5 and "weekend" not in reasons:
            reasons.append("weekend")
        closures = euronext_closures(probe.year)
        if probe in closures:
            label = f"holiday: {closures[probe]}"
            if label not in reasons:
                reasons.append(label)
        probe += timedelta(days=1)
    else:
        next_open = probe

    next_is_closed = not is_trading_day(tomorrow)
    in_window = window_start <= current <= close_at
    active = bool(is_trading_day(today) and next_is_closed and in_window and reasons)

    minutes_to_close = (close_at - current).total_seconds() / 60.0
    return {
        "active": active,
        "reason": " + ".join(reasons) if reasons else "",
        "reasons": reasons,
        "close_at": close_at.isoformat(),
        "window_start": window_start.isoformat(),
        "minutes_to_close": round(minutes_to_close, 1),
        "next_open_day": next_open.isoformat(),
        "now": current.isoformat(),
        "weekday": current.strftime("%A"),
        "tz": tz,
    }
