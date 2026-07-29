"""IG FR40 Cash CFD session SoT — weekly hours + Euronext holiday flatten/idle."""

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

# IG France 40 Cash CFD — Europe/London (IG indices CFD product details).
SESSION_TZ = "Europe/London"
SESSION_ZONE = ZoneInfo(SESSION_TZ)
SESSION_SOURCE = "IG France 40 Cash CFD (hardcoded)"
WEEKLY_OPEN_WEEKDAY = 6  # Sunday
WEEKLY_OPEN_TIME = time(23, 2)
WEEKLY_CLOSE_WEEKDAY = 4  # Friday
WEEKLY_CLOSE_TIME = time(22, 0)


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


def _to_london(now: datetime | None) -> datetime:
    current = now or datetime.now(SESSION_ZONE)
    if current.tzinfo is None:
        return current.replace(tzinfo=SESSION_ZONE)
    return current.astimezone(SESSION_ZONE)


def _session_day(london: datetime) -> date:
    """Calendar day whose holiday status applies (Sun after weekly open → Monday)."""
    d = london.date()
    if london.weekday() == WEEKLY_OPEN_WEEKDAY and london.time() >= WEEKLY_OPEN_TIME:
        return d + timedelta(days=1)
    return d


def _in_weekly_window(london: datetime) -> bool:
    """True from Sun 23:02 through Fri 22:00 London (exclusive of close instant)."""
    wd = london.weekday()
    t = london.time()
    if wd == 5:  # Saturday
        return False
    if wd == WEEKLY_OPEN_WEEKDAY:  # Sunday
        return t >= WEEKLY_OPEN_TIME
    if wd == WEEKLY_CLOSE_WEEKDAY:  # Friday
        return t < WEEKLY_CLOSE_TIME
    return True  # Mon–Thu


def is_dealing_open(now: datetime | None = None) -> bool:
    """
    True while the FR40 Cash CFD weekday session is dealing.

    Open Sun 23:02 → Fri 22:00 London; Mon–Thu overnight stays open.
    Closed on Euronext full-closure dates, and from 22:00 London on the eve
    of a non-trading day (holiday eve mirrors Friday weekly close).
    """
    london = _to_london(now)
    if not _in_weekly_window(london):
        return False
    tomorrow = london.date() + timedelta(days=1)
    if london.time() >= WEEKLY_CLOSE_TIME and not is_trading_day(tomorrow):
        return False
    return is_trading_day(_session_day(london))


def _sunday_open_on_or_after(d: date) -> datetime:
    days = (WEEKLY_OPEN_WEEKDAY - d.weekday()) % 7
    return datetime.combine(d + timedelta(days=days), WEEKLY_OPEN_TIME, tzinfo=SESSION_ZONE)


def _next_open_at(london: datetime) -> datetime:
    """Next weekly/holiday reopen at or after ``london`` (London tz)."""
    if is_dealing_open(london):
        return london.replace(second=0, microsecond=0)
    d = london.date()
    for _ in range(21):
        day_start = datetime.combine(d, time(0, 0), tzinfo=SESSION_ZONE)
        if day_start >= london and is_dealing_open(day_start):
            return day_start
        sun_open = datetime.combine(d, WEEKLY_OPEN_TIME, tzinfo=SESSION_ZONE)
        if (
            d.weekday() == WEEKLY_OPEN_WEEKDAY
            and sun_open >= london
            and is_dealing_open(sun_open)
        ):
            return sun_open
        d += timedelta(days=1)
    return _sunday_open_on_or_after(london.date() + timedelta(days=1))


def _next_close_at(london: datetime) -> datetime | None:
    """Next dealing close while currently open (Fri/holiday-eve 22:00 London)."""
    if not is_dealing_open(london):
        return None
    today = london.date()
    for offset in range(8):
        d = today + timedelta(days=offset)
        close_at = datetime.combine(d, WEEKLY_CLOSE_TIME, tzinfo=SESSION_ZONE)
        if close_at <= london:
            continue
        # Close applies on Fri weekly, or any day whose tomorrow is non-trading.
        tomorrow = d + timedelta(days=1)
        if d.weekday() == WEEKLY_CLOSE_WEEKDAY or not is_trading_day(tomorrow):
            if is_dealing_open(close_at - timedelta(minutes=1)):
                return close_at
    return None


def _closed_stretch_close_id(london: datetime) -> str:
    """Stable id for the closed stretch containing ``london`` (must be closed)."""
    d = london.date()
    for _ in range(21):
        close_at = datetime.combine(d, WEEKLY_CLOSE_TIME, tzinfo=SESSION_ZONE)
        if close_at <= london and is_dealing_open(close_at - timedelta(seconds=1)):
            return close_at.strftime("%Y%m%d_%H%M")
        d -= timedelta(days=1)
    return london.strftime("%Y%m%d_%H%M")


def _gap_reasons(from_day: date) -> list[str]:
    """Reasons for the closed stretch starting at ``from_day`` (may be Sat/holiday)."""
    reasons: list[str] = []
    probe = from_day
    for _ in range(14):
        if is_trading_day(probe):
            break
        if probe.weekday() >= 5 and "weekend" not in reasons:
            reasons.append("weekend")
        closures = euronext_closures(probe.year)
        if probe in closures:
            label = f"holiday: {closures[probe]}"
            if label not in reasons:
                reasons.append(label)
        probe += timedelta(days=1)
    return reasons


def flatten_check(
    now: datetime | None = None,
    *,
    close_hhmm: str | None = None,
    lead_minutes: int = 30,
    tz: str | None = None,
) -> dict[str, Any]:
    """
    Detect the pre-close flatten window before a weekend / holiday gap.

    Active when ``now`` is in ``[close - lead, close]`` on a trading day whose
    **next calendar day** is a non-trading day, and dealing is still open.

    Close clock is IG weekly close (22:00 Europe/London). ``close_hhmm`` / ``tz``
    are accepted for backward-compatible call sites but ignored as SoT.
    """
    del close_hhmm, tz  # SoT is hardcoded London weekly close.
    london = _to_london(now)
    today = london.date()
    tomorrow = today + timedelta(days=1)
    close_at = datetime.combine(today, WEEKLY_CLOSE_TIME, tzinfo=SESSION_ZONE)
    lead = max(0, int(lead_minutes))
    window_start = close_at - timedelta(minutes=lead)

    reasons = _gap_reasons(tomorrow)
    next_open_day = tomorrow
    probe = tomorrow
    for _ in range(14):
        if is_trading_day(probe):
            next_open_day = probe
            break
        probe += timedelta(days=1)
    else:
        next_open_day = probe

    next_is_closed = not is_trading_day(tomorrow)
    in_window = window_start <= london <= close_at
    dealing = is_dealing_open(london)
    active = bool(
        is_trading_day(today) and next_is_closed and in_window and reasons and dealing
    )

    minutes_to_close = (close_at - london).total_seconds() / 60.0
    return {
        "active": active,
        "reason": " + ".join(reasons) if reasons else "",
        "reasons": reasons,
        "close_at": close_at.isoformat(),
        "window_start": window_start.isoformat(),
        "minutes_to_close": round(minutes_to_close, 1),
        "next_open_day": next_open_day.isoformat(),
        "now": london.isoformat(),
        "weekday": london.strftime("%A"),
        "tz": SESSION_TZ,
        "dealing_open": dealing,
    }


def session_snapshot(
    now: datetime | None = None,
    *,
    flatten_lead_minutes: int = 30,
    flatten_enabled: bool = True,
) -> dict[str, Any]:
    """
    Unified session view for idle gate + flatten + Live badge.

    ``dealing_open`` drives scheduler idle. Flatten fields only apply while open.
    """
    london = _to_london(now)
    dealing = is_dealing_open(london)
    flatten = (
        flatten_check(london, lead_minutes=flatten_lead_minutes)
        if flatten_enabled
        else {
            "active": False,
            "reason": "",
            "reasons": [],
            "close_at": None,
            "window_start": None,
            "minutes_to_close": None,
            "next_open_day": None,
            "now": london.isoformat(),
            "weekday": london.strftime("%A"),
            "tz": SESSION_TZ,
            "dealing_open": dealing,
        }
    )
    next_open = None if dealing else _next_open_at(london)
    next_close = _next_close_at(london) if dealing else None
    close_id = None if dealing else _closed_stretch_close_id(london)

    return {
        "dealing_open": dealing,
        "now": london.isoformat(),
        "weekday": london.strftime("%A"),
        "tz": SESSION_TZ,
        "source": SESSION_SOURCE,
        "weekly_open": "Sun 23:02 Europe/London",
        "weekly_close": "Fri 22:00 Europe/London",
        "next_open": next_open.isoformat() if next_open else None,
        "next_close": next_close.isoformat() if next_close else None,
        "close_id": close_id,
        "flatten_enabled": bool(flatten_enabled),
        "flatten_now": bool(flatten.get("active")),
        "flatten_reason": flatten.get("reason") or "",
        "flatten_reasons": list(flatten.get("reasons") or []),
        "flatten_close_at": flatten.get("close_at"),
        "flatten_window_start": flatten.get("window_start"),
        "minutes_to_close": flatten.get("minutes_to_close"),
        "next_open_day": flatten.get("next_open_day"),
    }
