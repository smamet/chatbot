"""Market session calendars — FR40 Cash CFD + IG FX 24x5."""

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

SESSION_TZ = "Europe/London"
SESSION_ZONE = ZoneInfo(SESSION_TZ)

_FR40_SOURCE = "IG France 40 Cash CFD"
_FR40_OPEN_WEEKDAY = 6  # Sunday
_FR40_OPEN_TIME = time(23, 2)
_FR40_CLOSE_WEEKDAY = 4  # Friday
_FR40_CLOSE_TIME = time(22, 0)

_FX_SOURCE = "IG FX 24x5 (approx)"
_FX_OPEN_WEEKDAY = 6
_FX_OPEN_TIME = time(22, 5)
_FX_CLOSE_WEEKDAY = 4
_FX_CLOSE_TIME = time(21, 55)

# Backward-compatible aliases used by older call sites / tests.
SESSION_SOURCE = _FR40_SOURCE
WEEKLY_OPEN_WEEKDAY = _FR40_OPEN_WEEKDAY
WEEKLY_OPEN_TIME = _FR40_OPEN_TIME
WEEKLY_CLOSE_WEEKDAY = _FR40_CLOSE_WEEKDAY
WEEKLY_CLOSE_TIME = _FR40_CLOSE_TIME


def _calendar_params(calendar_id: str | None) -> dict[str, Any]:
    key = str(calendar_id or "euronext_fr40").strip().lower() or "euronext_fr40"
    if key in ("forex_ig", "fx", "forex"):
        return {
            "id": "forex_ig",
            "source": _FX_SOURCE,
            "open_weekday": _FX_OPEN_WEEKDAY,
            "open_time": _FX_OPEN_TIME,
            "close_weekday": _FX_CLOSE_WEEKDAY,
            "close_time": _FX_CLOSE_TIME,
            "use_euronext_holidays": False,
            "weekly_open_label": "Sun 22:05 Europe/London",
            "weekly_close_label": "Fri 21:55 Europe/London",
        }
    return {
        "id": "euronext_fr40",
        "source": _FR40_SOURCE,
        "open_weekday": _FR40_OPEN_WEEKDAY,
        "open_time": _FR40_OPEN_TIME,
        "close_weekday": _FR40_CLOSE_WEEKDAY,
        "close_time": _FR40_CLOSE_TIME,
        "use_euronext_holidays": True,
        "weekly_open_label": "Sun 23:02 Europe/London",
        "weekly_close_label": "Fri 22:00 Europe/London",
    }


def euronext_closures(year: int) -> dict[date, str]:
    """Return {date: reason} for Euronext Paris full closures in ``year``."""
    out: dict[date, str] = {}
    for month, day, name in _FIXED_CLOSURES:
        out[date(year, month, day)] = name
    eas = easter(year)
    out[eas - timedelta(days=2)] = "Good Friday"
    out[eas + timedelta(days=1)] = "Easter Monday"
    return out


def is_trading_day(d: date, *, calendar_id: str | None = None) -> bool:
    """True if weekday (and not a Euronext full closure for FR40 calendar)."""
    cal = _calendar_params(calendar_id)
    if d.weekday() >= 5:
        return False
    if cal["use_euronext_holidays"] and d in euronext_closures(d.year):
        return False
    return True


def _to_london(now: datetime | None) -> datetime:
    current = now or datetime.now(SESSION_ZONE)
    if current.tzinfo is None:
        return current.replace(tzinfo=SESSION_ZONE)
    return current.astimezone(SESSION_ZONE)


def _session_day(london: datetime, *, open_weekday: int, open_time: time) -> date:
    d = london.date()
    if london.weekday() == open_weekday and london.time() >= open_time:
        return d + timedelta(days=1)
    return d


def _in_weekly_window(
    london: datetime,
    *,
    open_weekday: int,
    open_time: time,
    close_weekday: int,
    close_time: time,
) -> bool:
    wd = london.weekday()
    t = london.time()
    if wd == 5:  # Saturday
        return False
    if wd == open_weekday:
        return t >= open_time
    if wd == close_weekday:
        return t < close_time
    return True  # Mon–Thu (and Sun after open)


def is_dealing_open(now: datetime | None = None, *, calendar_id: str | None = None) -> bool:
    cal = _calendar_params(calendar_id)
    london = _to_london(now)
    if not _in_weekly_window(
        london,
        open_weekday=cal["open_weekday"],
        open_time=cal["open_time"],
        close_weekday=cal["close_weekday"],
        close_time=cal["close_time"],
    ):
        return False
    tomorrow = london.date() + timedelta(days=1)
    if london.time() >= cal["close_time"] and not is_trading_day(
        tomorrow, calendar_id=cal["id"]
    ):
        return False
    return is_trading_day(
        _session_day(
            london, open_weekday=cal["open_weekday"], open_time=cal["open_time"]
        ),
        calendar_id=cal["id"],
    )


def _sunday_open_on_or_after(d: date, *, open_weekday: int, open_time: time) -> datetime:
    days = (open_weekday - d.weekday()) % 7
    return datetime.combine(d + timedelta(days=days), open_time, tzinfo=SESSION_ZONE)


def _next_open_at(london: datetime, *, calendar_id: str | None = None) -> datetime:
    cal = _calendar_params(calendar_id)
    if is_dealing_open(london, calendar_id=cal["id"]):
        return london.replace(second=0, microsecond=0)
    d = london.date()
    for _ in range(21):
        day_start = datetime.combine(d, time(0, 0), tzinfo=SESSION_ZONE)
        if day_start >= london and is_dealing_open(day_start, calendar_id=cal["id"]):
            return day_start
        sun_open = datetime.combine(d, cal["open_time"], tzinfo=SESSION_ZONE)
        if (
            d.weekday() == cal["open_weekday"]
            and sun_open >= london
            and is_dealing_open(sun_open, calendar_id=cal["id"])
        ):
            return sun_open
        d += timedelta(days=1)
    return _sunday_open_on_or_after(
        london.date() + timedelta(days=1),
        open_weekday=cal["open_weekday"],
        open_time=cal["open_time"],
    )


def _next_close_at(london: datetime, *, calendar_id: str | None = None) -> datetime | None:
    cal = _calendar_params(calendar_id)
    if not is_dealing_open(london, calendar_id=cal["id"]):
        return None
    today = london.date()
    for offset in range(8):
        d = today + timedelta(days=offset)
        close_at = datetime.combine(d, cal["close_time"], tzinfo=SESSION_ZONE)
        if close_at <= london:
            continue
        tomorrow = d + timedelta(days=1)
        if d.weekday() == cal["close_weekday"] or not is_trading_day(
            tomorrow, calendar_id=cal["id"]
        ):
            if is_dealing_open(close_at - timedelta(minutes=1), calendar_id=cal["id"]):
                return close_at
    return None


def _closed_stretch_close_id(london: datetime, *, calendar_id: str | None = None) -> str:
    cal = _calendar_params(calendar_id)
    d = london.date()
    for _ in range(21):
        close_at = datetime.combine(d, cal["close_time"], tzinfo=SESSION_ZONE)
        if close_at <= london and is_dealing_open(
            close_at - timedelta(seconds=1), calendar_id=cal["id"]
        ):
            return close_at.strftime("%Y%m%d_%H%M")
        d -= timedelta(days=1)
    return london.strftime("%Y%m%d_%H%M")


def _gap_reasons(from_day: date, *, calendar_id: str | None = None) -> list[str]:
    cal = _calendar_params(calendar_id)
    reasons: list[str] = []
    probe = from_day
    for _ in range(14):
        if is_trading_day(probe, calendar_id=cal["id"]):
            break
        if probe.weekday() >= 5 and "weekend" not in reasons:
            reasons.append("weekend")
        if cal["use_euronext_holidays"]:
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
    calendar_id: str | None = None,
) -> dict[str, Any]:
    del close_hhmm, tz
    cal = _calendar_params(calendar_id)
    london = _to_london(now)
    today = london.date()
    tomorrow = today + timedelta(days=1)
    close_at = datetime.combine(today, cal["close_time"], tzinfo=SESSION_ZONE)
    lead = max(0, int(lead_minutes))
    window_start = close_at - timedelta(minutes=lead)

    reasons = _gap_reasons(tomorrow, calendar_id=cal["id"])
    next_open_day = tomorrow
    probe = tomorrow
    for _ in range(14):
        if is_trading_day(probe, calendar_id=cal["id"]):
            next_open_day = probe
            break
        probe += timedelta(days=1)
    else:
        next_open_day = probe

    next_is_closed = not is_trading_day(tomorrow, calendar_id=cal["id"])
    in_window = window_start <= london <= close_at
    dealing = is_dealing_open(london, calendar_id=cal["id"])
    active = bool(
        is_trading_day(today, calendar_id=cal["id"])
        and next_is_closed
        and in_window
        and reasons
        and dealing
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
    calendar_id: str | None = None,
) -> dict[str, Any]:
    cal = _calendar_params(calendar_id)
    london = _to_london(now)
    dealing = is_dealing_open(london, calendar_id=cal["id"])
    flatten = (
        flatten_check(london, lead_minutes=flatten_lead_minutes, calendar_id=cal["id"])
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
    next_open = None if dealing else _next_open_at(london, calendar_id=cal["id"])
    next_close = _next_close_at(london, calendar_id=cal["id"]) if dealing else None
    close_id = (
        None if dealing else _closed_stretch_close_id(london, calendar_id=cal["id"])
    )

    return {
        "dealing_open": dealing,
        "now": london.isoformat(),
        "weekday": london.strftime("%A"),
        "tz": SESSION_TZ,
        "source": cal["source"],
        "calendar_id": cal["id"],
        "weekly_open": cal["weekly_open_label"],
        "weekly_close": cal["weekly_close_label"],
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
