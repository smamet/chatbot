"""Present IG historical price-point allowance for status / UI."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def present_ig_price_allowance(
    raw: dict[str, Any] | None,
    *,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    """
    Normalize stored allowance into UI-friendly fields.

    IG ``allowanceExpiry`` is seconds until weekly reset **at response time**.
    We subtract elapsed time since ``fetched_at`` when present.
    """
    if not isinstance(raw, dict) or not raw:
        return None
    remaining = raw.get("remaining")
    if remaining is None:
        remaining = raw.get("remainingAllowance")
    total = raw.get("total")
    if total is None:
        total = raw.get("totalAllowance")
    expiry_raw = raw.get("expiry")
    if expiry_raw is None:
        expiry_raw = raw.get("allowanceExpiry")

    clock = now or datetime.now(timezone.utc)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=timezone.utc)

    secs_left: int | None = None
    if expiry_raw is not None:
        try:
            base_secs = int(expiry_raw)
        except (TypeError, ValueError):
            base_secs = None
        if base_secs is not None:
            fetched = _parse_ts(raw.get("fetched_at"))
            if fetched is not None:
                elapsed = max(0.0, (clock - fetched.astimezone(timezone.utc)).total_seconds())
                secs_left = max(0, int(base_secs - elapsed))
            else:
                secs_left = max(0, base_secs)

    resets_at = None
    if secs_left is not None:
        resets_at = (clock + timedelta(seconds=secs_left)).isoformat()

    try:
        remaining_i = int(remaining) if remaining is not None else None
    except (TypeError, ValueError):
        remaining_i = None
    try:
        total_i = int(total) if total is not None else None
    except (TypeError, ValueError):
        total_i = None

    parts: list[str] = []
    if remaining_i is not None and total_i is not None:
        parts.append(f"{remaining_i:,} / {total_i:,} pts")
    elif remaining_i is not None:
        parts.append(f"{remaining_i:,} pts left")
    if secs_left is not None:
        parts.append(f"resets in {_human_duration(secs_left)}")
    if not parts:
        return None

    return {
        "remaining": remaining_i,
        "total": total_i,
        "resets_in_seconds": secs_left,
        "resets_at": resets_at,
        "resets_in_human": _human_duration(secs_left) if secs_left is not None else None,
        "fetched_at": raw.get("fetched_at"),
        "label": " · ".join(parts),
        "exhausted": remaining_i == 0 if remaining_i is not None else False,
    }


def pick_ig_price_allowance(
    *candidates: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Prefer the allowance blob with the newest ``fetched_at``."""
    best: dict[str, Any] | None = None
    best_ts: datetime | None = None
    for raw in candidates:
        if not isinstance(raw, dict) or not raw:
            continue
        ts = _parse_ts(raw.get("fetched_at"))
        if best is None:
            best = raw
            best_ts = ts
            continue
        if ts is not None and (best_ts is None or ts > best_ts):
            best = raw
            best_ts = ts
    return present_ig_price_allowance(best)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


def _human_duration(seconds: int) -> str:
    if seconds <= 0:
        return "now"
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes and not days:
        parts.append(f"{minutes}m")
    if not parts:
        parts.append("<1m")
    return " ".join(parts)
