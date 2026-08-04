from __future__ import annotations

from decimal import Decimal


def format_count(value: int | float) -> str:
    n = int(value)
    return f"{n:,}"


def format_count_compact(value: int | float) -> str:
    n = float(max(value, 0))
    for unit, threshold in (("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)):
        if n >= threshold:
            scaled = n / threshold
            if scaled >= 100:
                return f"{scaled:.0f}{unit}"
            if scaled >= 10:
                return f"{scaled:.1f}{unit}"
            text = f"{scaled:.2f}".rstrip("0").rstrip(".")
            return f"{text}{unit}"
    return str(int(n))


def format_count_tooltip(value: int | float) -> str:
    full = format_count(value)
    compact = format_count_compact(value)
    if full == compact:
        return full
    return f"{full} ({compact})"


def format_usd(amount: Decimal | float) -> str:
    value = Decimal(str(amount)).quantize(Decimal("0.01"))
    return f"${value:,.2f}"
