from __future__ import annotations


def format_trader_pnl(value: float | int | None, *, signed: bool = True) -> str:
    """Format account-currency PnL (e.g. ``+$15.00``, ``-$5.00``, ``$0.00``).

    Uses enough decimals for tiny FX price-unit leftovers; normal Mini FX /
    index moves render with 2 decimals.
    """
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    av = abs(v)
    if av == 0 or av >= 1:
        decimals = 2
    elif av >= 0.01:
        decimals = 4
    else:
        decimals = 5
    mag = f"{av:.{decimals}f}"
    if v > 0:
        return f"+${mag}" if signed else f"${mag}"
    if v < 0:
        return f"-${mag}"
    return f"${mag}"
