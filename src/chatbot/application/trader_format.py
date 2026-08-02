from __future__ import annotations

from chatbot.trader.instrument_economics import currency_symbol


def format_trader_pnl(
    value: float | int | None,
    *,
    signed: bool = True,
    currency: str | None = "USD",
) -> str:
    """Format account-currency PnL (e.g. ``+€12.50``, ``-$5.00``, ``$0.00``)."""
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
    sym = currency_symbol(currency)
    if v > 0:
        return f"+{sym}{mag}" if signed else f"{sym}{mag}"
    if v < 0:
        return f"-{sym}{mag}"
    return f"{sym}{mag}"
