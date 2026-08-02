from __future__ import annotations

from chatbot.application.trader_format import format_trader_pnl


def test_format_trader_pnl_index_style() -> None:
    assert format_trader_pnl(12.5) == "+$12.50"
    assert format_trader_pnl(-3.0) == "-$3.00"
    assert format_trader_pnl(0) == "$0.00"
    assert format_trader_pnl(12.5, signed=False) == "$12.50"


def test_format_trader_pnl_fx_price_deltas() -> None:
    assert format_trader_pnl(-0.0005) == "-$0.00050"
    assert format_trader_pnl(0.0015) == "+$0.00150"
    assert format_trader_pnl(0.003625) == "+$0.00363"


def test_format_trader_pnl_mid_range() -> None:
    assert format_trader_pnl(0.025) == "+$0.0250"


def test_format_trader_pnl_none() -> None:
    assert format_trader_pnl(None) == "—"
