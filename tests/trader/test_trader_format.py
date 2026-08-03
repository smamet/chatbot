from __future__ import annotations

from chatbot.application.trader_format import format_position_side, format_trader_pnl


def test_format_position_side() -> None:
    assert format_position_side("BUY") == "LONG"
    assert format_position_side("SELL") == "SHORT"
    assert format_position_side("long") == "LONG"
    assert format_position_side(None) == "—"


def test_format_trader_pnl_index_style() -> None:
    assert format_trader_pnl(12.5, currency="EUR") == "+€12.50"
    assert format_trader_pnl(-3.0, currency="EUR") == "-€3.00"
    assert format_trader_pnl(0, currency="USD") == "$0.00"
    assert format_trader_pnl(12.5, signed=False, currency="USD") == "$12.50"


def test_format_trader_pnl_fx_price_deltas() -> None:
    assert format_trader_pnl(-0.0005, currency="USD") == "-$0.00050"
    assert format_trader_pnl(0.0015, currency="USD") == "+$0.00150"
    assert format_trader_pnl(0.003625, currency="USD") == "+$0.00363"


def test_format_trader_pnl_mid_range() -> None:
    assert format_trader_pnl(0.025, currency="USD") == "+$0.0250"


def test_format_trader_pnl_none() -> None:
    assert format_trader_pnl(None) == "—"
