from __future__ import annotations

from chatbot.trader.instrument_economics import (
    currency_symbol,
    resolve_instrument_economics,
)


def test_currency_symbol() -> None:
    assert currency_symbol("EUR") == "€"
    assert currency_symbol("USD") == "$"
    assert currency_symbol("GBP") == "£"


def test_eurusd_mini_economics_from_ig_fields() -> None:
    market = {
        "instrument": {
            "valueOfOnePip": "1",
            "currencies": [{"code": "USD"}],
        },
        "snapshot": {
            "scalingFactor": 10000,
            "bid": 1.15,
            "offer": 1.1502,
        },
    }
    econ = resolve_instrument_economics(market, account_currency="USD")
    assert abs(econ.point_value - 10000.0) < 1e-9
    assert econ.currency == "USD"
    assert econ.source == "ig"


def test_cac_economics_from_ig_fields() -> None:
    market = {
        "instrument": {
            "valueOfOnePip": "1",
            "currencies": [{"code": "EUR"}],
        },
        "snapshot": {
            "scalingFactor": 1,
            "bid": 7800.0,
            "offer": 7801.0,
        },
    }
    econ = resolve_instrument_economics(market, account_currency="EUR")
    assert abs(econ.point_value - 1.0) < 1e-9
    assert econ.currency == "EUR"


def test_heuristic_fx_without_value_of_one_pip() -> None:
    market = {
        "instrument": {"currencies": [{"code": "USD"}]},
        "snapshot": {"scalingFactor": 10000, "bid": 1.1, "offer": 1.1},
    }
    econ = resolve_instrument_economics(market)
    assert abs(econ.point_value - 10000.0) < 1e-9
    assert econ.source == "heuristic"
