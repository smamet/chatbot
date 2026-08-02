from __future__ import annotations

from chatbot.application.trader_live_service import _build_trader_config
from chatbot.trader.ig_ohlc import ig_config_from_connector


def test_ig_config_from_connector_prefers_bot_epic() -> None:
    cfg = ig_config_from_connector(
        {"api_key": "k", "username": "u", "password": "p", "epic": "CS.D.LEGACY.IP"},
        epic="CS.D.EURUSD.MINI.IP",
    )
    assert cfg.epic == "CS.D.EURUSD.MINI.IP"


def test_ig_config_from_connector_legacy_fallback() -> None:
    cfg = ig_config_from_connector(
        {"api_key": "k", "username": "u", "password": "p", "epic": "CS.D.LEGACY.IP"},
    )
    assert cfg.epic == "CS.D.LEGACY.IP"


def test_build_trader_config_uses_bot_epic_not_connector() -> None:
    cfg = _build_trader_config(
        live_cfg={"mode": "off", "strategy": {}, "ig_connector_ids": []},
        integ_cfg={
            "symbol": "EURUSD",
            "epic": "CS.D.EURUSD.MINI.IP",
            "market_profile": "eurusd",
            "point_value": 10000,
            "pnl_currency": "USD",
        },
        primary_ig={
            "api_key": "k",
            "username": "u",
            "password": "p",
            "epic": "IX.D.CAC.BMU.IP",
            "acc_type": "DEMO",
        },
        tenant_slug="eurusd-trader",
        gemini_model="gemini-2.5-flash",
        market_profile="eurusd",
    )
    assert cfg.epic == "CS.D.EURUSD.MINI.IP"
    assert cfg.symbol == "EURUSD"
    assert abs(cfg.point_value - 10000.0) < 1e-9
    assert cfg.pnl_currency == "USD"
