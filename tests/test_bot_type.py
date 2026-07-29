"""Unit tests for bot_type + TraderSettings."""

from __future__ import annotations

from chatbot.domain.models.tenant import BotType, TenantConfig, TraderSettings


def test_trader_settings_roundtrip():
    cfg = TenantConfig(
        trader=TraderSettings(
            market_profile="cac40",
            symbol="DAX",
            epic="IX.D.DAX.IFMM.IP",
            fundmanager_url="https://fm.example",
            fundmanager_token="secret",
            max_open_positions=3,
        )
    )
    restored = TenantConfig.from_json(cfg.to_json())
    assert restored.trader.symbol == "DAX"
    assert restored.trader.epic == "IX.D.DAX.IFMM.IP"
    assert restored.trader.fundmanager_token == "secret"
    assert restored.trader.max_open_positions == 3


def test_with_trader_replace():
    cfg = TenantConfig()
    updated = cfg.with_trader(symbol="NAS100")
    assert updated.trader.symbol == "NAS100"
    assert cfg.trader.symbol == "CAC40"


def test_bot_type_enum():
    assert BotType.ASSISTANT.value == "assistant"
    assert BotType.TRADER.value == "trader"
