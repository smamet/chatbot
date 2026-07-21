from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.application.cac40_live_service import (
    clear_live_history,
    default_live_config,
    load_live_config,
    save_live_config,
    set_live_mode,
)
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import OrderPurpose, OrderType, Side, WorkingOrder
from chatbot.config.settings import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(DATA_ROOT=str(tmp_path), APP_SECRET_KEY="test-secret-key-32chars-minimum!!")


def test_live_config_roundtrip(settings: Settings) -> None:
    saved = save_live_config(
        settings,
        "demo-bot",
        {
            "mode": "paper",
            "ig_connector_ids": [3, 7],
            "strategy": {
                "max_open_positions": 2,
                "order_size": 0.5,
                "prevent_loss_exits": True,
                "llm_trigger_mode": "interval",
                "llm_every_n": 3,
                "llm_every_unit": "1h",
            },
        },
    )
    assert saved["mode"] == "paper"
    assert saved["ig_connector_ids"] == [3, 7]
    assert saved["strategy"]["max_open_positions"] == 2
    assert saved["strategy"]["llm_every_bars"] == 12
    loaded = load_live_config(settings, "demo-bot")
    assert loaded["mode"] == "paper"
    assert loaded["ig_connector_ids"] == [3, 7]
    assert loaded["strategy"]["max_open_positions"] == 2
    assert loaded["strategy"]["order_size"] == 0.5
    assert loaded["strategy"]["prevent_loss_exits"] is True
    assert loaded["strategy"]["llm_every_n"] == 3
    assert loaded["strategy"]["llm_every_unit"] == "1h"


def test_set_live_mode_requires_connectors(settings: Settings) -> None:
    save_live_config(settings, "demo-bot", default_live_config())
    with pytest.raises(ValueError, match="Select at least one"):
        set_live_mode(settings, "demo-bot", "live")


def test_run_cycle_now_rejects_off(settings: Settings) -> None:
    from chatbot.application.cac40_live_service import run_live_cycle_now

    save_live_config(settings, "demo-bot", default_live_config())
    # Minimal fake session/tenant path is heavy; just assert mode_off via load path
    # by calling set_live_mode off and checking message through the guard in function
    # without DB: exercise message construction via mode check by patching tenant lookup.
    from unittest.mock import MagicMock, patch

    session = MagicMock()
    with (
        patch("chatbot.application.cac40_live_service.TenantService") as mock_ts,
        patch("chatbot.application.cac40_live_service.IntegrationService") as mock_is,
    ):
        tenant = MagicMock()
        tenant.id = 1
        tenant.slug = "demo-bot"
        tenant.config.chat_model = "gemini-2.5-flash"
        mock_ts.return_value.get_by_slug.return_value = tenant
        mock_is.return_value.find_active.return_value = MagicMock(config={})
        result = run_live_cycle_now(session, settings, "demo-bot")
    assert result["ok"] is False
    assert result["error"] == "mode_off"
    assert "Off" in result["message"]


def test_trading_banner_none_when_inactive(settings: Settings) -> None:
    from unittest.mock import MagicMock, patch

    from chatbot.application.cac40_live_service import resolve_cac40_trading_banner

    session = MagicMock()
    with patch("chatbot.application.cac40_live_service.IntegrationService") as mock_is:
        mock_is.return_value.find_active.return_value = None
        assert (
            resolve_cac40_trading_banner(
                session,
                settings,
                tenant_id=1,
                slug="demo-bot",
                allowed_integrations=None,
            )
            is None
        )


def test_trading_banner_shows_mode_when_active(settings: Settings) -> None:
    from unittest.mock import MagicMock, patch

    from chatbot.application.cac40_live_service import resolve_cac40_trading_banner

    save_live_config(settings, "demo-bot", {"mode": "paper", "ig_connector_ids": [], "strategy": {}})
    session = MagicMock()
    with patch("chatbot.application.cac40_live_service.IntegrationService") as mock_is:
        mock_is.return_value.find_active.return_value = MagicMock()
        banner = resolve_cac40_trading_banner(
            session,
            settings,
            tenant_id=1,
            slug="demo-bot",
            allowed_integrations=None,
        )
    assert banner == {"active": True, "mode": "paper", "slug": "demo-bot"}


def test_live_cycle_slot_key_aligns_to_candle_close() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from chatbot.application.cac40_live_service import live_cycle_slot_key

    paris = ZoneInfo("Europe/Paris")
    # Before :00:15 → still previous quarter (11:45 close).
    assert live_cycle_slot_key(datetime(2026, 7, 21, 12, 0, 10, tzinfo=paris)).endswith(
        "T11:45:00+02:00"
    )
    # At/after :00:15 → 12:00 close slot.
    assert live_cycle_slot_key(datetime(2026, 7, 21, 12, 0, 15, tzinfo=paris)).endswith(
        "T12:00:00+02:00"
    )
    assert live_cycle_slot_key(datetime(2026, 7, 21, 12, 14, 59, tzinfo=paris)).endswith(
        "T12:00:00+02:00"
    )
    assert live_cycle_slot_key(datetime(2026, 7, 21, 12, 15, 20, tzinfo=paris)).endswith(
        "T12:15:00+02:00"
    )


def test_clear_history_removes_order_books(settings: Settings) -> None:
    from chatbot.application.cac40_live_service import clear_live_history, live_dir

    save_live_config(settings, "demo-bot", {"mode": "paper", "ig_connector_ids": [], "strategy": {}})
    books = live_dir(settings, "demo-bot") / "order_books"
    books.mkdir(parents=True, exist_ok=True)
    (books / "orders_1.json").write_text('{"x":"y"}', encoding="utf-8")
    clear_live_history(settings, "demo-bot")
    assert not (books / "orders_1.json").exists()


def test_resolve_selected_ig_requires_selection(settings: Settings) -> None:
    from unittest.mock import MagicMock

    from chatbot.application.cac40_live_service import _resolve_selected_ig_connectors

    conn_svc = MagicMock()
    connectors, warnings, err = _resolve_selected_ig_connectors(
        conn_svc, 1, {"ig_connector_ids": []}
    )
    assert connectors == []
    assert err == "no_ig_connector"
    conn_svc.find_ig.assert_not_called()


def test_clear_history_blocked_when_live(settings: Settings) -> None:
    save_live_config(
        settings,
        "demo-bot",
        {"mode": "live", "ig_connector_ids": [1], "strategy": {}},
    )
    with pytest.raises(ValueError, match="Cannot clear"):
        clear_live_history(settings, "demo-bot")


def test_ledger_state_roundtrip() -> None:
    cfg = Cac40Config(order_size=1.0)
    ledger = HedgeLedger(config=cfg, symbol="CAC40")
    ledger.last_price = 8000.0
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=7900.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
        )
    )
    state = ledger.to_state_dict()
    restored = HedgeLedger.from_state_dict(cfg, state)
    assert restored.last_price == 8000.0
    assert len(restored.working_orders) == 1
    order = next(iter(restored.working_orders.values()))
    assert order.level == 7900.0
    assert order.side == Side.BUY
