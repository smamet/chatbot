from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.application.cac40_live_service import (
    SYNC_LOG_MAX,
    append_sync_log,
    append_sync_log_from_payload,
    clear_live_history,
    clear_sync_log,
    default_live_config,
    group_open_book,
    load_live_config,
    preview_ig_book,
    read_live_book,
    read_sync_log,
    save_live_config,
    set_live_mode,
)
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import (
    ClosedTrade,
    LegRole,
    OrderPurpose,
    OrderType,
    Side,
    WorkingOrder,
)
from chatbot.config.settings import Settings


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    return Settings(DATA_ROOT=str(tmp_path), APP_SECRET_KEY="test-secret-key-32chars-minimum!!")


def test_live_config_roundtrip(settings: Settings) -> None:
    saved = save_live_config(
        settings,
        "demo-bot",
        {
            "mode": "live",
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
    assert saved["mode"] == "live"
    assert saved["ig_connector_ids"] == [3, 7]
    assert saved["strategy"]["max_open_positions"] == 2
    assert saved["strategy"]["llm_every_bars"] == 12
    loaded = load_live_config(settings, "demo-bot")
    assert loaded["mode"] == "live"
    assert loaded["ig_connector_ids"] == [3, 7]
    assert loaded["strategy"]["max_open_positions"] == 2
    assert loaded["strategy"]["order_size"] == 0.5
    assert loaded["strategy"]["prevent_loss_exits"] is True
    assert loaded["strategy"]["llm_every_n"] == 3
    assert loaded["strategy"]["llm_every_unit"] == "1h"


def test_legacy_paper_mode_coerces_to_off(settings: Settings) -> None:
    from chatbot.application.cac40_live_service import live_config_path, _write_json

    _write_json(
        live_config_path(settings, "demo-bot"),
        {"mode": "paper", "ig_connector_ids": [1], "strategy": {}},
    )
    loaded = load_live_config(settings, "demo-bot")
    assert loaded["mode"] == "off"
    # Self-heal persisted to disk.
    reloaded = load_live_config(settings, "demo-bot")
    assert reloaded["mode"] == "off"
    saved = save_live_config(
        settings, "demo-bot", {"mode": "paper", "ig_connector_ids": [1], "strategy": {}}
    )
    assert saved["mode"] == "off"
    coerced = set_live_mode(settings, "demo-bot", "paper")
    assert coerced["mode"] == "off"


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

    save_live_config(settings, "demo-bot", {"mode": "live", "ig_connector_ids": [1], "strategy": {}})
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
    assert banner == {"active": True, "mode": "live", "slug": "demo-bot"}


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

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
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


def test_normalize_decision_pnl_adds_realized() -> None:
    from chatbot.application.cac40_live_service import _normalize_decision_pnl

    out = _normalize_decision_pnl(
        {"net_upl": 1.5, "realized_session": 12.0, "legs_count": 0}
    )
    assert out["realized"] == 12.0
    assert out["realized_session"] == 12.0


def test_get_live_report_from_journal_cycle(settings: Settings, tmp_path: Path) -> None:
    from chatbot.application.cac40_live_service import (
        get_live_report,
        list_live_cycles,
        live_journal_dir,
        save_live_config,
    )

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [1], "strategy": {}})
    journal = live_journal_dir(settings, "demo-bot")
    cycle = journal / "20260721_120015"
    charts = cycle / "charts"
    charts.mkdir(parents=True)
    (charts / "chart_15m.png").write_bytes(b"png")
    (cycle / "cycle.json").write_text(
        __import__("json").dumps(
            {
                "ts": "2026-07-21T12:00:15+00:00",
                "cycle_dir": "20260721_120015",
                "decision": {
                    "analysis": {"bias": "long", "support": 1, "resistance": 2},
                    "actions": [{"op": "place"}],
                },
                "executed": ["ok"],
                "rejected": [],
                "charts_rel": "journal/20260721_120015/charts",
                "chart_files": [],
                "pnl": {"net_upl": 0, "realized_session": 3.5},
                "skipped": False,
                "snapshot": {
                    "positions": [],
                    "working_orders": [{"id": "w1"}],
                    "phase": "Flat",
                    "last_price": 8000,
                },
            }
        ),
        encoding="utf-8",
    )
    # Quiet skipped cycle should not appear in list/report.
    quiet = journal / "20260721_121515"
    quiet.mkdir(parents=True)
    (quiet / "cycle.json").write_text(
        __import__("json").dumps(
            {
                "ts": "2026-07-21T12:15:15+00:00",
                "cycle_dir": "20260721_121515",
                "skipped": True,
                "decision": None,
                "charts_rel": "",
                "chart_files": [],
            }
        ),
        encoding="utf-8",
    )

    cycles = list_live_cycles(settings, "demo-bot")
    assert len(cycles) == 1
    assert cycles[0]["cycle_id"] == "20260721_120015"
    assert cycles[0]["has_charts"] is True
    assert cycles[0]["bias"] == "long"

    report = get_live_report(settings, "demo-bot")
    assert len(report["decisions"]) == 1
    d = report["decisions"][0]
    assert d["bias"] == "long"
    assert d["pnl"]["realized"] == 3.5
    assert d["book"]["working_orders"] == 1
    assert d["charts"]
    assert "live/charts/20260721_120015/chart_15m.png" in d["charts"][0]["url"]


def test_get_live_report_merges_decisions_log(settings: Settings) -> None:
    from chatbot.application.cac40_live_service import (
        _write_json,
        get_live_report,
        live_decisions_path,
        live_journal_dir,
        save_live_config,
    )

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [1], "strategy": {}})
    journal = live_journal_dir(settings, "demo-bot")
    cycle = journal / "20260721_130000"
    cycle.mkdir(parents=True)
    (cycle / "cycle.json").write_text(
        __import__("json").dumps(
            {
                "ts": "2026-07-21T13:00:00+00:00",
                "cycle_dir": "20260721_130000",
                "decision": {"analysis": {"bias": "long"}, "actions": []},
                "executed": [],
                "rejected": [],
                "charts_rel": "journal/20260721_130000/charts",
                "chart_files": ["chart_15m.png"],
                "pnl": {"realized_session": 1.0},
                "skipped": False,
            }
        ),
        encoding="utf-8",
    )
    _write_json(
        live_decisions_path(settings, "demo-bot"),
        [
            {
                "ts": "2026-07-21T11:00:00+00:00",
                "cycle_dir": "legacy_only",
                "decision": {"analysis": {"bias": "short"}, "actions": []},
                "executed": [],
                "rejected": [],
                "charts_rel": "",
                "chart_files": [],
                "pnl": {"realized_session": 0.0},
                "skipped": False,
            },
            {
                "ts": "2026-07-21T13:00:00+00:00",
                "cycle_dir": "20260721_130000",
                "decision": {"analysis": {"bias": "long"}, "actions": []},
                "executed": [],
                "rejected": [],
                "charts_rel": "journal/20260721_130000/charts",
                "chart_files": ["chart_15m.png"],
                "pnl": {"realized_session": 1.0},
                "skipped": False,
            },
        ],
    )
    report = get_live_report(settings, "demo-bot")
    biases = [d["bias"] for d in report["decisions"]]
    assert biases == ["long", "short"]
    assert report["report"]["llm_calls_total"] == 2


def test_llm_schedule_persists_and_seeds(settings: Settings) -> None:
    from datetime import datetime, timezone

    from chatbot.application.cac40_live_service import (
        _write_json,
        live_journal_dir,
        load_llm_schedule,
        save_llm_schedule,
        save_live_config,
    )

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [1], "strategy": {}})
    journal = live_journal_dir(settings, "demo-bot")
    cycle = journal / "20260721_120000"
    cycle.mkdir(parents=True)
    (cycle / "cycle.json").write_text(
        __import__("json").dumps(
            {
                "ts": "2026-07-21T12:00:00+00:00",
                "cycle_dir": "20260721_120000",
                "decision": {"analysis": {"bias": "long"}, "actions": []},
                "executed": [],
                "rejected": [],
                "charts_rel": "journal/20260721_120000/charts",
                "chart_files": ["chart_15m.png"],
                "skipped": False,
            }
        ),
        encoding="utf-8",
    )
    seeded = load_llm_schedule(settings, "demo-bot")
    assert seeded["last_llm_at"] == "2026-07-21T12:00:00+00:00"

    save_llm_schedule(
        settings,
        "demo-bot",
        last_llm_at=datetime(2026, 7, 21, 18, 0, tzinfo=timezone.utc),
        every_bars=24,
        mode="interval",
    )
    loaded = load_llm_schedule(settings, "demo-bot")
    assert loaded["last_llm_at"].startswith("2026-07-21T18:00:00")


def test_live_report_render_with_realized_session_only(settings: Settings) -> None:
    """Regression: run.html must not 500 when pnl only has realized_session."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    from chatbot.application.cac40_live_service import get_live_report, save_live_config
    from chatbot.interfaces.web.templates import dumps_json

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
    from chatbot.application.cac40_live_service import live_decisions_path, _write_json

    _write_json(
        live_decisions_path(settings, "demo-bot"),
        [
            {
                "ts": "2026-07-21T12:00:00Z",
                "decision": {"analysis": {"bias": "flat"}, "actions": []},
                "executed": [],
                "rejected": [],
                "charts_rel": "",
                "chart_files": [],
                "pnl": {"net_upl": 0, "realized_session": 0.0},
                "skipped": False,
            }
        ],
    )
    run = get_live_report(settings, "demo-bot")
    assert run["decisions"][0]["pnl"]["realized"] == 0.0

    env = Environment(
        loader=FileSystemLoader("src/chatbot/interfaces/web/templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["dumps_json"] = dumps_json

    class _Req:
        url = type("U", (), {"path": "/x"})()
        state = type("S", (), {"cac40_trading": None})()

    html = env.get_template("cac40/run.html").render(
        request=_Req(),
        user=type("U", (), {"role": "admin"})(),
        tenant=type("T", (), {"slug": "demo-bot", "name": "Demo"})(),
        title="Live",
        run=run,
        live=True,
    )
    assert "Live results" in html
    assert "flat" in html or "no decision" in html or "PnL" in html
    assert "Open book" in html


def test_group_open_book_position_tp_entry_orphan() -> None:
    positions = [
        {
            "id": "p1",
            "side": "BUY",
            "size": 1.0,
            "entry": 7800.0,
            "role": "primary",
            "deal_id": "DI_POS",
            "upl": 12.5,
        }
    ]
    working = [
        {
            "id": "o_tp",
            "type": "LIMIT",
            "side": "SELL",
            "level": 7850.0,
            "size": 1.0,
            "purpose": "tp",
            "position_id": "p1",
            "deal_id": "attached:DI_POS:tp",
        },
        {
            "id": "o_entry",
            "type": "LIMIT",
            "side": "SELL",
            "level": 7900.0,
            "size": 1.0,
            "purpose": "entry",
            "deal_id": "DI_ENTRY",
        },
        {
            "id": "o_entry_tp",
            "type": "LIMIT",
            "side": "BUY",
            "level": 7850.0,
            "size": 1.0,
            "purpose": "tp",
            "parent_order_id": "o_entry",
            "deal_id": "",
        },
        {
            "id": "o_orphan",
            "type": "STOP",
            "side": "SELL",
            "level": 7700.0,
            "size": 1.0,
            "purpose": "hedge_cover",
            "deal_id": "DI_ORPHAN",
        },
    ]
    groups = group_open_book(positions, working)
    assert [g["kind"] for g in groups] == ["position", "entry", "orphan"]

    pos_g = groups[0]
    assert pos_g["parent"]["row_kind"] == "position"
    assert pos_g["parent"]["id"] == "p1"
    assert pos_g["parent"]["level"] == 7800.0
    assert pos_g["parent"]["upl"] == 12.5
    assert len(pos_g["children"]) == 1
    assert pos_g["children"][0]["purpose"] == "tp"
    assert pos_g["children"][0]["link"] == "p1"

    entry_g = groups[1]
    assert entry_g["parent"]["purpose"] == "entry"
    assert len(entry_g["children"]) == 1
    assert entry_g["children"][0]["id"] == "o_entry_tp"
    assert entry_g["children"][0]["link"] == "o_entry"

    orphan_g = groups[2]
    assert orphan_g["parent"]["id"] == "o_orphan"
    assert orphan_g["children"] == []


def test_group_open_book_empty() -> None:
    assert group_open_book([], []) == []


def test_read_live_book_includes_groups(settings: Settings) -> None:
    from chatbot.application.cac40_live_service import (
        live_state_path,
        write_live_status,
        _write_json,
    )

    write_live_status(
        settings,
        "demo-bot",
        {"last_cycle_at": "2026-07-24T12:15:48.151923+00:00"},
    )
    _write_json(
        live_state_path(settings, "demo-bot"),
        {
            "phase": "Long",
            "last_price": 7810.5,
            "positions": [
                {
                    "id": "p1",
                    "side": "BUY",
                    "size": 1.0,
                    "entry": 7800.0,
                    "role": "primary",
                    "deal_id": "DI1",
                    "upl": 5.0,
                }
            ],
            "working_orders": [
                {
                    "id": "o1",
                    "type": "LIMIT",
                    "side": "SELL",
                    "level": 7850.0,
                    "size": 1.0,
                    "purpose": "tp",
                    "position_id": "p1",
                    "deal_id": "attached:DI1:tp",
                }
            ],
        },
    )
    book = read_live_book(settings, "demo-bot")
    assert book["phase"] == "Long"
    assert book["last_price"] == 7810.5
    assert book["as_of"] == "2026-07-24T12:15:48.151923+00:00"
    assert len(book["groups"]) == 1
    assert book["groups"][0]["kind"] == "position"
    assert book["groups"][0]["children"][0]["purpose"] == "tp"


def test_closed_trade_ig_confirmed_roundtrip() -> None:
    trade = ClosedTrade(
        id="p1",
        side=Side.BUY,
        size=1.0,
        entry=7800.0,
        exit=7810.0,
        role=LegRole.PRIMARY,
        realized_pnl=10.0,
        opened_at="",
        closed_at="",
        bars_held=1,
        deal_id="DI1",
        ig_confirmed=True,
    )
    restored = ClosedTrade.from_dict(trade.to_dict())
    assert restored.ig_confirmed is True
    assert restored.deal_id == "DI1"
    legacy = ClosedTrade.from_dict({"id": "p2", "side": "SELL", "size": 1})
    assert legacy.ig_confirmed is False


def test_close_position_ig_confirmed_flag() -> None:
    from chatbot.cac40.models import PositionLeg

    ledger = HedgeLedger(Cac40Config())
    leg = PositionLeg(
        id="p1",
        side=Side.BUY,
        size=1.0,
        entry=7800.0,
        role=LegRole.PRIMARY,
        deal_id="DI1",
    )
    ledger.positions[leg.id] = leg
    ledger.last_price = 7810.0
    trade = ledger.close_position("p1", 7810.0, ig_confirmed=True)
    assert trade is not None
    assert trade.ig_confirmed is True
    paper = ledger.close_position("missing", 0.0)
    assert paper is None


def test_sync_log_append_cap_and_clear(settings: Settings) -> None:
    for i in range(SYNC_LOG_MAX + 5):
        append_sync_log(
            settings,
            "demo-bot",
            {"source": "cycle", "dropped": [{"i": i}], "imported_orders": []},
        )
    rows = read_sync_log(settings, "demo-bot", limit=500)
    assert len(rows) == SYNC_LOG_MAX
    assert rows[0]["dropped"][0]["i"] == SYNC_LOG_MAX + 4
    clear_sync_log(settings, "demo-bot")
    assert read_sync_log(settings, "demo-bot") == []


def test_append_sync_log_from_payload_only_when_changed(settings: Settings) -> None:
    assert (
        append_sync_log_from_payload(
            settings,
            "demo-bot",
            {"ts": "2026-07-24T12:00:00+00:00", "working_order_sync": {}, "reconcile": {}},
        )
        is False
    )
    assert read_sync_log(settings, "demo-bot") == []
    assert (
        append_sync_log_from_payload(
            settings,
            "demo-bot",
            {
                "ts": "2026-07-24T12:15:00+00:00",
                "cycle_dir": "20260724_121500",
                "working_order_sync": {
                    "dropped": [{"order_id": "o1", "deal_id": "DIX"}],
                    "imported": [],
                },
                "reconcile": {},
            },
        )
        is True
    )
    rows = read_sync_log(settings, "demo-bot")
    assert len(rows) == 1
    assert rows[0]["cycle_id"] == "20260724_121500"
    assert len(rows[0]["dropped"]) == 1


def test_preview_ig_book_diff_statuses(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from chatbot.application.cac40_live_service import live_state_path, _write_json

    save_live_config(
        settings,
        "demo-bot",
        {"mode": "live", "ig_connector_ids": [1], "strategy": {}},
    )
    _write_json(
        live_state_path(settings, "demo-bot"),
        {
            "phase": "Long",
            "positions": [
                {
                    "id": "p1",
                    "side": "BUY",
                    "size": 1.0,
                    "entry": 7800.0,
                    "role": "primary",
                    "deal_id": "DI_LOCAL",
                    "upl": 0.0,
                }
            ],
            "working_orders": [
                {
                    "id": "o_gone",
                    "type": "LIMIT",
                    "side": "SELL",
                    "level": 7900.0,
                    "size": 1.0,
                    "purpose": "entry",
                    "deal_id": "DI_GONE",
                }
            ],
            "closed_trades": [
                {
                    "id": "p_old",
                    "side": "BUY",
                    "size": 1.0,
                    "entry": 7700.0,
                    "exit": 7750.0,
                    "role": "primary",
                    "realized_pnl": 50.0,
                    "opened_at": "",
                    "closed_at": "",
                    "bars_held": 2,
                    "deal_id": "DI_REOPEN",
                    "phantom": False,
                    "ig_confirmed": True,
                }
            ],
        },
    )

    def fake_fetch(session, settings_arg, slug):
        return {
            "ok": True,
            "cfg": Cac40Config(),
            "connectors": [(1, {})],
            "primary_id": 1,
            "live_cfg": {"mode": "live"},
            "positions": [
                {
                    "deal_id": "DI_REOPEN",
                    "side": Side.BUY,
                    "size": 1.0,
                    "level": 7805.0,
                    "epic": "IX.D.CAC.BMU.IP",
                    "limit_level": None,
                    "stop_level": None,
                },
                {
                    "deal_id": "DI_NEW",
                    "side": Side.SELL,
                    "size": 1.0,
                    "level": 7820.0,
                    "epic": "IX.D.CAC.BMU.IP",
                    "limit_level": 7780.0,
                    "stop_level": None,
                },
            ],
            "raw_orders": [],
        }

    monkeypatch.setattr(
        "chatbot.application.cac40_live_service._fetch_ig_snapshot",
        fake_fetch,
    )
    preview = preview_ig_book(None, settings, "demo-bot")  # type: ignore[arg-type]
    assert preview["ok"] is True
    statuses = {
        (g["parent"].get("deal_id") or g["parent"].get("id")): g["parent"]["status"]
        for g in preview["groups"]
    }
    assert statuses.get("DI_LOCAL") == "remove"
    assert statuses.get("DI_GONE") == "remove"
    assert statuses.get("DI_NEW") == "new"
    closed = {c["deal_id"]: c["status"] for c in preview["closed_trades"]}
    assert closed["DI_REOPEN"] == "reopened"
    # State file untouched (local still has DI_LOCAL).
    book = read_live_book(settings, "demo-bot")
    assert any(p.get("deal_id") == "DI_LOCAL" for p in book["positions"])


def test_clear_live_history_clears_sync_log(settings: Settings) -> None:
    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
    append_sync_log(settings, "demo-bot", {"source": "cycle", "dropped": [1]})
    assert read_sync_log(settings, "demo-bot")
    clear_live_history(settings, "demo-bot")
    assert read_sync_log(settings, "demo-bot") == []
