from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.application.trader_live_service import (
    SYNC_LOG_MAX,
    adapt_decision_for_replay,
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
from chatbot.trader.config import TraderConfig
from chatbot.trader.hedge_ledger import HedgeLedger
from chatbot.trader.models import (
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
    from chatbot.application.trader_live_service import live_config_path, _write_json

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
    from chatbot.application.trader_live_service import run_live_cycle_now

    save_live_config(settings, "demo-bot", default_live_config())
    from unittest.mock import MagicMock, patch

    session = MagicMock()
    with patch("chatbot.application.trader_live_service.TenantService") as mock_ts:
        tenant = MagicMock()
        tenant.id = 1
        tenant.slug = "demo-bot"
        tenant.is_trader = True
        tenant.config.chat_model = "gemini-2.5-flash"
        tenant.config.trader = MagicMock(
            symbol="CAC40",
            epic="IX.D.CAC.BMU.IP",
            fundmanager_url="",
            fundmanager_token="",
            max_open_positions=4,
            market_profile="cac40",
        )
        mock_ts.return_value.get_by_slug.return_value = tenant
        result = run_live_cycle_now(session, settings, "demo-bot")
    assert result["ok"] is False
    assert result["error"] == "mode_off"
    assert "Off" in result["message"]


def test_trading_banner_none_when_inactive(settings: Settings) -> None:
    from unittest.mock import MagicMock

    from chatbot.application.trader_live_service import resolve_trader_trading_banner

    session = MagicMock()
    assert (
        resolve_trader_trading_banner(
            session,
            settings,
            tenant_id=1,
            slug="demo-bot",
            bot_type="assistant",
        )
        is None
    )


def test_trading_banner_shows_mode_when_active(settings: Settings) -> None:
    from unittest.mock import MagicMock

    from chatbot.application.trader_live_service import resolve_trader_trading_banner

    save_live_config(settings, "demo-bot", {"mode": "live", "ig_connector_ids": [1], "strategy": {}})
    session = MagicMock()
    banner = resolve_trader_trading_banner(
        session,
        settings,
        tenant_id=1,
        slug="demo-bot",
        bot_type="trader",
    )
    assert banner is not None
    assert banner["active"] is True
    assert banner["mode"] == "live"
    assert banner["slug"] == "demo-bot"
    # No stream worker heartbeat in tests → down (or stale/ok if files exist).
    assert banner["stream"] in {"ok", "stale", "down"}


def test_trading_banner_stream_off_when_mode_off(settings: Settings) -> None:
    from unittest.mock import MagicMock

    from chatbot.application.trader_live_service import resolve_trader_trading_banner

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
    banner = resolve_trader_trading_banner(
        MagicMock(),
        settings,
        tenant_id=1,
        slug="demo-bot",
        bot_type="trader",
    )
    assert banner == {"active": True, "mode": "off", "slug": "demo-bot", "stream": "off"}


def test_live_cycle_slot_key_aligns_to_candle_close() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from chatbot.application.trader_live_service import live_cycle_slot_key

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
    from chatbot.application.trader_live_service import clear_live_history, live_dir

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
    books = live_dir(settings, "demo-bot") / "order_books"
    books.mkdir(parents=True, exist_ok=True)
    (books / "orders_1.json").write_text('{"x":"y"}', encoding="utf-8")
    clear_live_history(settings, "demo-bot")
    assert not (books / "orders_1.json").exists()


def test_resolve_selected_ig_requires_selection(settings: Settings) -> None:
    from unittest.mock import MagicMock

    from chatbot.application.trader_live_service import _resolve_selected_ig_connectors

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
    cfg = TraderConfig(order_size=1.0)
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
    from chatbot.application.trader_live_service import _normalize_decision_pnl

    out = _normalize_decision_pnl(
        {"net_upl": 1.5, "realized_session": 12.0, "legs_count": 0}
    )
    assert out["realized"] == 12.0
    assert out["realized_session"] == 12.0


def test_get_live_report_from_journal_cycle(settings: Settings, tmp_path: Path) -> None:
    from chatbot.application.trader_live_service import (
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
    from chatbot.application.trader_live_service import (
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

    from chatbot.application.trader_live_service import (
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

    from chatbot.application.trader_live_service import get_live_report, save_live_config
    from chatbot.application.trader_format import format_trader_pnl
    from chatbot.interfaces.web.templates import dumps_json

    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
    from chatbot.application.trader_live_service import live_decisions_path, _write_json

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
    env.filters["format_trader_pnl"] = format_trader_pnl

    class _Req:
        url = type("U", (), {"path": "/x"})()
        state = type("S", (), {"trader_trading": None})()

    html = env.get_template("trader/run.html").render(
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
    from chatbot.application.trader_live_service import (
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
    from chatbot.trader.models import PositionLeg

    ledger = HedgeLedger(TraderConfig())
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
    # Bare replace_open repair with no true deltas must not log.
    assert (
        append_sync_log_from_payload(
            settings,
            "demo-bot",
            {
                "ts": "2026-07-24T12:10:00+00:00",
                "working_order_sync": {
                    "imported": [],
                    "dropped": [],
                    "changed": False,
                },
                "reconcile": {
                    "repaired": True,
                    "changed": False,
                    "opened": [],
                    "imported": [],
                    "closed": [],
                    "dropped_orders": [],
                    "quarantined": [],
                    "repair": {
                        "mode": "replace_open",
                        "imported_positions": [{"deal_id": "DI1"}],
                        "imported_orders": [{"deal_id": "WO1"}],
                    },
                },
            },
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


def test_append_sync_log_from_payload_logs_true_opened_delta(settings: Settings) -> None:
    assert (
        append_sync_log_from_payload(
            settings,
            "demo-bot",
            {
                "ts": "2026-07-24T13:00:00+00:00",
                "cycle_dir": "20260724_130000",
                "working_order_sync": {"changed": True},
                "reconcile": {
                    "changed": True,
                    "repaired": True,
                    "opened": [{"deal_id": "DI_NEW", "side": "BUY", "size": 1.0}],
                    "imported": [],
                    "closed": [],
                    "repair": {"mode": "replace_open", "changed": True},
                },
            },
        )
        is True
    )
    rows = read_sync_log(settings, "demo-bot")
    assert len(rows) == 1
    assert rows[0]["opened"][0]["deal_id"] == "DI_NEW"
    assert rows[0]["desync"] is True


def test_preview_ig_book_diff_statuses(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    from chatbot.application.trader_live_service import live_state_path, _write_json

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
            "cfg": TraderConfig(),
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
        "chatbot.application.trader_live_service._fetch_ig_snapshot",
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


def test_preview_keeps_wo_attached_tp_in_sync(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attached TP on a pending WO must not show as remove when IG still has it."""
    from chatbot.application.trader_live_service import live_state_path, _write_json

    epic = "CS.D.EURUSD.MINI.IP"
    save_live_config(
        settings,
        "demo-bot",
        {"mode": "live", "ig_connector_ids": [1], "strategy": {}},
    )
    _write_json(
        live_state_path(settings, "demo-bot"),
        {
            "phase": "Flat",
            "positions": [],
            "working_orders": [
                {
                    "id": "o174",
                    "type": "LIMIT",
                    "side": "SELL",
                    "level": 1.14,
                    "size": 1.0,
                    "purpose": "entry",
                    "deal_id": "DIAAAAX7C24B",
                },
                {
                    "id": "o_tp",
                    "type": "LIMIT",
                    "side": "BUY",
                    "level": 1.1385,
                    "size": 1.0,
                    "purpose": "tp",
                    "parent_order_id": "o174",
                    "deal_id": "attached:DIAAAAX7C24B:tp",
                },
            ],
            "closed_trades": [],
        },
    )

    def fake_fetch(session, settings_arg, slug):
        return {
            "ok": True,
            "cfg": TraderConfig(epic=epic),
            "connectors": [(1, {})],
            "primary_id": 1,
            "live_cfg": {"mode": "live"},
            "positions": [],
            "raw_orders": [
                {
                    "dealId": "DIAAAAX7C24B",
                    "epic": epic,
                    "direction": "SELL",
                    "orderType": "LIMIT",
                    "orderLevel": 1.14,
                    "orderSize": 1.0,
                    "limitDistance": 15.0,
                }
            ],
        }

    monkeypatch.setattr(
        "chatbot.application.trader_live_service._fetch_ig_snapshot",
        fake_fetch,
    )
    preview = preview_ig_book(None, settings, "demo-bot")  # type: ignore[arg-type]
    assert preview["ok"] is True
    assert preview["counts"]["remove"] == 0
    assert preview["counts"]["in_sync"] == 2
    by_deal = {}
    for g in preview["groups"]:
        by_deal[g["parent"].get("deal_id")] = g["parent"]["status"]
        for c in g.get("children") or []:
            by_deal[c.get("deal_id")] = c["status"]
    assert by_deal["DIAAAAX7C24B"] == "in_sync"
    assert by_deal["attached:DIAAAAX7C24B:tp"] == "in_sync"


def test_build_live_panel_snapshot_fingerprint(settings: Settings) -> None:
    from chatbot.application.trader_live_service import (
        build_live_panel_snapshot,
        live_journal_dir,
        live_state_path,
        write_live_status,
        _write_json,
    )

    save_live_config(
        settings,
        "demo-bot",
        {"mode": "live", "ig_connector_ids": [1], "strategy": {}},
    )
    write_live_status(settings, "demo-bot", {"last_cycle_at": "2026-07-29T06:00:00+00:00"})
    _write_json(
        live_state_path(settings, "demo-bot"),
        {
            "phase": "Flat",
            "last_price": 1.14,
            "positions": [],
            "working_orders": [
                {
                    "id": "o1",
                    "type": "LIMIT",
                    "side": "SELL",
                    "level": 1.14,
                    "size": 1.0,
                    "purpose": "entry",
                    "deal_id": "DI1",
                }
            ],
        },
    )
    cycle_dir = live_journal_dir(settings, "demo-bot") / "20260729T060000Z"
    cycle_dir.mkdir(parents=True)
    _write_json(
        cycle_dir / "cycle.json",
        {
            "ts": "2026-07-29T06:00:15+00:00",
            "decision": {"analysis": {"bias": "bearish"}},
            "executed": [{}],
            "rejected": [],
            "chart_files": ["chart_15m.png"],
        },
    )

    first = build_live_panel_snapshot(settings, "demo-bot", cycle_limit=3)
    assert first["mode"] == "live"
    assert first["fingerprint"]
    assert len(first["cycles"]) == 1
    assert first["cycles"][0]["cycle_id"] == "20260729T060000Z"
    assert first["book"]["groups"]

    second = build_live_panel_snapshot(settings, "demo-bot", cycle_limit=3)
    assert second["fingerprint"] == first["fingerprint"]

    _write_json(
        live_state_path(settings, "demo-bot"),
        {
            "phase": "Flat",
            "last_price": 1.141,
            "positions": [],
            "working_orders": [
                {
                    "id": "o1",
                    "type": "LIMIT",
                    "side": "SELL",
                    "level": 1.14,
                    "size": 1.0,
                    "purpose": "entry",
                    "deal_id": "DI1",
                },
                {
                    "id": "o_tp",
                    "type": "LIMIT",
                    "side": "BUY",
                    "level": 1.1385,
                    "size": 1.0,
                    "purpose": "tp",
                    "parent_order_id": "o1",
                    "deal_id": "attached:DI1:tp",
                },
            ],
        },
    )
    third = build_live_panel_snapshot(settings, "demo-bot", cycle_limit=3)
    assert third["fingerprint"] != first["fingerprint"]


def test_clear_live_history_clears_sync_log(settings: Settings) -> None:
    save_live_config(settings, "demo-bot", {"mode": "off", "ig_connector_ids": [], "strategy": {}})
    append_sync_log(settings, "demo-bot", {"source": "cycle", "dropped": [1]})
    assert read_sync_log(settings, "demo-bot")
    clear_live_history(settings, "demo-bot")
    assert read_sync_log(settings, "demo-bot") == []


def test_adapt_decision_for_replay_clears_resting_brackets() -> None:
    """Stale cancel ids must not block re-place — clear current entry/hedge first."""
    from chatbot.trader.models import LlmAction, LlmAnalysis, LlmDecision

    ledger = HedgeLedger(config=TraderConfig(symbol="EURUSD"), symbol="EURUSD")
    ledger.last_price = 1.145
    entry = WorkingOrder(
        id="o436",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=1.13763,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
        deal_id="DIAAAAX7LU5QZAF",
    )
    tp = WorkingOrder(
        id="o437",
        type=OrderType.LIMIT,
        side=Side.SELL,
        level=1.1465,
        size=1.0,
        purpose=OrderPurpose.TP,
        parent_order_id="o436",
        deal_id="attached:DIAAAAX7LU5QZAF:tp",
    )
    hedge = WorkingOrder(
        id="o438",
        type=OrderType.STOP,
        side=Side.SELL,
        level=1.13763,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
        deal_id="DIAAAAX7LT7KQA5",
    )
    ledger.place_order(entry)
    ledger.place_order(tp)
    ledger.place_order(hedge)

    stored = LlmDecision(
        analysis=LlmAnalysis(support=1.14, resistance=1.147, bias="long"),
        actions=[
            LlmAction(op="cancel_order", order_id="o404", reason="stale"),
            LlmAction(op="cancel_order", order_id="o405", reason="stale tp"),
            LlmAction(
                op="place_limit",
                side="BUY",
                level=1.1405,
                size=1.0,
                purpose="entry",
            ),
            LlmAction(
                op="place_limit",
                side="SELL",
                level=1.1465,
                size=1.0,
                purpose="tp",
            ),
            LlmAction(
                op="place_stop",
                side="SELL",
                level=1.139,
                size=1.0,
                purpose="hedge_cover",
            ),
        ],
    )
    adapted = adapt_decision_for_replay(stored, ledger)
    cancel_ids = [
        a.order_id for a in adapted.actions if a.op == "cancel_order"
    ]
    assert "o404" not in cancel_ids
    assert "o405" not in cancel_ids
    assert "o436" in cancel_ids
    assert "o438" in cancel_ids
    assert "o437" not in cancel_ids  # cascaded from parent
    assert any(a.op == "place_limit" and a.level == 1.1405 for a in adapted.actions)


def test_build_live_order_index_from_journal(settings: Settings) -> None:
    from chatbot.application.trader_live_service import (
        build_live_order_index,
        live_dir,
        live_journal_dir,
        save_live_config,
    )

    save_live_config(settings, "ord-bot", {"mode": "paper", "ig_connector_ids": [], "strategy": {}})
    root = live_journal_dir(settings, "ord-bot")
    older = root / "20260730_100000"
    newer = root / "20260730_101559"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "cycle.json").write_text(
        __import__("json").dumps(
            {
                "ts": "2026-07-30T10:00:00+00:00",
                "snapshot": {
                    "working_orders": [
                        {
                            "id": "o583",
                            "type": "STOP",
                            "side": "BUY",
                            "level": 1.1475,
                            "size": 1.0,
                            "purpose": "hedge_cover",
                            "deal_id": "DIAAAAX7NW28BAX",
                        }
                    ]
                },
                "executed": [],
                "mirror": [],
            }
        ),
        encoding="utf-8",
    )
    (newer / "cycle.json").write_text(
        __import__("json").dumps(
            {
                "ts": "2026-07-30T10:15:59+00:00",
                "snapshot": {
                    "working_orders": [
                        {
                            "id": "o583",
                            "type": "STOP",
                            "side": "BUY",
                            "level": 1.1475,
                            "size": 1.0,
                            "purpose": "hedge_cover",
                            "deal_id": "",
                        },
                        {
                            "id": "o584",
                            "type": "STOP",
                            "side": "SELL",
                            "level": 1.139,
                            "size": 1.0,
                            "purpose": "hedge_cover",
                            "deal_id": "DIAAAAX7QHZV6AL",
                        },
                    ]
                },
                "executed": ["cancel_order:o582", "place_stop:o584@1.139"],
                "mirror": [
                    {
                        "errors": [
                            "place:o583:IG working order rejected: reason=ATTACHED_ORDER_LEVEL_ERROR "
                            "(BUY STOP @ 1.1475)"
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    # Persist open book with o584 only.
    state = live_dir(settings, "ord-bot") / "state.json"
    state.write_text(
        __import__("json").dumps(
            {
                "working_orders": [
                    {
                        "id": "o584",
                        "type": "STOP",
                        "side": "SELL",
                        "level": 1.139,
                        "size": 1.0,
                        "purpose": "hedge_cover",
                        "deal_id": "DIAAAAX7QHZV6AL",
                    }
                ],
                "positions": [],
            }
        ),
        encoding="utf-8",
    )

    rows = build_live_order_index(settings, "ord-bot")
    by_id = {r["id"]: r for r in rows}
    assert by_id["o583"]["status"] == "rejected"
    assert by_id["o583"]["reject_reason"] == "ATTACHED_ORDER_LEVEL_ERROR"
    assert abs(float(by_id["o583"]["level"]) - 1.1475) < 1e-9
    assert by_id["o584"]["status"] == "open"
    assert by_id["o582"]["status"] == "cancelled"
    # Newest created first (o584/o582 stamped in newer cycle before o583's first_seen).
    assert rows[0]["id"] in ("o584", "o582", "o583")
