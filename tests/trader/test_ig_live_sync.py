"""Live IG book sync: no OHLC fills, dealId bind, replace repair, phantom quarantine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from chatbot.application.trader_live_service import (
    adopt_ig_snapshot_into_ledger,
    sync_open_book_from_ig,
)
from chatbot.trader.config import TraderConfig
from chatbot.trader.hedge_ledger import HedgeLedger
from chatbot.trader.models import (
    LegRole,
    OrderPurpose,
    OrderType,
    Side,
    WorkingOrder,
)
from chatbot.trader.scheduler import LiveScheduler


def _cfg(**kwargs) -> TraderConfig:
    base = dict(
        spread_points=0,
        point_value=1.0,
        order_size=1.0,
        max_open_positions=4,
        epic="IX.D.CAC.IFD.IP",
    )
    base.update(kwargs)
    return TraderConfig(**base)


def test_live_process_bar_does_not_close_on_tp_touch(tmp_path: Path):
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig._security = "sec"
    ledger = sched.ig.ledger
    ledger.last_price = 8430
    leg = ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_OPEN")
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8420.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            position_id=leg.id,
            deal_id="WO_TP",
        )
    )
    # Bar touches 8420 — paper would close; live must not.
    events = ledger.process_bar(
        {"open": 8430, "high": 8435, "low": 8415, "close": 8425},
        ts="t1",
        apply_fills=False,
    )
    assert events == []
    assert leg.id in ledger.positions
    assert ledger.legs_count() == 1


def test_sync_drops_vanished_wo_keeps_open_position(tmp_path: Path):
    """replace_open drops vanished WOs; open IG position is re-imported."""
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    ledger = sched.ig.ledger
    ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_OPEN")
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8420.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            deal_id="WO_TP",
        )
    )
    sched.ig.list_open_positions = MagicMock(
        return_value=[
            {
                "deal_id": "DI_OPEN",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8455.0,
            }
        ]
    )
    sched.ig.list_working_orders = MagicMock(return_value=[])  # TP vanished
    out = sched._sync_ledger_from_ig()
    assert out["ran"] is True
    assert out["repaired"] is True
    assert out["changed"] is True
    assert any(d.get("deal_id") == "WO_TP" for d in out.get("dropped_orders") or [])
    assert any(p.deal_id == "DI_OPEN" for p in ledger.positions.values())
    assert not any((o.deal_id or "") == "WO_TP" for o in ledger.working_orders.values())


def test_sync_open_book_unchanged_when_local_matches_ig():
    cfg = _cfg()
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8455
    ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_OPEN")
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8420.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            deal_id="WO_TP",
        )
    )
    out = sync_open_book_from_ig(
        ledger,
        positions=[
            {
                "deal_id": "DI_OPEN",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8455.0,
            }
        ],
        working_orders=[
            {
                "deal_id": "WO_TP",
                "epic": cfg.epic,
                "side": Side.BUY,
                "type": OrderType.LIMIT,
                "level": 8420.0,
                "size": 1.0,
            }
        ],
        epic=cfg.epic,
    )
    assert out["repaired"] is True
    assert out["changed"] is False
    assert out["opened"] == []
    assert out["imported"] == []
    assert out["dropped_orders"] == []
    assert out["closed"] == []
    assert ledger.legs_count() == 1
    assert any(o.deal_id == "WO_TP" for o in ledger.working_orders.values())


def test_sync_open_book_detects_size_drift_and_new_order():
    cfg = _cfg()
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8455
    ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_OPEN")
    out = sync_open_book_from_ig(
        ledger,
        positions=[
            {
                "deal_id": "DI_OPEN",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 2.0,
                "level": 8455.0,
            }
        ],
        working_orders=[
            {
                "deal_id": "WO_NEW",
                "epic": cfg.epic,
                "side": Side.BUY,
                "type": OrderType.STOP,
                "level": 8460.0,
                "size": 2.0,
            }
        ],
        epic=cfg.epic,
    )
    assert out["changed"] is True
    assert any(r.get("deal_id") == "DI_OPEN" and r.get("size") == 2.0 for r in out["opened"])
    assert any(r.get("deal_id") == "WO_NEW" for r in out["imported"])
    leg = next(iter(ledger.positions.values()))
    assert leg.size == 2.0


def test_sync_closes_when_ig_position_gone(tmp_path: Path):
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    ledger = sched.ig.ledger
    ledger.last_price = 8420
    ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_GONE")
    sched.ig.list_open_positions = MagicMock(return_value=[])
    sched.ig.list_working_orders = MagicMock(return_value=[])
    out = sched._sync_ledger_from_ig()
    assert out["ran"] is True
    assert out["changed"] is True
    assert out["closed"]
    assert ledger.legs_count() == 0
    assert ledger.realized_session == 35.0  # 8455-8420


def test_sync_repairs_local_flat_ig_short(tmp_path: Path):
    from chatbot.trader.models import ClosedTrade

    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    ledger = sched.ig.ledger
    ledger.last_price = 8437
    ledger.realized_session = 35.0
    ledger.cash = 35.0
    ledger.closed_trades.append(
        ClosedTrade(
            id="p3",
            side=Side.SELL,
            size=1.0,
            entry=8455.0,
            exit=8420.0,
            role=LegRole.PRIMARY,
            realized_pnl=35.0,
            opened_at="",
            closed_at="",
            bars_held=1,
            deal_id="DI_STILL_OPEN",
        )
    )
    assert ledger.legs_count() == 0

    sched.ig.list_open_positions = MagicMock(
        return_value=[
            {
                "deal_id": "DI_STILL_OPEN",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8455.0,
                "stop_level": 8458.0,
                "limit_level": 8425.0,
            }
        ]
    )
    sched.ig.list_working_orders = MagicMock(return_value=[])
    out = sched._sync_ledger_from_ig()
    assert out["repaired"] or out["opened"]
    assert ledger.legs_count() == 1
    leg = next(iter(ledger.positions.values()))
    assert leg.deal_id == "DI_STILL_OPEN"
    assert any(t.phantom for t in ledger.closed_trades)
    assert abs(ledger.realized_session) < 1e-9


def test_replace_open_quarantines_phantom():
    cfg = _cfg()
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8437
    from chatbot.trader.models import ClosedTrade

    ledger.realized_session = 35.0
    ledger.cash = 35.0
    ledger.closed_trades.append(
        ClosedTrade(
            id="p3",
            side=Side.SELL,
            size=1.0,
            entry=8455.0,
            exit=8420.0,
            role=LegRole.PRIMARY,
            realized_pnl=35.0,
            opened_at="",
            closed_at="",
            bars_held=1,
            deal_id="DI1",
        )
    )
    result = adopt_ig_snapshot_into_ledger(
        ledger,
        positions=[
            {
                "deal_id": "DI1",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8455.0,
                "limit_level": 8425.0,
                "stop_level": 8458.0,
            }
        ],
        working_orders=[],
        epic=cfg.epic,
        mode="replace_open",
    )
    assert result["replaced"] is True
    assert result["quarantined"] == ["p3"]
    assert ledger.legs_count() == 1
    assert abs(ledger.realized_session) < 1e-9
    # Attached stop/limit modeled locally
    purposes = {o.purpose for o in ledger.working_orders.values()}
    assert OrderPurpose.TP in purposes
    assert OrderPurpose.HEDGE_COVER in purposes


def test_market_close_live_posts_ig(tmp_path: Path):
    cfg = _cfg()
    from chatbot.trader.ig_connector import IgConnector

    conn = IgConnector(cfg, dry_run=False)
    conn._cst = "cst"
    conn._security = "sec"
    conn.ledger.last_price = 8430
    leg = conn.ledger._open_leg(
        Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_CLOSE"
    )
    conn.resolve_order_currency = MagicMock(return_value="EUR")
    conn.resolve_order_expiry = MagicMock(return_value="-")

    class _Resp:
        is_error = False
        content = b'{"dealReference":"REF1"}'
        request = MagicMock(url="http://x")

        def json(self):
            return {"dealReference": "REF1"}

    conn._client = MagicMock()
    conn._client.post = MagicMock(return_value=_Resp())
    conn.confirm_deal = MagicMock(
        return_value={"dealStatus": "ACCEPTED", "level": 8430.0, "dealId": "DI_CLOSE"}
    )
    conn.market_close(leg.id)
    assert conn._client.post.called
    headers = conn._client.post.call_args.kwargs.get("headers") or {}
    assert headers.get("_method") == "DELETE"
    assert headers.get("VERSION") == "1"
    body = conn._client.post.call_args.kwargs.get("json") or {}
    assert body.get("dealId") == "DI_CLOSE"
    assert body.get("direction") == "BUY"
    # Close-by-dealId must not carry open-position fields.
    assert "epic" not in body
    assert "expiry" not in body
    assert "guaranteedStop" not in body
    assert leg.id not in conn.ledger.positions


def test_mirror_entry_with_tp_child_sends_limit_level(tmp_path: Path):
    """Same-cycle entry+TP must push IG working order with attached limitLevel."""
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    sched.ig.list_working_orders = MagicMock(return_value=[])
    ledger = sched.ig.ledger
    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=8455.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
        )
    )
    tp = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8425.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            parent_order_id=entry.id,
        )
    )
    hedge = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=8470.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            parent_order_id=entry.id,
        )
    )

    def _push(order, *, currency=None, limit_level=None, stop_level=None):
        assert stop_level is None  # never IG attached stop-loss
        if order.id == entry.id:
            assert limit_level == 8425.0
            order.deal_id = "WO_ENTRY_1"
            sched.ig.last_ig_result = {
                "deal_status": "ACCEPTED",
                "limit_level": limit_level,
            }
        elif order.id == hedge.id:
            assert order.purpose == OrderPurpose.HEDGE_COVER
            assert limit_level is None
            order.deal_id = "WO_HEDGE_1"
            sched.ig.last_ig_result = {"deal_status": "ACCEPTED"}
        else:
            raise AssertionError(f"unexpected push for {order.id}")
        return order

    sched.ig.push_working_order = MagicMock(side_effect=_push)
    from chatbot.trader.risk_gate import GateResult

    sched._mirror_orders_to_ig(GateResult(executed=[f"place_limit:{entry.id}"]))
    assert sched.ig.push_working_order.call_count == 2
    book = sched._load_order_book(0)
    assert book.get(entry.id) == "WO_ENTRY_1"
    assert book.get(tp.id) == "attached:WO_ENTRY_1:tp"
    assert tp.deal_id == "attached:WO_ENTRY_1:tp"
    assert book.get(hedge.id) == "WO_HEDGE_1"  # same cycle, forceOpen STOP
    vias = {p["order_id"]: p.get("via") for p in sched.last_mirror_results[0]["placed"]}
    assert vias.get(hedge.id) == "force_open_hedge"
    errs = sched.last_mirror_results[0].get("errors") or []
    assert not errs


def test_mirror_hedge_force_opens_after_primary(tmp_path: Path):
    """hedge_cover on an open primary is a forceOpen reverse STOP, not stopLevel."""
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    sched.ig.list_working_orders = MagicMock(return_value=[])
    ledger = sched.ig.ledger
    leg = ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_P")
    hedge = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=8470.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            position_id=leg.id,
        )
    )
    sched.ig.update_position_protection = MagicMock(
        side_effect=AssertionError("must not attach stopLevel for hedge")
    )

    def _push(order, *, currency=None, limit_level=None, stop_level=None):
        assert order.id == hedge.id
        assert order.purpose == OrderPurpose.HEDGE_COVER
        assert stop_level is None
        assert limit_level is None
        order.deal_id = "WO_HEDGE_1"
        return order

    sched.ig.push_working_order = MagicMock(side_effect=_push)
    from chatbot.trader.risk_gate import GateResult

    sched._mirror_orders_to_ig(GateResult(executed=[f"place_stop:{hedge.id}"]))
    assert sched.ig.push_working_order.called
    book = sched._load_order_book(0)
    assert book.get(hedge.id) == "WO_HEDGE_1"
    via = sched.last_mirror_results[0]["placed"][0].get("via")
    assert via == "force_open_hedge"


def test_mirror_attaches_tp_instead_of_force_open(tmp_path: Path):
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    sched.ig.list_working_orders = MagicMock(return_value=[])
    ledger = sched.ig.ledger
    leg = ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_P")
    tp = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8425.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            position_id=leg.id,
        )
    )
    sched.ig.update_position_protection = MagicMock(return_value={})
    sched.ig.push_working_order = MagicMock(side_effect=AssertionError("must not push TP"))
    from chatbot.trader.risk_gate import GateResult

    sched._mirror_orders_to_ig(GateResult(executed=[f"place_limit:{tp.id}"]))
    assert sched.last_mirror_results, sched.last_mirror_results
    assert not sched.last_mirror_results[0].get("errors"), sched.last_mirror_results
    sched.ig.update_position_protection.assert_called()
    kwargs = sched.ig.update_position_protection.call_args
    assert kwargs.args[0] == "DI_P"
    assert kwargs.kwargs.get("limit_level") == 8425.0
    book = sched._load_order_book(0)
    assert book.get(tp.id) == "attached:DI_P:tp"
    assert tp.deal_id == "attached:DI_P:tp"


def test_entry_fill_upgrades_bracket_tp_no_duplicate(tmp_path: Path):
    """Entry fill + adopt must leave exactly one attached:{pos}:tp (no empty ghost)."""
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    ledger = sched.ig.ledger
    ledger.last_price = 8350.0

    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=8350.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
        )
    )
    tp = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8325.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            parent_order_id=entry.id,
        )
    )
    hedge = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=8360.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            parent_order_id=entry.id,
        )
    )

    def _push(order, *, currency=None, limit_level=None, stop_level=None):
        if order.id == entry.id:
            order.deal_id = "WO_ENTRY_1"
            sched.ig.last_ig_result = {
                "deal_status": "ACCEPTED",
                "limit_level": limit_level,
            }
        elif order.id == hedge.id:
            order.deal_id = "WO_HEDGE_1"
            sched.ig.last_ig_result = {"deal_status": "ACCEPTED"}
        else:
            raise AssertionError(f"unexpected push for {order.id}")
        return order

    sched.ig.push_working_order = MagicMock(side_effect=_push)
    from chatbot.trader.risk_gate import GateResult

    sched._mirror_orders_to_ig(GateResult(executed=[f"place_limit:{entry.id}"]))
    assert tp.deal_id == "attached:WO_ENTRY_1:tp"

    # Entry filled on IG: WO gone; position keeps limitLevel; hedge still working.
    sched.ig.list_working_orders = MagicMock(
        return_value=[
            {
                "dealId": "WO_HEDGE_1",
                "epic": cfg.epic,
                "direction": "BUY",
                "orderType": "STOP",
                "orderLevel": 8360.0,
                "orderSize": 1.0,
            }
        ]
    )
    sched.ig.list_open_positions = MagicMock(
        return_value=[
            {
                "deal_id": "DI_POS_1",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8350.0,
                "limit_level": 8325.0,
            }
        ]
    )

    out = sched._sync_ledger_from_ig()
    assert out.get("opened") or out.get("repaired")
    assert entry.id not in ledger.working_orders

    tps = [
        o
        for o in ledger.working_orders.values()
        if o.purpose == OrderPurpose.TP
    ]
    assert len(tps) == 1, [(o.id, o.deal_id) for o in tps]
    assert tps[0].deal_id == "attached:DI_POS_1:tp"
    assert tps[0].position_id in ledger.positions
    assert not any(not (o.deal_id or "").strip() for o in tps)
    hedges = [
        o
        for o in ledger.working_orders.values()
        if o.purpose == OrderPurpose.HEDGE_COVER
    ]
    assert len(hedges) == 1
    assert hedges[0].deal_id == "WO_HEDGE_1"


def test_sync_purges_ghost_exposure_orders(tmp_path: Path):
    """replace_open rebuild drops empty-deal_id ghosts; keeps IG-bound WOs."""
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    ledger = sched.ig.ledger
    ghost_entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=8375.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
            deal_id="",
        )
    )
    ghost_hedge = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=8400.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            deal_id="",
        )
    )
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.SELL,
            level=8340.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            deal_id="DI_OK",
        )
    )
    sched.ig.list_open_positions = MagicMock(return_value=[])
    sched.ig.list_working_orders = MagicMock(
        return_value=[
            {
                "dealId": "DI_OK",
                "epic": cfg.epic,
                "direction": "SELL",
                "orderType": "STOP",
                "orderLevel": 8340.0,
                "orderSize": 1.0,
            }
        ]
    )
    out = sched._sync_ledger_from_ig()
    assert out["repaired"] is True
    assert ghost_entry.id not in ledger.working_orders
    assert ghost_hedge.id not in ledger.working_orders
    assert any((o.deal_id or "") == "DI_OK" for o in ledger.working_orders.values())
    snap = ledger.get_snapshot()
    assert all((o.deal_id or "").strip() for o in snap.working_orders)


def test_sync_rebuilds_attached_tp_from_position(tmp_path: Path):
    """Ghost empty TP is wiped; attached:{pos}:tp comes from IG limitLevel."""
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    ledger = sched.ig.ledger
    leg = ledger._open_leg(Side.SELL, 1.0, 8350.0, LegRole.PRIMARY, deal_id="DI_POS")
    ghost = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8325.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            position_id=leg.id,
            deal_id="",
        )
    )
    sched.ig.list_open_positions = MagicMock(
        return_value=[
            {
                "deal_id": "DI_POS",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8350.0,
                "limit_level": 8325.0,
            }
        ]
    )
    sched.ig.list_working_orders = MagicMock(return_value=[])
    out = sched._sync_ledger_from_ig()
    assert out["repaired"] is True
    assert ghost.id not in ledger.working_orders
    tps = [o for o in ledger.working_orders.values() if o.purpose == OrderPurpose.TP]
    assert len(tps) == 1
    assert tps[0].deal_id == "attached:DI_POS:tp"


def test_mirror_size_drift_cancels_and_replaces(tmp_path: Path):
    from chatbot.trader.risk_gate import GateResult

    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    sched.ig.snap_level = MagicMock(side_effect=lambda x, epic=None: float(x))
    hedge = sched.ig.ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=8410.0,
            size=2.0,
            purpose=OrderPurpose.HEDGE_COVER,
            deal_id="WO_OLD",
        )
    )
    sched._save_order_book(0, {hedge.id: "WO_OLD"})
    sched.ig.list_working_orders = MagicMock(
        return_value=[
            {
                "workingOrderData": {
                    "dealId": "WO_OLD",
                    "orderSize": 1.0,
                    "orderLevel": 8410.0,
                    "direction": "BUY",
                    "orderType": "STOP",
                }
            }
        ]
    )
    sched.ig.cancel_working_order = MagicMock(return_value={"dealId": "WO_OLD"})

    def _push(order, *, currency=None, limit_level=None, stop_level=None):
        order.deal_id = "WO_NEW"
        return order

    sched.ig.push_working_order = MagicMock(side_effect=_push)
    sched.ig.amend_working_order_by_deal_id = MagicMock(
        side_effect=AssertionError("size drift must not PUT-amend")
    )

    sched._mirror_orders_to_ig(GateResult())
    assert sched.ig.cancel_working_order.called
    assert sched.ig.cancel_working_order.call_args.args[0] == "WO_OLD"
    assert sched.ig.push_working_order.called
    book = sched._load_order_book(0)
    assert book.get(hedge.id) == "WO_NEW"
    cancelled = sched.last_mirror_results[0]["cancelled"]
    assert any(c.get("via") == "size_replace" for c in cancelled)


def test_mirror_level_drift_puts_and_preserves_limit(tmp_path: Path):
    from chatbot.trader.risk_gate import GateResult

    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    sched.ig.snap_level = MagicMock(side_effect=lambda x, epic=None: float(x))
    entry = sched.ig.ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=8410.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
            deal_id="WO_ENTRY",
        )
    )
    tp = sched.ig.ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8380.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            parent_order_id=entry.id,
            deal_id="attached:WO_ENTRY:tp",
        )
    )
    sched._save_order_book(
        0, {entry.id: "WO_ENTRY", tp.id: "attached:WO_ENTRY:tp"}
    )
    sched.ig.list_working_orders = MagicMock(
        return_value=[
            {
                "workingOrderData": {
                    "dealId": "WO_ENTRY",
                    "orderSize": 1.0,
                    "orderLevel": 8400.0,
                    "limitLevel": 8380.0,
                    "direction": "SELL",
                    "orderType": "LIMIT",
                }
            }
        ]
    )
    sched.ig.cancel_working_order = MagicMock(
        side_effect=AssertionError("level-only must not cancel")
    )
    sched.ig.push_working_order = MagicMock(
        side_effect=AssertionError("level-only must not re-place")
    )
    sched.ig.amend_working_order_by_deal_id = MagicMock(return_value=8410.0)

    sched._mirror_orders_to_ig(GateResult())
    assert sched.ig.amend_working_order_by_deal_id.called
    kwargs = sched.ig.amend_working_order_by_deal_id.call_args.kwargs
    assert kwargs["level"] == 8410.0
    assert kwargs["limit_level"] == 8380.0
    book = sched._load_order_book(0)
    assert book.get(entry.id) == "WO_ENTRY"
    assert book.get(tp.id) == "attached:WO_ENTRY:tp"
    amended = sched.last_mirror_results[0]["amended"]
    assert amended and amended[0]["order_id"] == entry.id


def test_mirror_skips_attached_sentinel_for_amend(tmp_path: Path):
    from chatbot.trader.risk_gate import GateResult

    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    leg = sched.ig.ledger._open_leg(
        Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_P"
    )
    tp = sched.ig.ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8425.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            position_id=leg.id,
            deal_id="attached:DI_P:tp",
        )
    )
    sched._save_order_book(0, {tp.id: "attached:DI_P:tp"})
    sched.ig.list_working_orders = MagicMock(return_value=[])
    sched.ig.amend_working_order_by_deal_id = MagicMock(
        side_effect=AssertionError("attached sentinel must not amend")
    )
    sched.ig.cancel_working_order = MagicMock(
        side_effect=AssertionError("attached sentinel must not cancel")
    )
    sched.ig.update_position_protection = MagicMock()

    sched._mirror_orders_to_ig(GateResult())
    assert not sched.ig.amend_working_order_by_deal_id.called
    assert not sched.ig.cancel_working_order.called
    book = sched._load_order_book(0)
    assert book.get(tp.id) == "attached:DI_P:tp"


def test_mirror_tp_on_existing_entry_amends_limit(tmp_path: Path):
    """TP added after entry is already on IG must PUT limit on the entry WO."""
    from chatbot.trader.risk_gate import GateResult

    cfg = _cfg(epic="CS.D.EURUSD.MINI.IP")
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    ledger = sched.ig.ledger
    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=1.138,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
            deal_id="WO_ENTRY_EXISTING",
        )
    )
    tp = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=1.1395,
            size=1.0,
            purpose=OrderPurpose.TP,
            parent_order_id=entry.id,
        )
    )
    sched._save_order_book(0, {entry.id: "WO_ENTRY_EXISTING"})
    sched.ig.list_working_orders = MagicMock(
        return_value=[
            {
                "dealId": "WO_ENTRY_EXISTING",
                "epic": cfg.epic,
                "direction": "BUY",
                "orderType": "LIMIT",
                "orderLevel": 1.138,
                "orderSize": 1.0,
            }
        ]
    )
    sched.ig.snap_level = MagicMock(side_effect=lambda x, epic=None: float(x))
    sched.ig.push_working_order = MagicMock(
        side_effect=AssertionError("must not re-place entry")
    )

    def _amend(deal_id, *, order_type, level, limit_level=None, stop_level=None):
        assert deal_id == "WO_ENTRY_EXISTING"
        assert limit_level == 1.1395
        sched.ig.last_ig_result = {
            "deal_status": "ACCEPTED",
            "limit_distance": 15.0,
            "tp_attached": True,
        }
        return float(level)

    sched.ig.amend_working_order_by_deal_id = MagicMock(side_effect=_amend)
    sched._mirror_orders_to_ig(GateResult(executed=[f"place_limit:{tp.id}"]))
    assert tp.deal_id == "attached:WO_ENTRY_EXISTING:tp"
    book = sched._load_order_book(0)
    assert book.get(tp.id) == "attached:WO_ENTRY_EXISTING:tp"
    vias = {p["order_id"]: p.get("via") for p in sched.last_mirror_results[0]["placed"]}
    assert vias.get(tp.id) == "entry_amend_limitDistance"
    assert not (sched.last_mirror_results[0].get("errors") or [])


def test_mirror_orphan_tp_binds_to_open_position(tmp_path: Path):
    """TP left with parent_order_id after entry fill attaches via position API."""
    from chatbot.trader.risk_gate import GateResult

    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    sched.ig.epic_compatible_with_account = MagicMock(return_value=True)
    ledger = sched.ig.ledger
    leg = ledger._open_leg(Side.SELL, 1.0, 8455.0, LegRole.PRIMARY, deal_id="DI_POS")
    tp = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8425.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            parent_order_id="o_gone_entry",
        )
    )
    sched.ig.list_working_orders = MagicMock(return_value=[])
    sched.ig.update_position_protection = MagicMock(return_value={})
    sched.ig.push_working_order = MagicMock(
        side_effect=AssertionError("must not forceOpen TP")
    )
    sched._mirror_orders_to_ig(GateResult(executed=[f"place_limit:{tp.id}"]))
    sched.ig.update_position_protection.assert_called_once()
    kwargs = sched.ig.update_position_protection.call_args
    assert kwargs.args[0] == "DI_POS"
    assert kwargs.kwargs.get("limit_level") == 8425.0
    assert tp.deal_id == "attached:DI_POS:tp"
    assert tp.position_id == leg.id
    assert tp.parent_order_id is None


def test_stream_skip_reloads_ledger_from_disk(tmp_path: Path):
    """Skipping REST sync must pick up stream-written attached TP children."""
    import json

    live_dir = tmp_path / "live"
    live_dir.mkdir()
    journal = live_dir / "journal"
    journal.mkdir()
    cfg = _cfg(epic="CS.D.EURUSD.MINI.IP")
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=journal, dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    # Stale in-memory book: entry only, no nested TP.
    entry = sched.ig.ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=1.138,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
            deal_id="WO_E",
        )
    )
    # Disk has stream reconcile with attached TP.
    (live_dir / "state.json").write_text(
        json.dumps(
            {
                "phase": "Flat",
                "positions": [],
                "working_orders": [
                    {
                        "id": entry.id,
                        "type": "LIMIT",
                        "side": "BUY",
                        "level": 1.138,
                        "size": 1.0,
                        "purpose": "entry",
                        "deal_id": "WO_E",
                    },
                    {
                        "id": "o_tp",
                        "type": "LIMIT",
                        "side": "SELL",
                        "level": 1.1395,
                        "size": 1.0,
                        "purpose": "tp",
                        "parent_order_id": entry.id,
                        "deal_id": "attached:WO_E:tp",
                    },
                ],
                "closed_trades": [],
            }
        ),
        encoding="utf-8",
    )
    sched.stream_status_path = live_dir / "stream_status.json"
    sched.stream_status_path.write_text(
        json.dumps(
            {
                "ok": True,
                "book_reconciled_at": "2099-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    sched._last_rest_book_sync_at = None
    # Force fresh: patch helper.
    from chatbot.application import trader_stream_service as tss

    orig = tss.stream_book_reconcile_is_fresh
    tss.stream_book_reconcile_is_fresh = lambda raw: True  # type: ignore[assignment]
    try:
        out = sched._sync_ledger_from_ig()
    finally:
        tss.stream_book_reconcile_is_fresh = orig  # type: ignore[assignment]
    assert out.get("skipped_stream_fresh") is True
    assert out.get("reloaded_from_disk") is True
    tps = [
        o
        for o in sched.ig.ledger.working_orders.values()
        if o.purpose == OrderPurpose.TP
    ]
    assert len(tps) == 1
    assert tps[0].deal_id == "attached:WO_E:tp"
