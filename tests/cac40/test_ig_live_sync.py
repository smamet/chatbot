"""Live IG book sync: no OHLC fills, dealId bind, replace repair, phantom quarantine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from chatbot.application.cac40_live_service import adopt_ig_snapshot_into_ledger
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import (
    LegRole,
    OrderPurpose,
    OrderType,
    Side,
    WorkingOrder,
)
from chatbot.cac40.scheduler import LiveScheduler


def _cfg(**kwargs) -> Cac40Config:
    base = dict(
        spread_points=0,
        point_value=1.0,
        order_size=1.0,
        max_open_positions=4,
        epic="IX.D.CAC.IFD.IP",
    )
    base.update(kwargs)
    return Cac40Config(**base)


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


def test_wo_vanish_does_not_close_without_positions(tmp_path: Path):
    cfg = _cfg()
    sched = LiveScheduler(
        cfg, api_key="x", journal_dir=tmp_path / "j", dry_run=False, sleep_seconds=1
    )
    sched.ig._cst = "cst"
    ledger = sched.ig.ledger
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
    sched.ig.list_working_orders = MagicMock(return_value=[])  # TP vanished
    out = sched._sync_working_orders_from_ig()
    assert out["changed"] is True
    assert out["dropped"]
    assert leg.id in ledger.positions  # still open — positions reconcile decides


def test_reconcile_closes_when_ig_position_gone(tmp_path: Path):
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
    out = sched._reconcile_positions_from_ig(force=True)
    assert out["ran"] is True
    assert out["closed"]
    assert ledger.legs_count() == 0
    assert ledger.realized_session == 35.0  # 8455-8420


def test_reconcile_repairs_local_flat_ig_short(tmp_path: Path):
    from chatbot.cac40.models import ClosedTrade

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
    out = sched._reconcile_positions_from_ig(force=True)
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
    from chatbot.cac40.models import ClosedTrade

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
    from chatbot.cac40.ig_connector import IgConnector

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
    body = conn._client.post.call_args.kwargs.get("json") or {}
    assert body.get("dealId") == "DI_CLOSE"
    assert body.get("direction") == "BUY"
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
        elif order.id == hedge.id:
            assert order.purpose == OrderPurpose.HEDGE_COVER
            assert limit_level is None
            order.deal_id = "WO_HEDGE_1"
        else:
            raise AssertionError(f"unexpected push for {order.id}")
        return order

    sched.ig.push_working_order = MagicMock(side_effect=_push)
    from chatbot.cac40.risk_gate import GateResult

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
    from chatbot.cac40.risk_gate import GateResult

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
    from chatbot.cac40.risk_gate import GateResult

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
        elif order.id == hedge.id:
            order.deal_id = "WO_HEDGE_1"
        else:
            raise AssertionError(f"unexpected push for {order.id}")
        return order

    sched.ig.push_working_order = MagicMock(side_effect=_push)
    from chatbot.cac40.risk_gate import GateResult

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

    wo_out = sched._sync_working_orders_from_ig()
    assert any(d.get("deal_id") == "WO_ENTRY_1" for d in wo_out.get("dropped") or [])
    assert entry.id not in ledger.working_orders

    rec = sched._reconcile_positions_from_ig(force=True)
    assert rec.get("opened") or rec.get("repaired")

    tps = [
        o
        for o in ledger.working_orders.values()
        if o.purpose == OrderPurpose.TP
    ]
    assert len(tps) == 1, [(o.id, o.deal_id) for o in tps]
    assert tps[0].id == tp.id
    assert tps[0].deal_id == "attached:DI_POS_1:tp"
    assert tps[0].position_id in ledger.positions
    assert not any(not (o.deal_id or "").strip() for o in tps)


def test_drop_ghost_empty_tp_when_canonical_attached_exists(tmp_path: Path):
    """WO sync drops empty-deal_id TP when attached:{pos}:tp already present."""
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
    canonical = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8325.0,
            size=1.0,
            purpose=OrderPurpose.TP,
            position_id=leg.id,
            deal_id="attached:DI_POS:tp",
        )
    )
    sched.ig.list_working_orders = MagicMock(return_value=[])
    out = sched._sync_working_orders_from_ig()
    assert ghost.id not in ledger.working_orders
    assert canonical.id in ledger.working_orders
    assert any(d.get("reason") == "ghost_attached_tp" for d in out.get("dropped") or [])
