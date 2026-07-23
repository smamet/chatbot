from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import OrderPurpose, OrderType, Side, WorkingOrder


def test_hedge_mode_keeps_long_and_short():
    cfg = Cac40Config(spread_points=0.0, slippage_points=0.0, max_open_positions=4)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=100,
            size=1,
            purpose=OrderPurpose.ENTRY,
        )
    )
    # First bar fills buy (active_from_bar=1)
    ledger.process_bar({"open": 100, "high": 101, "low": 99, "close": 100.5})
    assert ledger.legs_count() == 1
    assert next(iter(ledger.positions.values())).side == Side.BUY

    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.SELL,
            level=105,
            size=1,
            purpose=OrderPurpose.HEDGE_COVER,
        )
    )
    # Orders placed mid-run activate next bar only
    ledger.process_bar({"open": 104, "high": 104.5, "low": 103, "close": 104})
    assert ledger.legs_count() == 1
    ledger.process_bar({"open": 104, "high": 106, "low": 103, "close": 105})
    assert ledger.legs_count() == 2
    sides = {p.side for p in ledger.positions.values()}
    assert sides == {Side.BUY, Side.SELL}
    assert ledger.position_summary() == "hedged"


def test_close_by_position_id_only():
    cfg = Cac40Config(spread_points=0.0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    long_id = ledger.market_open(Side.BUY, 1)
    short_id = ledger.market_open(Side.SELL, 1)
    assert ledger.legs_count() == 2
    ledger.close_position(long_id, 110)
    assert long_id not in ledger.positions
    assert short_id in ledger.positions
    assert ledger.closed_trades[-1].realized_pnl == 10


def test_tp_without_position_id_never_opens_leg():
    cfg = Cac40Config(spread_points=0.0, max_open_positions=4)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=100,
            size=1,
            purpose=OrderPurpose.TP,  # missing position_id
        )
    )
    events = ledger.process_bar({"open": 100, "high": 101, "low": 99, "close": 100})
    assert ledger.legs_count() == 0
    assert events[0]["type"] == "rejected_fill"
    assert events[0]["reason"] == "tp_missing_position_id"


def test_hedge_fill_without_primary_rejected():
    cfg = Cac40Config(spread_points=0.0, slippage_points=0.0, max_open_positions=4)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=100,
            size=1,
            purpose=OrderPurpose.HEDGE_COVER,
        )
    )
    events = ledger.process_bar({"open": 99, "high": 101, "low": 98, "close": 100})
    assert ledger.legs_count() == 0
    assert events[0]["type"] == "rejected_fill"
    assert events[0]["reason"] == "hedge_without_primary"


def test_fill_refused_at_max_open_positions():
    cfg = Cac40Config(spread_points=0.0, max_open_positions=1, allow_market_orders=True)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.market_open(Side.BUY, 1)
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=100,
            size=1,
            purpose=OrderPurpose.ENTRY,
        )
    )
    events = ledger.process_bar({"open": 100, "high": 101, "low": 99, "close": 100})
    assert ledger.legs_count() == 1
    assert events[0]["type"] == "rejected_fill"
    assert events[0]["reason"] == "max_positions"


def test_dormant_bracket_children_arm_on_entry_fill():
    cfg = Cac40Config(spread_points=0.0, slippage_points=0.0, max_open_positions=4)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=101,
            size=1,
            purpose=OrderPurpose.ENTRY,
        )
    )
    tp = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=98,
            size=1,
            purpose=OrderPurpose.TP,
            parent_order_id=entry.id,
        )
    )
    hedge = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=103,
            size=1,
            purpose=OrderPurpose.HEDGE_COVER,
            parent_order_id=entry.id,
        )
    )
    # Bar touches TP but NOT entry — children stay dormant.
    events = ledger.process_bar({"open": 100, "high": 100.5, "low": 97, "close": 100})
    assert ledger.legs_count() == 0
    assert tp.id in ledger.working_orders
    assert hedge.id in ledger.working_orders
    assert not events

    # Entry fills; children arm. Same bar also hits hedge (BUY hedge_cover on fall).
    events = ledger.process_bar({"open": 100.5, "high": 102, "low": 100, "close": 101.5})
    assert any(e.get("type") == "open" for e in events)
    # Hedge fills same bar after arming → 2 legs.
    assert ledger.legs_count() == 2
    assert tp.id in ledger.working_orders
    assert ledger.working_orders[tp.id].position_id is not None
    assert hedge.id not in ledger.working_orders


def test_cancel_entry_cascades_bracket_children():
    cfg = Cac40Config(spread_points=0.0)
    ledger = HedgeLedger(config=cfg)
    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=101,
            size=1,
            purpose=OrderPurpose.ENTRY,
        )
    )
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=98,
            size=1,
            purpose=OrderPurpose.TP,
            parent_order_id=entry.id,
        )
    )
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=103,
            size=1,
            purpose=OrderPurpose.HEDGE_COVER,
            parent_order_id=entry.id,
        )
    )
    assert len(ledger.working_orders) == 3
    ledger.cancel_order(entry.id)
    assert not ledger.working_orders


def test_working_order_parent_roundtrip():
    order = WorkingOrder(
        id="o9",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=100,
        size=1,
        purpose=OrderPurpose.TP,
        parent_order_id="o1",
    )
    restored = WorkingOrder.from_dict(order.to_dict())
    assert restored.parent_order_id == "o1"
