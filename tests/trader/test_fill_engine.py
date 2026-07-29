from chatbot.cac40.config import Cac40Config
from chatbot.cac40.fill_engine import (
    evaluate_limit_fill,
    evaluate_stop_fill,
    resolve_intrabar_conflict,
)
from chatbot.cac40.models import OrderPurpose, OrderType, Side, WorkingOrder


def _order(**kwargs):
    base = dict(
        id="o1",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=100.0,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    base.update(kwargs)
    return WorkingOrder(**base)


def test_limit_buy_touch():
    cfg = Cac40Config(spread_points=2.0)
    bar = {"open": 101, "high": 102, "low": 99, "close": 100.5}
    fill = evaluate_limit_fill(_order(side=Side.BUY, level=100), bar, cfg)
    assert fill is not None
    assert fill.fill_price == 101.0  # level + half spread


def test_limit_buy_gap_open():
    cfg = Cac40Config(spread_points=0.0)
    bar = {"open": 98, "high": 99, "low": 97, "close": 98.5}
    fill = evaluate_limit_fill(_order(side=Side.BUY, level=100), bar, cfg)
    assert fill is not None
    assert fill.fill_price == 98  # open gap


def test_stop_sell_gap_open():
    cfg = Cac40Config(slippage_points=0.5)
    bar = {"open": 95, "high": 96, "low": 94, "close": 95.5}
    order = _order(type=OrderType.STOP, side=Side.SELL, level=100)
    fill = evaluate_stop_fill(order, bar, cfg)
    assert fill is not None
    assert fill.fill_price == 94.5  # open - slippage


def test_intrabar_pessimistic_prefers_stop():
    cfg = Cac40Config()
    stop = _order(id="s1", type=OrderType.STOP, side=Side.SELL, level=99, purpose=OrderPurpose.HEDGE_COVER, position_id="p1")
    tp = _order(id="t1", type=OrderType.LIMIT, side=Side.SELL, level=105, purpose=OrderPurpose.TP, position_id="p1")
    bar = {"open": 100, "high": 106, "low": 98, "close": 104}
    fills = [
        (stop, evaluate_stop_fill(stop, bar, cfg)),
        (tp, evaluate_limit_fill(tp, bar, cfg)),
    ]
    fills = [(o, f) for o, f in fills if f]
    resolved = resolve_intrabar_conflict(fills, pessimistic=True)
    assert len(resolved) == 1
    assert resolved[0][0].type == OrderType.STOP
