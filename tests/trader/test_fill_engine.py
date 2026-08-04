from chatbot.trader.config import TraderConfig
from chatbot.trader.fill_engine import (
    evaluate_limit_fill,
    evaluate_stop_fill,
    resolve_intrabar_conflict,
)
from chatbot.trader.models import OrderPurpose, OrderType, Side, WorkingOrder


def _order(**kwargs):
    # Index-scale prices → infer_point_size == 1.0 (points == price).
    base = dict(
        id="o1",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=8000.0,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    base.update(kwargs)
    return WorkingOrder(**base)


def test_limit_buy_touch():
    cfg = TraderConfig(spread_points=2.0)
    bar = {"open": 8001, "high": 8002, "low": 7999, "close": 8000.5}
    fill = evaluate_limit_fill(_order(side=Side.BUY, level=8000), bar, cfg)
    assert fill is not None
    assert fill.fill_price == 8001.0  # level + half spread


def test_limit_buy_gap_open():
    cfg = TraderConfig(spread_points=0.0)
    bar = {"open": 7998, "high": 7999, "low": 7997, "close": 7998.5}
    fill = evaluate_limit_fill(_order(side=Side.BUY, level=8000), bar, cfg)
    assert fill is not None
    assert fill.fill_price == 7998  # open gap


def test_stop_sell_gap_open():
    cfg = TraderConfig(slippage_points=0.5)
    bar = {"open": 7995, "high": 7996, "low": 7994, "close": 7995.5}
    order = _order(type=OrderType.STOP, side=Side.SELL, level=8000)
    fill = evaluate_stop_fill(order, bar, cfg)
    assert fill is not None
    assert fill.fill_price == 7994.5  # open - slippage


def test_intrabar_pessimistic_prefers_stop():
    cfg = TraderConfig()
    stop = _order(
        id="s1",
        type=OrderType.STOP,
        side=Side.SELL,
        level=7999,
        purpose=OrderPurpose.HEDGE_COVER,
        position_id="p1",
    )
    tp = _order(
        id="t1",
        type=OrderType.LIMIT,
        side=Side.SELL,
        level=8005,
        purpose=OrderPurpose.TP,
        position_id="p1",
    )
    bar = {"open": 8000, "high": 8006, "low": 7998, "close": 8004}
    fills = [
        (stop, evaluate_stop_fill(stop, bar, cfg)),
        (tp, evaluate_limit_fill(tp, bar, cfg)),
    ]
    fills = [(o, f) for o, f in fills if f]
    resolved = resolve_intrabar_conflict(fills, pessimistic=True)
    assert len(resolved) == 1
    assert resolved[0][0].type == OrderType.STOP
