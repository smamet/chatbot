"""Working-order clearance vs dealable quote (IG ATTACHED_ORDER_LEVEL_ERROR)."""

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.ig_connector import IgConnector
from chatbot.cac40.models import OrderPurpose, OrderType, Side, WorkingOrder


def _conn(*, last_price: float = 8412.0) -> IgConnector:
    cfg = Cac40Config(epic="IX.D.CAC.BMU.IP", spread_points=1.5)
    conn = IgConnector(cfg, dry_run=True)
    conn.ledger.last_price = last_price
    return conn


def test_buy_stop_hedge_widened_to_clearance_above_offer() -> None:
    """BUY STOP @ 8440 with mid~8412 must widen (~80 pts) before IG place."""
    conn = _conn(last_price=8412.0)
    order = WorkingOrder(
        id="o289",
        type=OrderType.STOP,
        side=Side.BUY,
        level=8440.0,
        size=2.0,
        purpose=OrderPurpose.HEDGE_COVER,
    )
    conn.ledger.place_order(order)
    level, tp, notes = conn.apply_working_order_clearance(order)
    assert tp is None
    assert notes
    assert level >= 8412.0 + 80.0 - 1.0  # half-spread on offer proxy
    assert order.level == level
    assert conn.ledger.working_orders["o289"].level == level
    conn.close()


def test_sell_limit_entry_widened_above_offer() -> None:
    conn = _conn(last_price=8412.0)
    order = WorkingOrder(
        id="e1",
        type=OrderType.LIMIT,
        side=Side.SELL,
        level=8420.0,  # too close / through
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    level, tp, notes = conn.apply_working_order_clearance(order, limit_level=8390.0)
    assert notes
    assert level > 8420.0
    assert tp is not None and tp < level
    conn.close()


def test_buy_limit_with_attached_tp_widened_from_tight_bracket() -> None:
    """Regression: entry 8380 + TP 8390 with last 8388.1."""
    conn = _conn(last_price=8388.1)
    order = WorkingOrder(
        id="o271",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=8380.0,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    level, tp, notes = conn.apply_working_order_clearance(order, limit_level=8390.0)
    assert notes
    assert level <= 8388.1 - 25.0 + 1.0
    assert tp is not None and tp >= level + 25.0 - 1.0
    conn.close()


def test_clearance_uses_cached_bid_offer_not_mid_alone() -> None:
    conn = _conn(last_price=8412.0)
    conn.last_dealable_bid = 8410.0
    conn.last_dealable_offer = 8414.0
    order = WorkingOrder(
        id="h1",
        type=OrderType.STOP,
        side=Side.BUY,
        level=8440.0,
        size=1.0,
        purpose=OrderPurpose.HEDGE_COVER,
    )
    level, _, notes = conn.apply_working_order_clearance(order)
    assert notes
    assert level >= 8414.0 + 80.0 - 0.1
    conn.close()


def test_resolve_price_step_ignores_min_stop_distance() -> None:
    conn = _conn()
    conn._market_cache[conn.config.epic] = {
        "dealingRules": {
            "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 50.0},
        }
    }
    assert conn.resolve_price_step() == 0.1
    conn._market_cache[conn.config.epic] = {
        "dealingRules": {
            "minStepDistance": {"unit": "POINTS", "value": 1.0},
            "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 50.0},
        }
    }
    assert conn.resolve_price_step() == 1.0
    conn.close()


def test_rejected_confirm_clears_phantom_deal_id() -> None:
    cfg = Cac40Config(epic="IX.D.CAC.BMU.IP")
    conn = IgConnector(cfg, dry_run=False)
    conn._cst = "cst"
    conn._security = "sec"
    conn.ledger.last_price = 8412.0
    order = WorkingOrder(
        id="o289",
        type=OrderType.STOP,
        side=Side.BUY,
        level=8495.0,
        size=1.0,
        purpose=OrderPurpose.HEDGE_COVER,
    )
    conn.ledger.place_order(order)
    conn.resolve_order_currency = lambda **k: "EUR"  # type: ignore[method-assign]
    conn.resolve_order_expiry = lambda **k: "-"  # type: ignore[method-assign]
    conn.snap_level = lambda level, epic=None: float(level)  # type: ignore[method-assign]
    conn.apply_working_order_clearance = lambda order, limit_level=None: (  # type: ignore[method-assign]
        float(order.level),
        limit_level,
        [],
    )

    class _Resp:
        is_error = False
        content = b'{"dealReference":"REF1"}'
        request = type("R", (), {"url": "http://x"})()

        def json(self):
            return {"dealReference": "REF1"}

    conn._client.post = lambda *a, **k: _Resp()  # type: ignore[method-assign]
    conn.confirm_deal = lambda ref, **k: {  # type: ignore[method-assign]
        "dealStatus": "REJECTED",
        "reason": "ATTACHED_ORDER_LEVEL_ERROR",
        "dealId": "DI_PHANTOM",
    }
    try:
        conn.push_working_order(order)
        assert False, "expected IgApiError"
    except Exception as exc:
        assert "ATTACHED_ORDER_LEVEL_ERROR" in str(exc)
    assert order.deal_id == ""
    assert conn.ledger.working_orders["o289"].deal_id == ""
    conn.close()
