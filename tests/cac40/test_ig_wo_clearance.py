"""Working-order clearance vs dealable quote (IG ATTACHED_ORDER_LEVEL_ERROR)."""

from chatbot.cac40.config import Cac40Config
from chatbot.cac40.ig_connector import IgConnector
from chatbot.cac40.models import OrderPurpose, OrderType, Side, WorkingOrder


def _conn(*, last_price: float = 8412.0) -> IgConnector:
    cfg = Cac40Config(epic="IX.D.CAC.BMU.IP", spread_points=1.5)
    conn = IgConnector(cfg, dry_run=True)
    conn.ledger.last_price = last_price
    return conn


def test_buy_stop_too_close_widened_to_min() -> None:
    conn = _conn(last_price=8412.0)
    conn.last_dealable_bid = 8410.0
    conn.last_dealable_offer = 8414.0
    order = WorkingOrder(
        id="o289",
        type=OrderType.STOP,
        side=Side.BUY,
        level=8420.0,  # only 6 pts above offer
        size=2.0,
        purpose=OrderPurpose.HEDGE_COVER,
    )
    conn.ledger.place_order(order)
    level, tp, notes = conn.apply_working_order_clearance(order)
    assert tp is None
    assert notes
    assert level >= 8414.0 + 12.0 - 0.1
    conn.close()


def test_buy_limit_too_far_clamped_inward() -> None:
    """Bare BUY LIMIT ~84pts away must clamp toward market when max is tight."""
    conn = _conn(last_price=8446.0)
    conn.last_dealable_bid = 8445.0
    conn.last_dealable_offer = 8447.0
    conn._market_cache[conn.config.epic] = {
        "dealingRules": {
            "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 8.0},
            "maxStopOrLimitDistance": {"unit": "POINTS", "value": 40.0},
        }
    }
    order = WorkingOrder(
        id="e1",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=8362.0,  # ~83 pts below — beyond max 40
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    level, tp, notes = conn.apply_working_order_clearance(order)
    assert tp is None
    assert any("beyond max" in n for n in notes)
    assert level >= 8445.0 - 40.0 - 0.1
    assert level <= 8445.0 - 8.0 + 0.1
    conn.close()


def test_buy_limit_omits_tp_when_through_market() -> None:
    conn = _conn(last_price=8388.1)
    order = WorkingOrder(
        id="o271",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=8300.0,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    level, tp, notes = conn.apply_working_order_clearance(order, limit_level=8390.0)
    assert any("omit_tp_attach" in n for n in notes)
    assert tp is None
    conn.close()


def test_buy_limit_keeps_tp_when_above_offer() -> None:
    conn = _conn(last_price=8412.0)
    conn.last_dealable_bid = 8410.0
    conn.last_dealable_offer = 8414.0
    order = WorkingOrder(
        id="e1",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=8300.0,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    level, tp, notes = conn.apply_working_order_clearance(order, limit_level=8500.0)
    assert tp is not None and tp >= 8500.0 - 0.1
    assert not any("omit_tp_attach" in n for n in notes)
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
