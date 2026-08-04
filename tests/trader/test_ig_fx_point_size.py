"""FX dealingRules POINTS must convert to price (not CAC40 index points)."""

from unittest.mock import MagicMock

from chatbot.trader.config import TraderConfig
from chatbot.trader.ig_connector import IgApiError, IgConnector
from chatbot.trader.models import OrderPurpose, OrderType, Side, WorkingOrder


def _fx_market(*, scaling: float = 10000.0) -> dict:
    return {
        "snapshot": {
            "bid": 1.13698,
            "offer": 1.13704,
            "marketStatus": "TRADEABLE",
            "scalingFactor": scaling,
        },
        "dealingRules": {
            "minStepDistance": {"unit": "POINTS", "value": 5.0},
            "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 2.0},
            "maxStopOrLimitDistance": {"unit": "PERCENTAGE", "value": 75.0},
            "minDealSize": {"unit": "POINTS", "value": 0.1},
        },
        "instrument": {"name": "EUR/USD Mini", "expiry": "-"},
    }


def test_eurusd_point_size_and_step_from_scaling_factor() -> None:
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.13701
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]

    assert ig.resolve_point_size() == 0.0001
    # FX snap uses pipette (point/10), not minStepDistance=5 (that over-snaps vs UI).
    assert abs(ig.resolve_price_step() - 0.00001) < 1e-12
    assert abs(ig.resolve_min_stop_or_limit_distance() - 0.0002) < 1e-12
    # 75% of mid
    assert abs(ig.resolve_max_stop_or_limit_distance() - 1.13701 * 0.75) < 1e-9


def test_eurusd_snap_preserves_five_decimal_ui_levels() -> None:
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.13701
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]

    assert abs(ig.snap_level(1.13721) - 1.13721) < 1e-12
    assert abs(ig.snap_level(1.13730) - 1.13730) < 1e-12
    # Must not snap 1.1373 up to 1.1375 (old minStepDistance=5 bug).
    assert abs(ig.snap_level(1.1373) - 1.1373) < 1e-12


def test_eurusd_clearance_does_not_use_index_floors() -> None:
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.13701
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]

    min_c, max_c = ig.working_order_clearance_points(OrderType.LIMIT)
    assert min_c < 0.01  # not 12 index points
    assert max_c < 1.0
    assert max_c > min_c


def test_eurusd_snap_keeps_fx_levels() -> None:
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.13701
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]

    # Must not snap ~1.13 down to 0 / multiples of 5.
    assert abs(ig.snap_level(1.1315) - 1.1315) < 1e-9
    offset = 0.01
    buy = ig.snap_level(1.13701 - offset)
    assert buy > 1.0


def test_eurusd_sell_entry_keeps_tp_below_bid() -> None:
    """Mean-reversion SELL above mid with TP a few pips below bid must attach."""
    ig = IgConnector(
        TraderConfig(epic="CS.D.EURUSD.MINI.IP", spread_points=0.00006), dry_run=True
    )
    ig._cst = "cst"
    ig.ledger.last_price = 1.13672
    ig.last_dealable_bid = 1.13670
    ig.last_dealable_offer = 1.13676
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]
    order = WorkingOrder(
        id="o1",
        type=OrderType.LIMIT,
        side=Side.SELL,
        level=1.1385,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    ig.ledger.place_order(order)
    _level, tp, notes = ig.apply_working_order_clearance(order, limit_level=1.136)
    assert tp is not None, notes
    assert abs(float(tp) - 1.136) < 1e-6
    assert not any("omit_tp_attach" in n for n in notes)


def test_eurusd_mean_reversion_entry_not_muted_to_maxstop_distance() -> None:
    """Regression: maxStop value 75 must not mute BUY LIMIT 1.1405→~1.1376."""
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.14518
    ig.last_dealable_bid = 1.14512
    ig.last_dealable_offer = 1.14518
    # Corrupt minNormal mimicking a maxStop "75" leak as POINTS.
    bad = _fx_market()
    bad["dealingRules"] = {
        **bad["dealingRules"],
        "minNormalStopOrLimitDistance": {"unit": "POINTS", "value": 75.0},
    }
    ig.get_market = MagicMock(return_value=bad)  # type: ignore[method-assign]

    min_c, _max_c = ig.working_order_clearance_points(OrderType.LIMIT)
    assert min_c <= 0.003 + 1e-12  # ≤30pts @ 0.0001

    order = WorkingOrder(
        id="o407",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=1.1405,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    ig.ledger.place_order(order)
    level, tp, notes = ig.apply_working_order_clearance(order, limit_level=1.1465)
    assert abs(level - 1.1405) < 1e-9, (level, notes)
    assert tp is not None
    assert not any("too close" in n for n in notes)


def test_eurusd_push_refuses_large_silent_rewrite() -> None:
    """Fail closed if clearance would still move FX levels by >5 points."""
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.14518
    ig.last_dealable_bid = 1.14512
    ig.last_dealable_offer = 1.14518
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]

    order = WorkingOrder(
        id="o1",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=1.1405,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    ig.ledger.place_order(order)

    # Force a huge widen after normal clearance math.
    def _huge_clearance(order, limit_level=None):  # noqa: ANN001
        order.level = 1.13763
        if order.id in ig.ledger.working_orders:
            ig.ledger.working_orders[order.id].level = 1.13763
        return 1.13763, limit_level, ["BUY LIMIT 1.1405->1.13763 (too close)"]

    ig.apply_working_order_clearance = _huge_clearance  # type: ignore[method-assign]
    try:
        ig.push_working_order(order, limit_level=1.1465)
        raise AssertionError("expected IgApiError")
    except IgApiError as exc:
        assert "refused silent rewrite" in str(exc)
    assert abs(float(order.level) - 1.1405) < 1e-9


def test_eurusd_clearance_uses_dealable_when_last_price_unset() -> None:
    """Index floors must not apply when last_price=0 but dealable is FX."""
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 0.0
    ig.last_dealable_bid = 1.14512
    ig.last_dealable_offer = 1.14518
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]

    min_c, max_c = ig.working_order_clearance_points(OrderType.LIMIT)
    assert min_c < 0.01
    assert max_c < 1.0
    order = WorkingOrder(
        id="o1",
        type=OrderType.LIMIT,
        side=Side.BUY,
        level=1.1405,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    level, _tp, notes = ig.apply_working_order_clearance(order)
    assert abs(level - 1.1405) < 1e-9, (level, notes)


def test_points_to_price_fx_vs_index() -> None:
    from chatbot.trader.point_size import points_to_price

    assert abs(points_to_price(1.5, 1.15) - 0.00015) < 1e-12
    assert abs(points_to_price(15.0, 1.15) - 0.0015) < 1e-12
    assert abs(points_to_price(1.5, 8000.0) - 1.5) < 1e-12
