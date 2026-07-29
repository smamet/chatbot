"""FX dealingRules POINTS must convert to price (not CAC40 index points)."""

from unittest.mock import MagicMock

from chatbot.trader.config import TraderConfig
from chatbot.trader.ig_connector import IgConnector
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


def test_eurusd_attach_uses_limit_distance_like_ig_ui() -> None:
    """IG web UI sends Limit Distance in POINTS; bot must match."""
    ig = IgConnector(TraderConfig(epic="CS.D.EURUSD.MINI.IP"), dry_run=True)
    ig._cst = "cst"
    ig.ledger.last_price = 1.1365
    ig.get_market = MagicMock(return_value=_fx_market())  # type: ignore[method-assign]
    body: dict = {}
    ig._attach_working_order_tp(body, level=1.1385, limit_level=1.13648)
    assert "limitDistance" in body
    assert "limitLevel" not in body
    # 1.1385 - 1.13648 ≈ 20 points @ 0.0001 (IG UI showed ~19.7)
    assert 19.0 <= body["limitDistance"] <= 21.0
