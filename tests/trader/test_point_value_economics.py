from __future__ import annotations

from chatbot.trader.config import TraderConfig
from chatbot.trader.hedge_ledger import HedgeLedger, price_move_pnl
from chatbot.trader.models import ClosedTrade, LegRole, Side
from chatbot.trader.profiles import get_profile, point_value_for_symbol


def test_eurusd_profile_point_value_matches_ig_mini() -> None:
    """IG EURUSD Mini: $1/pip → 3 pips on size 1 = $3."""
    assert get_profile("eurusd").default_point_value == 10000.0
    assert point_value_for_symbol("EURUSD") == 10000.0
    pnl = price_move_pnl(Side.BUY, 1.0, 1.1535, 1.1538, 10000.0)
    assert abs(pnl - 3.0) < 1e-9


def test_reconcile_rescales_legacy_unit_priced_closes() -> None:
    cfg = TraderConfig(symbol="EURUSD", point_value=10000.0)
    state = {
        "symbol": "EURUSD",
        "cash": 0.0015,
        "realized_session": 0.0015,
        "closed_trades": [
            {
                "id": "p1",
                "side": "SELL",
                "size": 1.0,
                "entry": 1.14,
                "exit": 1.1385,
                "role": "primary",
                "realized_pnl": 0.0015,
                "opened_at": "",
                "closed_at": "",
                "bars_held": 1,
                "deal_id": "d1",
                "ig_confirmed": True,
            }
        ],
    }
    ledger = HedgeLedger.from_state_dict(cfg, state)
    assert abs(ledger.closed_trades[0].realized_pnl - 15.0) < 1e-9
    assert abs(ledger.realized_session - 15.0) < 1e-9
    assert abs(ledger.cash - 15.0) < 1e-9


def test_reconcile_is_idempotent() -> None:
    cfg = TraderConfig(symbol="EURUSD", point_value=10000.0)
    ledger = HedgeLedger(config=cfg, symbol="EURUSD")
    ledger.closed_trades.append(
        ClosedTrade(
            id="p1",
            side=Side.BUY,
            size=1.0,
            entry=1.138,
            exit=1.1395,
            role=LegRole.PRIMARY,
            realized_pnl=15.0,
            opened_at="",
            closed_at="",
            bars_held=1,
        )
    )
    ledger.realized_session = 15.0
    ledger.cash = 15.0
    assert ledger.reconcile_closed_trades_to_point_value() == 0.0
    assert ledger.closed_trades[0].realized_pnl == 15.0


def test_close_position_uses_point_value() -> None:
    cfg = TraderConfig(spread_points=0.0, point_value=10000.0)
    ledger = HedgeLedger(config=cfg, symbol="EURUSD")
    ledger.last_price = 1.1535
    pid = ledger.market_open(Side.BUY, 1.0)
    trade = ledger.close_position(pid, 1.1538)
    assert trade is not None
    assert abs(trade.realized_pnl - 3.0) < 1e-9
