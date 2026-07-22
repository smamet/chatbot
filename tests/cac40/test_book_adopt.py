from chatbot.application.cac40_live_service import adopt_ig_snapshot_into_ledger
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import OrderPurpose, OrderType, Side


def test_adopt_imports_position_and_orders_idempotent():
    cfg = Cac40Config(spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8450

    positions = [
        {
            "deal_id": "DIAAAA1",
            "epic": "IX.D.CAC.IFD.IP",
            "side": Side.SELL,
            "size": 1.0,
            "level": 8455.0,
        }
    ]
    working_orders = [
        {
            "dealId": "WO_ENTRY",
            "epic": "IX.D.CAC.IFD.IP",
            "direction": "SELL",
            "orderType": "LIMIT",
            "orderLevel": 8460.0,
            "orderSize": 1.0,
        },
        {
            "dealId": "WO_TP",
            "epic": "IX.D.CAC.IFD.IP",
            "direction": "BUY",
            "orderType": "LIMIT",
            "orderLevel": 8425.0,
            "orderSize": 1.0,
        },
        {
            "dealId": "WO_HEDGE",
            "epic": "IX.D.CAC.IFD.IP",
            "direction": "BUY",
            "orderType": "STOP",
            "orderLevel": 8470.0,
            "orderSize": 1.0,
        },
    ]

    first = adopt_ig_snapshot_into_ledger(
        ledger,
        positions=positions,
        working_orders=working_orders,
        epic="IX.D.CAC.IFD.IP",
    )
    assert len(first["imported_positions"]) == 1
    assert len(first["imported_orders"]) == 3
    assert ledger.legs_count() == 1
    leg = next(iter(ledger.positions.values()))
    assert leg.deal_id == "DIAAAA1"
    assert leg.side == Side.SELL

    by_deal = {o.deal_id: o for o in ledger.working_orders.values()}
    assert by_deal["WO_TP"].purpose == OrderPurpose.TP
    assert by_deal["WO_TP"].position_id == leg.id
    assert by_deal["WO_HEDGE"].purpose == OrderPurpose.HEDGE_COVER
    assert by_deal["WO_ENTRY"].purpose == OrderPurpose.ENTRY
    assert by_deal["WO_ENTRY"].type == OrderType.LIMIT

    second = adopt_ig_snapshot_into_ledger(
        ledger,
        positions=positions,
        working_orders=working_orders,
        epic="IX.D.CAC.IFD.IP",
    )
    assert second["imported_positions"] == []
    assert second["imported_orders"] == []
    assert ledger.legs_count() == 1
    assert len(ledger.working_orders) == 3
