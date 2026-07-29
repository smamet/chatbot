from chatbot.application.trader_live_service import adopt_ig_snapshot_into_ledger
from chatbot.trader.config import TraderConfig
from chatbot.trader.hedge_ledger import HedgeLedger
from chatbot.trader.models import OrderPurpose, OrderType, Side, WorkingOrder


def test_adopt_imports_position_and_orders_idempotent():
    cfg = TraderConfig(spread_points=0)
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
    # Opposite LIMIT stays ENTRY (not fake TP). Opposite STOP may nest as hedge.
    assert by_deal["WO_TP"].purpose == OrderPurpose.ENTRY
    assert by_deal["WO_TP"].position_id is None
    assert by_deal["WO_HEDGE"].purpose == OrderPurpose.HEDGE_COVER
    assert by_deal["WO_HEDGE"].position_id == leg.id
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


def test_adopt_nests_attached_tp_sl_under_entry_wo():
    from chatbot.application.trader_live_service import group_open_book

    cfg = TraderConfig(spread_points=0, epic="CS.D.EURUSD.MINI.IP")
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 1.1395

    result = adopt_ig_snapshot_into_ledger(
        ledger,
        positions=[],
        working_orders=[
            {
                "dealId": "WO_SELL_LIMIT",
                "epic": cfg.epic,
                "direction": "SELL",
                "orderType": "LIMIT",
                "orderLevel": 1.14000,
                "orderSize": 1.0,
                # IG chart: Limit @ 1.13850 = 15 pts TP on the entry
                "limitDistance": 15.0,
                "stopDistance": 10.0,
            },
            {
                "dealId": "WO_BUY_STOP",
                "epic": cfg.epic,
                "direction": "BUY",
                "orderType": "STOP",
                "orderLevel": 1.14100,
                "orderSize": 1.0,
                "limitLevel": 1.14250,
            },
        ],
        epic=cfg.epic,
        mode="replace_open",
    )
    by_deal = {o.deal_id: o for o in ledger.working_orders.values()}
    entry = by_deal["WO_SELL_LIMIT"]
    assert entry.purpose == OrderPurpose.ENTRY
    tp = by_deal["attached:WO_SELL_LIMIT:tp"]
    sl = by_deal["attached:WO_SELL_LIMIT:hedge_cover"]
    assert tp.purpose == OrderPurpose.TP
    assert tp.parent_order_id == entry.id
    assert abs(tp.level - 1.13850) < 1e-9
    assert sl.purpose == OrderPurpose.HEDGE_COVER
    assert sl.parent_order_id == entry.id
    assert abs(sl.level - 1.14100) < 1e-9

    other = by_deal["WO_BUY_STOP"]
    other_tp = by_deal["attached:WO_BUY_STOP:tp"]
    assert other_tp.parent_order_id == other.id
    assert other_tp.level == 1.14250
    assert "attached:WO_BUY_STOP:hedge_cover" not in by_deal

    # Separate opposite LIMITs must not appear; only attached children nest.
    assert len(result["imported_orders"]) == 5  # 2 entries + 2 TP/SL + 1 TP

    state_orders = [o.to_dict() for o in ledger.working_orders.values()]
    groups = group_open_book([], state_orders)
    entry_groups = [g for g in groups if g["kind"] == "entry"]
    sell_g = next(g for g in entry_groups if g["parent"]["deal_id"] == "WO_SELL_LIMIT")
    assert len(sell_g["children"]) == 2
    child_purposes = {c["purpose"] for c in sell_g["children"]}
    assert child_purposes == {"tp", "hedge_cover"}
    assert all(c["link"] == entry.id for c in sell_g["children"])


def test_adopt_upgrades_empty_tp_instead_of_minting_second():
    cfg = TraderConfig(spread_points=0, epic="IX.D.CAC.IFD.IP")
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8350
    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=8350.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
            deal_id="WO_ENTRY",
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
            deal_id="",
        )
    )
    # Entry already filled — gone from working book.
    ledger.working_orders.pop(entry.id, None)

    result = adopt_ig_snapshot_into_ledger(
        ledger,
        positions=[
            {
                "deal_id": "DI_POS",
                "epic": cfg.epic,
                "side": Side.SELL,
                "size": 1.0,
                "level": 8350.0,
                "limit_level": 8325.0,
            }
        ],
        working_orders=[],
        epic=cfg.epic,
        mode="additive",
    )
    tps = [o for o in ledger.working_orders.values() if o.purpose == OrderPurpose.TP]
    assert len(tps) == 1
    assert tps[0].id == tp.id
    assert tps[0].deal_id == "attached:DI_POS:tp"
    assert any(r.get("upgraded") for r in result["imported_orders"])


def test_parse_attached_deal_id_legacy_two_part():
    from chatbot.trader.models import attached_deal_id, parse_attached_deal_id

    assert attached_deal_id("DI1", OrderPurpose.TP) == "attached:DI1:tp"
    assert parse_attached_deal_id("attached:DI1:tp") == ("DI1", "tp")
    assert parse_attached_deal_id("attached:DI1") == ("DI1", "tp")
    assert parse_attached_deal_id("WO_REAL") is None
