from chatbot.cac40.config import Cac40Config
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.models import LlmAction, LlmAnalysis, LlmDecision, Side
from chatbot.cac40.risk_gate import RiskGate


def _decision(*actions: LlmAction) -> LlmDecision:
    return LlmDecision(
        analysis=LlmAnalysis(support=99, resistance=101, bias="hold"),
        actions=list(actions),
    )


def test_rejects_market_when_disabled():
    cfg = Cac40Config(allow_market_orders=False)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    gate = RiskGate(cfg, ledger)
    result = gate.apply(_decision(LlmAction(op="market_open", side="BUY", size=1)))
    assert any("market_disabled" in r for r in result.rejected)


def test_allows_profitable_market_close_when_market_orders_disabled():
    """Hedge→new S/R: lock a winning hedge even when allow_market_orders is off."""
    from chatbot.cac40.models import LegRole

    cfg = Cac40Config(allow_market_orders=False, spread_points=0, prevent_loss_exits=True)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    primary = ledger._open_leg(Side.BUY, 1.0, 100.0, LegRole.PRIMARY)
    hedge = ledger._open_leg(Side.SELL, 1.0, 95.0, LegRole.HEDGE)
    ledger.last_price = 90  # short hedge profitable
    gate = RiskGate(cfg, ledger)

    lost = gate.apply(_decision(LlmAction(op="market_close", position_id=primary.id)))
    assert any("market_disabled" in r for r in lost.rejected)
    assert primary.id in ledger.positions

    won = gate.apply(_decision(LlmAction(op="market_close", position_id=hedge.id)))
    assert won.executed
    assert hedge.id not in ledger.positions
    assert primary.id in ledger.positions


def test_max_open_positions_counts_legs():
    cfg = Cac40Config(max_open_positions=1, allow_market_orders=True, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(LlmAction(op="place_limit", side="BUY", level=98, size=1, purpose="entry"))
    )
    assert any("max_positions" in r for r in result.rejected)


def test_rejects_duplicate_entry_working_order():
    cfg = Cac40Config(order_size=1.0, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    gate = RiskGate(cfg, ledger)
    first = gate.apply(
        _decision(LlmAction(op="place_limit", side="SELL", level=101, size=1, purpose="entry"))
    )
    assert first.executed
    second = gate.apply(
        _decision(LlmAction(op="place_limit", side="SELL", level=100.5, size=1, purpose="entry"))
    )
    assert any("duplicate_entry" in r for r in second.rejected)
    assert len(ledger.working_orders) == 1


def test_rejects_hedge_cover_without_primary():
    cfg = Cac40Config(order_size=1.0, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(op="place_stop", side="BUY", level=102, size=1, purpose="hedge_cover")
        )
    )
    assert any("hedge_without_primary" in r for r in result.rejected)
    assert not ledger.working_orders


def test_auto_links_tp_when_single_leg():
    cfg = Cac40Config(order_size=1.0, allow_market_orders=True, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.SELL, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(LlmAction(op="place_limit", side="BUY", level=98, size=5, purpose="tp"))
    )
    assert result.executed
    order = next(iter(ledger.working_orders.values()))
    assert order.position_id == pid
    assert order.size == 1.0  # clamped


def test_rejects_tp_without_position_when_multiple_legs():
    cfg = Cac40Config(order_size=1.0, allow_market_orders=True, spread_points=0, max_open_positions=4)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.market_open(Side.BUY, 1)
    ledger.market_open(Side.SELL, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(LlmAction(op="place_limit", side="SELL", level=101, size=1, purpose="tp"))
    )
    assert any("missing_position_id" in r for r in result.rejected)


def test_rejects_duplicate_hedge_for_same_position():
    cfg = Cac40Config(order_size=1.0, allow_market_orders=True, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.SELL, 1)
    gate = RiskGate(cfg, ledger)
    first = gate.apply(
        _decision(
            LlmAction(
                op="place_stop",
                side="BUY",
                level=102,
                size=1,
                purpose="hedge_cover",
                position_id=pid,
            )
        )
    )
    assert first.executed
    second = gate.apply(
        _decision(
            LlmAction(
                op="place_stop",
                side="BUY",
                level=103,
                size=1,
                purpose="hedge_cover",
                position_id=pid,
            )
        )
    )
    assert any("duplicate_hedge" in r for r in second.rejected)


def test_loss_exit_flag_off_allows_losing_market_close():
    cfg = Cac40Config(allow_market_orders=True, spread_points=0, prevent_loss_exits=False)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    ledger.last_price = 95  # underwater
    gate = RiskGate(cfg, ledger)
    result = gate.apply(_decision(LlmAction(op="market_close", position_id=pid)))
    assert result.executed
    assert pid not in ledger.positions


def test_loss_exit_flag_on_blocks_losing_market_close_allows_profit():
    cfg = Cac40Config(allow_market_orders=True, spread_points=0, prevent_loss_exits=True)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    ledger.last_price = 95
    gate = RiskGate(cfg, ledger)
    lost = gate.apply(_decision(LlmAction(op="market_close", position_id=pid)))
    assert any("loss_exit_blocked" in r for r in lost.rejected)
    assert pid in ledger.positions

    ledger.last_price = 105
    won = gate.apply(_decision(LlmAction(op="market_close", position_id=pid)))
    assert won.executed
    assert pid not in ledger.positions


def test_loss_exit_flag_on_blocks_losing_tp_allows_profit_tp():
    cfg = Cac40Config(allow_market_orders=True, spread_points=0, prevent_loss_exits=True)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)

    bad = gate.apply(
        _decision(LlmAction(op="place_limit", side="SELL", level=99, size=1, purpose="tp", position_id=pid))
    )
    assert any("loss_exit_blocked" in r for r in bad.rejected)
    assert not ledger.working_orders

    good = gate.apply(
        _decision(LlmAction(op="place_limit", side="SELL", level=101, size=1, purpose="tp", position_id=pid))
    )
    assert good.executed
    assert len(ledger.working_orders) == 1


def test_loss_exit_flag_on_rejects_losing_close_fill():
    from chatbot.cac40.models import OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(spread_points=0, prevent_loss_exits=True)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    # Bypass RiskGate: resting TP below entry
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=99,
            size=1,
            purpose=OrderPurpose.TP,
            position_id=pid,
        )
    )
    events = ledger.process_bar({"open": 100, "high": 100, "low": 98, "close": 99})
    assert any(e.get("type") == "rejected_fill" and e.get("reason") == "loss_exit_blocked" for e in events)
    assert pid in ledger.positions
    assert not ledger.working_orders  # cancelled, not left working


def test_rejects_same_level_same_side_entry():
    from chatbot.cac40.models import LegRole

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
        llm_level_band_points=15.0,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8345
    ledger._open_leg(Side.BUY, 1.0, 8341.0, LegRole.PRIMARY)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(op="place_limit", side="BUY", level=8340.0, size=1, purpose="entry")
        )
    )
    assert any("same_level_primary" in r for r in result.rejected)
    assert not any(o.purpose.value == "entry" for o in ledger.working_orders.values())


def test_rejects_opposite_side_entry_while_longs_unhedged():
    """Naked BUY legs must be hedged before a new SELL entry is allowed."""
    from chatbot.cac40.models import LegRole, OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
        llm_level_band_points=15.0,
        prevent_loss_exits=True,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8380
    ledger._open_leg(Side.BUY, 1.0, 8420.0, LegRole.PRIMARY)
    ledger._open_leg(Side.BUY, 1.0, 8341.0, LegRole.PRIMARY)
    gate = RiskGate(cfg, ledger)
    naked = gate.apply(
        _decision(
            LlmAction(op="place_limit", side="SELL", level=8380.0, size=1, purpose="entry"),
        )
    )
    assert any("unhedged_open_book" in r for r in naked.rejected)

    # Hedge longs first in the same decision, then short entry is allowed.
    ok = gate.apply(
        _decision(
            LlmAction(
                op="place_stop", side="SELL", level=8330.0, size=1, purpose="hedge_cover"
            ),
            LlmAction(op="place_limit", side="SELL", level=8380.0, size=1, purpose="entry"),
            LlmAction(op="place_limit", side="BUY", level=8350.0, size=1, purpose="tp"),
            LlmAction(
                op="place_stop", side="BUY", level=8390.0, size=1, purpose="hedge_cover"
            ),
        )
    )
    assert any(e.startswith("place_limit:") and "@8380" in e for e in ok.executed)
    assert not any("unhedged_open_book" in r for r in ok.rejected)
    long_hedge = next(
        o
        for o in ledger.working_orders.values()
        if o.purpose == OrderPurpose.HEDGE_COVER and o.side == Side.SELL
    )
    assert long_hedge.size == 2.0


def test_allows_opposite_side_short_entry_when_longs_hedged():
    """Open BUY with working SELL hedge → SELL entry (short) is allowed."""
    from chatbot.cac40.models import LegRole, OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
        llm_level_band_points=15.0,
        prevent_loss_exits=True,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8380
    p1 = ledger._open_leg(Side.BUY, 1.0, 8341.0, LegRole.PRIMARY)
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.SELL,
            level=8330.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            position_id=p1.id,
        )
    )
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(op="place_limit", side="SELL", level=8420.0, size=1, purpose="entry"),
            LlmAction(op="place_limit", side="BUY", level=8380.0, size=1, purpose="tp"),
            LlmAction(
                op="place_stop", side="BUY", level=8430.0, size=1, purpose="hedge_cover"
            ),
        )
    )
    assert len(result.executed) == 3
    assert not result.rejected
    assert any(
        o.purpose.value == "entry" and o.side == Side.SELL
        for o in ledger.working_orders.values()
    )


def test_same_level_entry_allowed_after_primary_closed():
    from chatbot.cac40.models import LegRole

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
        llm_level_band_points=15.0,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8350
    leg = ledger._open_leg(Side.BUY, 1.0, 8341.0, LegRole.PRIMARY)
    ledger.close_position(leg.id, 8350.0)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(op="place_limit", side="BUY", level=8340.0, size=1, purpose="entry")
        )
    )
    assert result.executed
    assert not any("same_level_primary" in r for r in result.rejected)


def test_hedge_cover_sizes_to_full_long_exposure():
    """Two BUY legs → SELL hedge_cover must be size 2, not order_size."""
    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8340
    ledger.market_open(Side.BUY, 1)
    ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="place_stop",
                side="SELL",
                level=8330.0,
                size=1.0,
                purpose="hedge_cover",
            )
        )
    )
    assert result.executed
    assert not result.rejected
    hedge = next(iter(ledger.working_orders.values()))
    assert hedge.purpose.value == "hedge_cover"
    assert hedge.size == 2.0


def test_hedge_cover_includes_working_entry_with_open_leg():
    """Open BUY + new BUY entry at a different level → SELL hedge covers both (size 2)."""
    from chatbot.cac40.models import LegRole

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
        prevent_loss_exits=True,
        llm_level_band_points=15.0,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8380
    # Primary far from new support entry so same_level_primary does not fire.
    ledger._open_leg(Side.BUY, 1.0, 8420.0, LegRole.PRIMARY)
    gate = RiskGate(cfg, ledger)
    # Hedge existing long first, then new entry (+ hedge sized to cover both).
    result = gate.apply(
        _decision(
            LlmAction(
                op="place_stop", side="SELL", level=8400, size=1, purpose="hedge_cover"
            ),
            LlmAction(op="place_limit", side="BUY", level=8338, size=1, purpose="entry"),
            LlmAction(op="place_limit", side="SELL", level=8360, size=1, purpose="tp"),
            LlmAction(
                op="place_stop", side="SELL", level=8330, size=1, purpose="hedge_cover"
            ),
        )
    )
    assert not any("unhedged_open_book" in r for r in result.rejected)
    assert any(o.purpose.value == "entry" for o in ledger.working_orders.values())
    sell_hedges = [
        o
        for o in ledger.working_orders.values()
        if o.purpose.value == "hedge_cover" and o.side == Side.SELL
    ]
    assert sum(o.size for o in sell_hedges) == 2.0


def test_hedge_cover_residual_after_filled_opposing_hedge():
    """+2 BUY and −1 SELL filled → further SELL hedge covers residual 1 only."""
    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8340
    ledger.market_open(Side.BUY, 1)
    ledger.market_open(Side.BUY, 1)
    ledger.market_open(Side.SELL, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="place_stop",
                side="SELL",
                level=8330.0,
                size=1.0,
                purpose="hedge_cover",
            )
        )
    )
    assert result.executed
    hedge = next(iter(ledger.working_orders.values()))
    assert hedge.size == 1.0


def test_hedge_cover_residual_after_working_hedge():
    """+2 BUY with a resting 1-lot SELL hedge → next SELL hedge is residual 1."""
    from chatbot.cac40.models import OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8340
    p1 = ledger.market_open(Side.BUY, 1)
    ledger.market_open(Side.BUY, 1)
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.SELL,
            level=8330.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
            position_id=p1,
        )
    )
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="place_stop",
                side="SELL",
                level=8325.0,
                size=1.0,
                purpose="hedge_cover",
            )
        )
    )
    assert result.executed
    hedges = [
        o for o in ledger.working_orders.values() if o.purpose == OrderPurpose.HEDGE_COVER
    ]
    assert len(hedges) == 2
    assert sorted(h.size for h in hedges) == [1.0, 1.0]


def test_bracket_entry_tp_hedge_accepted():
    """Screenshot decision: SELL entry + BUY TP + BUY stop hedge in one batch."""
    cfg = Cac40Config(order_size=1.0, spread_points=0, prevent_loss_exits=True)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8447.0
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="place_limit",
                side="SELL",
                level=8455.0,
                size=1.0,
                purpose="entry",
            ),
            LlmAction(
                op="place_limit",
                side="BUY",
                level=8425.0,
                size=1.0,
                purpose="tp",
            ),
            LlmAction(
                op="place_stop",
                side="BUY",
                level=8465.0,
                size=1.0,
                purpose="hedge_cover",
            ),
        )
    )
    assert len(result.executed) == 3
    assert not result.rejected
    by_purpose = {o.purpose.value: o for o in ledger.working_orders.values()}
    assert "entry" in by_purpose and "tp" in by_purpose and "hedge_cover" in by_purpose
    entry = by_purpose["entry"]
    assert by_purpose["tp"].parent_order_id == entry.id
    assert by_purpose["hedge_cover"].parent_order_id == entry.id
    assert by_purpose["tp"].position_id is None


def test_bracket_hedge_not_beyond_entry_rejected():
    cfg = Cac40Config(order_size=1.0, spread_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(op="place_limit", side="SELL", level=101, size=1, purpose="entry"),
            LlmAction(
                op="place_stop",
                side="BUY",
                level=100,  # below entry — would fire without primary
                size=1,
                purpose="hedge_cover",
            ),
        )
    )
    assert any("hedge_not_beyond_entry" in r for r in result.rejected)
    assert any(o.purpose.value == "entry" for o in ledger.working_orders.values())
    assert not any(o.purpose.value == "hedge_cover" for o in ledger.working_orders.values())


def test_bracket_prefers_working_entry_over_existing_leg():
    """With one open leg, a new entry+TP+hedge must bracket the entry, not the old leg."""
    cfg = Cac40Config(
        order_size=1.0,
        spread_points=0,
        prevent_loss_exits=True,
        max_open_positions=4,
        allow_market_orders=True,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    old_pid = ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            # Protect existing long before opposite-side short entry.
            LlmAction(
                op="place_stop", side="SELL", level=95, size=1, purpose="hedge_cover"
            ),
            LlmAction(op="place_limit", side="SELL", level=101, size=1, purpose="entry"),
            LlmAction(op="place_limit", side="BUY", level=98, size=1, purpose="tp"),
            LlmAction(
                op="place_stop", side="BUY", level=103, size=1, purpose="hedge_cover"
            ),
        )
    )
    assert not result.rejected
    entry = next(o for o in ledger.working_orders.values() if o.purpose.value == "entry")
    tp = next(o for o in ledger.working_orders.values() if o.purpose.value == "tp")
    short_hedge = next(
        o
        for o in ledger.working_orders.values()
        if o.purpose.value == "hedge_cover" and o.side == Side.BUY
    )
    assert tp.parent_order_id == entry.id
    assert tp.position_id is None
    assert short_hedge.parent_order_id == entry.id
    assert tp.position_id != old_pid


def test_ig_working_order_body_includes_limit_level():
    from chatbot.cac40.ig_connector import IgConnector
    from chatbot.cac40.models import OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(epic="IX.D.CAC.IFD.IP")
    conn = IgConnector(cfg, dry_run=True)
    order = WorkingOrder(
        id="o1",
        type=OrderType.LIMIT,
        side=Side.SELL,
        level=8455.0,
        size=1.0,
        purpose=OrderPurpose.ENTRY,
    )
    body = conn._ig_working_order_body(
        order, limit_level=8425.0, stop_level=8465.0
    )  # noqa: SLF001
    assert body["limitLevel"] == 8425.0
    assert body["stopLevel"] == 8465.0
    assert body["level"] == 8455.0
    conn.close()


def test_amend_hedge_cover_grows_to_full_cover_excluding_self():
    """Short + working entry + size-1 hedge → amend sizes hedge to 2."""
    from chatbot.cac40.models import LegRole, OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8360
    ledger._open_leg(Side.SELL, 1.0, 8348.0, LegRole.PRIMARY)
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.SELL,
            level=8370.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
        )
    )
    hedge = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.BUY,
            level=8400.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
        )
    )
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="amend_order",
                order_id=hedge.id,
                side="BUY",
                level=8410.0,
                size=2.0,
                purpose="hedge_cover",
            )
        )
    )
    assert not result.rejected
    assert hedge.size == 2.0
    assert hedge.level == 8410.0
    assert any(f"amend_order:{hedge.id}->8410.0x2" in e for e in result.executed)


def test_amend_hedge_cover_with_second_hedge_uses_residual():
    """Two hedges: amending one targets residual, not full book."""
    from chatbot.cac40.models import LegRole, OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(
        order_size=1.0,
        allow_market_orders=True,
        spread_points=0,
        max_open_positions=4,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8360
    ledger._open_leg(Side.BUY, 1.0, 8300.0, LegRole.PRIMARY)
    ledger._open_leg(Side.BUY, 1.0, 8310.0, LegRole.PRIMARY)
    hedge_a = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.SELL,
            level=8280.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
        )
    )
    ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.STOP,
            side=Side.SELL,
            level=8270.0,
            size=1.0,
            purpose=OrderPurpose.HEDGE_COVER,
        )
    )
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="amend_order",
                order_id=hedge_a.id,
                side="SELL",
                level=8285.0,
                size=2.0,
                purpose="hedge_cover",
            )
        )
    )
    assert not result.rejected
    # Other hedge still covers 1 → amend keeps size 1 (residual), only level moves.
    assert hedge_a.size == 1.0
    assert hedge_a.level == 8285.0
    assert any(f"amend_order:{hedge_a.id}->8285.0" in e for e in result.executed)
    assert not any("x2" in e for e in result.executed)


def test_amend_entry_clamps_size_to_order_size():
    from chatbot.cac40.models import OrderPurpose, OrderType, WorkingOrder

    cfg = Cac40Config(order_size=1.0, spread_points=0, max_open_positions=4)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 8360
    entry = ledger.place_order(
        WorkingOrder(
            id="",
            type=OrderType.LIMIT,
            side=Side.BUY,
            level=8300.0,
            size=1.0,
            purpose=OrderPurpose.ENTRY,
        )
    )
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="amend_order",
                order_id=entry.id,
                side="BUY",
                level=8290.0,
                size=5.0,
                purpose="entry",
            )
        )
    )
    assert not result.rejected
    assert entry.level == 8290.0
    assert entry.size == 1.0
    assert any(f"amend_order:{entry.id}->8290.0" in e for e in result.executed)
