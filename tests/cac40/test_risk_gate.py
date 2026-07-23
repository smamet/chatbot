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
    cfg = Cac40Config(allow_market_orders=False, min_exit_profit_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    gate = RiskGate(cfg, ledger)
    result = gate.apply(_decision(LlmAction(op="market_open", side="BUY", size=1)))
    assert any("market_disabled" in r for r in result.rejected)


def test_max_open_positions_counts_legs():
    cfg = Cac40Config(max_open_positions=1, allow_market_orders=True, spread_points=0, min_exit_profit_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(LlmAction(op="place_limit", side="BUY", level=98, size=1, purpose="entry"))
    )
    assert any("max_positions" in r for r in result.rejected)


def test_rejects_duplicate_entry_working_order():
    cfg = Cac40Config(order_size=1.0, spread_points=0, min_exit_profit_points=0)
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
    cfg = Cac40Config(order_size=1.0, spread_points=0, min_exit_profit_points=0)
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
    cfg = Cac40Config(order_size=1.0, allow_market_orders=True, spread_points=0, min_exit_profit_points=0)
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
    cfg = Cac40Config(order_size=1.0, allow_market_orders=True, spread_points=0, max_open_positions=4, min_exit_profit_points=0)
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
    cfg = Cac40Config(order_size=1.0, allow_market_orders=True, spread_points=0, min_exit_profit_points=0)
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
    cfg = Cac40Config(allow_market_orders=True, spread_points=0, prevent_loss_exits=False, min_exit_profit_points=0)
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    ledger.last_price = 95  # underwater
    gate = RiskGate(cfg, ledger)
    result = gate.apply(_decision(LlmAction(op="market_close", position_id=pid)))
    assert result.executed
    assert pid not in ledger.positions


def test_loss_exit_flag_on_blocks_losing_market_close_allows_profit():
    cfg = Cac40Config(allow_market_orders=True, spread_points=0, prevent_loss_exits=True, min_exit_profit_points=0)
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
    cfg = Cac40Config(allow_market_orders=True, spread_points=0, prevent_loss_exits=True, min_exit_profit_points=0)
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

    cfg = Cac40Config(spread_points=0, prevent_loss_exits=True, min_exit_profit_points=0)
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


def test_bracket_entry_tp_hedge_accepted():
    """Screenshot decision: SELL entry + BUY TP + BUY stop hedge in one batch."""
    cfg = Cac40Config(order_size=1.0, spread_points=0, prevent_loss_exits=True, min_exit_profit_points=0)
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
    cfg = Cac40Config(order_size=1.0, spread_points=0, min_exit_profit_points=0)
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
        min_exit_profit_points=0,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    old_pid = ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(op="place_limit", side="SELL", level=101, size=1, purpose="entry"),
            LlmAction(op="place_limit", side="BUY", level=98, size=1, purpose="tp"),
            LlmAction(
                op="place_stop", side="BUY", level=103, size=1, purpose="hedge_cover"
            ),
        )
    )
    assert len(result.executed) == 3
    assert not result.rejected
    by_purpose = {o.purpose.value: o for o in ledger.working_orders.values()}
    entry = by_purpose["entry"]
    assert by_purpose["tp"].parent_order_id == entry.id
    assert by_purpose["tp"].position_id is None
    assert by_purpose["hedge_cover"].parent_order_id == entry.id
    assert by_purpose["tp"].position_id != old_pid


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
    body = conn._ig_working_order_body(order, limit_level=8425.0)  # noqa: SLF001
    assert body["limitLevel"] == 8425.0
    assert body["level"] == 8455.0
    conn.close()


def test_min_exit_profit_blocks_small_tp_allows_large():
    cfg = Cac40Config(
        allow_market_orders=True,
        spread_points=0,
        min_exit_profit_points=15,
        prevent_loss_exits=False,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)

    small = gate.apply(
        _decision(
            LlmAction(
                op="place_limit",
                side="SELL",
                level=105,
                size=1,
                purpose="tp",
                position_id=pid,
            )
        )
    )
    assert any("min_profit_blocked" in r for r in small.rejected)
    assert not ledger.working_orders

    ok = gate.apply(
        _decision(
            LlmAction(
                op="place_limit",
                side="SELL",
                level=120,
                size=1,
                purpose="tp",
                position_id=pid,
            )
        )
    )
    assert ok.executed
    assert len(ledger.working_orders) == 1


def test_min_exit_profit_zero_allows_tiny_tp():
    cfg = Cac40Config(
        allow_market_orders=True,
        spread_points=0,
        min_exit_profit_points=0,
        prevent_loss_exits=False,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    result = gate.apply(
        _decision(
            LlmAction(
                op="place_limit",
                side="SELL",
                level=100.5,
                size=1,
                purpose="tp",
                position_id=pid,
            )
        )
    )
    assert result.executed
    assert not result.rejected


def test_min_exit_profit_blocks_amend_into_sub_min():
    cfg = Cac40Config(
        allow_market_orders=True,
        spread_points=0,
        min_exit_profit_points=15,
    )
    ledger = HedgeLedger(config=cfg)
    ledger.last_price = 100
    pid = ledger.market_open(Side.BUY, 1)
    gate = RiskGate(cfg, ledger)
    placed = gate.apply(
        _decision(
            LlmAction(
                op="place_limit",
                side="SELL",
                level=120,
                size=1,
                purpose="tp",
                position_id=pid,
            )
        )
    )
    assert placed.executed
    oid = next(iter(ledger.working_orders))
    amended = gate.apply(
        _decision(LlmAction(op="amend_order", order_id=oid, level=105))
    )
    assert any("min_profit_blocked" in r for r in amended.rejected)
    assert ledger.working_orders[oid].level == 120
