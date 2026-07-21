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
