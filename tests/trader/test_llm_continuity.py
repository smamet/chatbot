from __future__ import annotations

import json

from chatbot.trader.llm_decision import build_user_payload, summarize_decision
from chatbot.trader.models import LlmAction, LlmAnalysis, LlmDecision, MarketSnapshot


def test_build_user_payload_includes_continuity_fields():
    snap = MarketSnapshot(symbol="CAC40", last_price=8300.0, phase="Flat")
    last = {
        "bias": "short_from_resistance",
        "support": 8270.0,
        "resistance": 8340.0,
        "actions": [{"op": "place_limit", "side": "SELL", "level": 8340.0, "purpose": "entry"}],
    }
    raw = build_user_payload(
        snap,
        "Flat",
        order_size=1.0,
        max_open_positions=4,
        last_decision=last,
        allow_market_orders=False,
    )
    data = json.loads(raw)
    assert data["order_size"] == 1.0
    assert data["max_open_positions"] == 4
    assert data["last_decision"]["bias"] == "short_from_resistance"
    assert "Do not duplicate" in data["instructions"]
    assert "market_open is disabled" in data["instructions"]


def test_summarize_decision():
    decision = LlmDecision(
        analysis=LlmAnalysis(support=1.0, resistance=2.0, bias="hold"),
        actions=[LlmAction(op="cancel_order", order_id="o1")],
    )
    summary = summarize_decision(decision)
    assert summary is not None
    assert summary["bias"] == "hold"
    assert summary["actions"][0]["order_id"] == "o1"
