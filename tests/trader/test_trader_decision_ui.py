"""Unit tests for LLM decision action summary chips."""

from chatbot.application.trader_decision_ui import (
    decision_search_ids,
    summarize_llm_action,
    summarize_llm_actions,
)


def test_summarize_empty_is_hold() -> None:
    assert summarize_llm_actions(None) == ["Hold"]
    assert summarize_llm_actions([]) == ["Hold"]


def test_summarize_amend_tp_and_entry() -> None:
    assert summarize_llm_action({"op": "amend_order", "purpose": "tp", "level": 8120}) == (
        "TP change → 8120"
    )
    assert summarize_llm_action(
        {"op": "amend_order", "purpose": "entry", "level": 8050.5}
    ) == "Entry change → 8050.5"


def test_summarize_place_entry_tp_hedge() -> None:
    assert summarize_llm_action(
        {"op": "place_limit", "purpose": "entry", "level": 8050}
    ) == "Entry place @ 8050"
    assert summarize_llm_action(
        {"op": "place_stop", "purpose": "entry", "level": 8040}
    ) == "Entry stop @ 8040"
    assert summarize_llm_action(
        {"op": "place_limit", "purpose": "tp", "level": 8120}
    ) == "TP place @ 8120"
    assert summarize_llm_action(
        {"op": "place_stop", "purpose": "hedge_cover", "level": 8000}
    ) == "Hedge @ 8000"


def test_summarize_cancel_and_market() -> None:
    assert summarize_llm_action({"op": "cancel_order", "purpose": "hedge_cover"}) == (
        "Cancel hedge"
    )
    assert summarize_llm_action({"op": "cancel_order", "purpose": "tp"}) == "Cancel TP"
    assert summarize_llm_action({"op": "cancel_order", "purpose": "entry"}) == (
        "Cancel entry (limit)"
    )
    assert summarize_llm_action(
        {"op": "cancel_order", "purpose": "entry", "type": "STOP"}
    ) == "Cancel entry (stop)"
    assert summarize_llm_action({"op": "cancel_order", "order_id": "o12"}) == "Cancel o12"
    assert summarize_llm_action({"op": "market_open"}) == "Market open"
    assert summarize_llm_action({"op": "market_close"}) == "Market close"


def test_summarize_cancel_resolves_from_working_orders() -> None:
    working = [
        {"id": "o10", "purpose": "entry", "type": "LIMIT"},
        {"id": "o11", "purpose": "tp", "type": "LIMIT"},
        {"id": "o12", "purpose": "hedge_cover", "type": "STOP"},
    ]
    chips = summarize_llm_actions(
        [
            {"op": "cancel_order", "order_id": "o10"},
            {"op": "cancel_order", "order_id": "o11"},
            {"op": "cancel_order", "order_id": "o12"},
        ],
        working_orders=working,
    )
    assert chips == ["Cancel entry (limit)", "Cancel TP", "Cancel hedge"]


def test_summarize_actions_list() -> None:
    chips = summarize_llm_actions(
        [
            {"op": "amend_order", "purpose": "tp", "level": 8120},
            {"op": "market_close"},
        ]
    )
    assert chips == ["TP change → 8120", "Market close"]


def test_decision_search_ids_from_actions_snapshot_and_gate() -> None:
    ids = decision_search_ids(
        actions=[
            {"op": "cancel_order", "order_id": "o1801"},
            {"op": "market_close", "position_id": "p1581"},
        ],
        snapshot={
            "working_orders": [{"id": "o123", "purpose": "entry"}],
            "positions": [{"id": "p99"}],
        },
        executed=["place_limit:o456@1.15"],
        rejected=["cancel_order:unknown:o999"],
    )
    assert ids == ["o123", "o1801", "o456", "o999", "p1581", "p99"]
