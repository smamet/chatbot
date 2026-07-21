from __future__ import annotations

from chatbot.automation.modules.registry import module_for_hook_type


def test_module_for_order_hook() -> None:
    mod = module_for_hook_type("order.create")
    assert mod is not None
    assert mod.id == "core.orders"


def test_module_for_quote_hook() -> None:
    mod = module_for_hook_type("quote.create")
    assert mod is not None
    assert mod.id == "erpnext.quote"


def test_quote_module_uses_validation_fulfillment() -> None:
    mod = module_for_hook_type("quote.create")
    assert mod is not None
    assert mod.fulfillment_mode.value == "validation"
