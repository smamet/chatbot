from __future__ import annotations

from chatbot.application.order_command_extractor import extract_order_command
from chatbot.domain.models.order import OrderAction


def test_extract_without_marker_returns_original_reply() -> None:
    out = extract_order_command("Hello customer")
    assert out.clean_reply == "Hello customer"
    assert out.command is None
    assert out.command_json is None


def test_extract_create_marker_parses_command() -> None:
    text = (
        "Thank you, done.\n"
        "===JF030A===\n"
        '{"action":"create","name":"Ana","tel":"23057770000","products":[{"qty":2,"product":"Diffuser"}]}'
    )
    out = extract_order_command(text)
    assert out.clean_reply == "Thank you, done."
    assert out.command is not None
    assert out.command.action == OrderAction.CREATE
    assert out.command.name == "Ana"
    assert out.command.tel == "23057770000"
    assert len(out.command.products) == 1
    assert out.command.products[0].qty == 2
    assert out.command_json is not None
    assert '"action":"create"' in out.command_json


def test_extract_update_marker_parses_address_and_products() -> None:
    text = (
        "Address updated.\n"
        "===JF030A===\n"
        '{"action":"update","tel":"23057770000","address":"Quatre Bornes","products":[{"qty":"1","product":"Mist"}]}'
    )
    out = extract_order_command(text)
    assert out.command is not None
    assert out.command.action == OrderAction.UPDATE
    assert out.command.address == "Quatre Bornes"
    assert out.command.products[0].product == "Mist"


def test_extract_delete_marker_parses_reason() -> None:
    text = "Cancelled.\n===JF030A===\n" '{"action":"delete","tel":"23057770000","reason":"customer cancelled"}'
    out = extract_order_command(text)
    assert out.command is not None
    assert out.command.action == OrderAction.DELETE
    assert out.command.reason == "customer cancelled"


def test_extract_invalid_json_does_not_crash_or_emit_command() -> None:
    text = "Reply\n===JF030A===\n{not json}"
    out = extract_order_command(text)
    assert out.clean_reply == "Reply"
    assert out.command is None
