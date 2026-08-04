from __future__ import annotations

import json

from evenor.application.hook_extractor import extract_hook
from evenor.domain.constants import HOOK_MARKER, LEGACY_HOOK_MARKER


def test_extract_hook_no_marker() -> None:
    out = extract_hook("Hello there")
    assert out.clean_reply == "Hello there"
    assert out.hook_type is None


def test_extract_hook_with_type() -> None:
    payload = {"type": "lead.capture", "email": "a@b.com"}
    text = f"Thanks.\n{HOOK_MARKER}\n{json.dumps(payload)}"
    out = extract_hook(text)
    assert out.clean_reply == "Thanks."
    assert out.hook_type == "lead.capture"
    assert out.payload_json is not None
    assert json.loads(out.payload_json) == payload


def test_extract_hook_legacy_action_maps_to_order() -> None:
    payload = {"action": "create", "tel": "123"}
    text = f"OK\n{LEGACY_HOOK_MARKER}\n{json.dumps(payload)}"
    out = extract_hook(text)
    assert out.hook_type == "order"
    assert json.loads(out.payload_json or "{}")["action"] == "create"


def test_extract_hook_invalid_json() -> None:
    text = f"Hi\n{HOOK_MARKER}\nnot-json"
    out = extract_hook(text)
    assert out.clean_reply == "Hi"
    assert out.hook_type is None
