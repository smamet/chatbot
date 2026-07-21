from __future__ import annotations

HOOK_MARKER = "===HOOK==="
LEGACY_HOOK_MARKER = "===JF030A==="

BASE_HOOK_FORMAT = """When you need to trigger an action, append the following block AFTER your reply (never inside it):
===HOOK===
{"type": "<action_type>", ...payload fields...}

The type field identifies the action.
Only emit ===HOOK=== when an action must actually be triggered. Never invent payloads."""

DEFAULT_HOOK_INSTRUCTIONS = BASE_HOOK_FORMAT + """

Examples (legacy combined prompt):
- "order.create", "order.update", "order.delete" for order management
- "lead.capture" for lead generation
"""
