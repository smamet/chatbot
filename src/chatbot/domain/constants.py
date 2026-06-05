from __future__ import annotations

HOOK_MARKER = "===HOOK==="
LEGACY_HOOK_MARKER = "===JF030A==="

DEFAULT_HOOK_INSTRUCTIONS = """When you need to trigger an action, append the following block AFTER your reply (never inside it):
===HOOK===
{"type": "<action_type>", ...payload fields...}

The type field identifies the action. Examples:
- "order.create", "order.update", "order.delete" for order management
- "lead.capture" for lead generation
- Any custom type agreed with the system operator

Only emit ===HOOK=== when an action must actually be triggered. Never invent payloads."""
