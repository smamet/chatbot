from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol


class FulfillmentMode(StrEnum):
    WORKER = "worker"
    VALIDATION = "validation"


@dataclass(frozen=True, slots=True)
class ParsedHook:
    hook_type: str
    payload: dict[str, Any]


class HookModule(Protocol):
    id: str
    label: str
    description: str
    hook_type_prefixes: tuple[str, ...]
    requires_integration: str | None
    fulfillment_mode: FulfillmentMode
    ui_enabled: bool

    def prompt_fragment(self) -> str: ...

    def matches(self, hook_type: str) -> bool: ...

    def parse(self, payload: dict[str, Any]) -> ParsedHook | None: ...

    def handle_worker(self, session: Any, hook: Any) -> None: ...
