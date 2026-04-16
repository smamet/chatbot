from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chatbot.domain.models.message import ChatMessage


@dataclass(frozen=True, slots=True)
class LlmUsage:
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LlmResult:
    text: str
    usage: LlmUsage


@runtime_checkable
class LlmClient(Protocol):
    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
    ) -> LlmResult:
        """Generate assistant reply from ordered chat history (excluding pending user turn if caller merges)."""
        ...
