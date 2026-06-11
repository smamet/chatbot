from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from chatbot.domain.models.attachment import Attachment
from chatbot.domain.models.message import ChatMessage


from chatbot.domain.models.context_debug import ContextDebugInfo


@dataclass(frozen=True, slots=True)
class LlmUsage:
    prompt_tokens: int | None = None
    candidates_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LlmResult:
    text: str
    usage: LlmUsage
    hook_type: str | None = None
    hook_payload_json: str | None = None
    hook_event_id: int | None = None
    context_debug: ContextDebugInfo | None = None


@runtime_checkable
class LlmClient(Protocol):
    def generate_chat(
        self,
        *,
        system_instruction: str,
        messages: list[ChatMessage],
        attachments: list[Attachment] | None = None,
    ) -> LlmResult:
        """Generate assistant reply from ordered chat history (excluding pending user turn if caller merges)."""
        ...
