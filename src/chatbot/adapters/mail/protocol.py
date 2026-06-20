from __future__ import annotations

from typing import Protocol

from chatbot.adapters.mail.types import EmailMessage


class EmailSender(Protocol):
    def send(self, message: EmailMessage) -> str: ...
