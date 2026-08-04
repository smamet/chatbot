from __future__ import annotations

from typing import Protocol

from evenor.adapters.mail.types import EmailMessage


class EmailSender(Protocol):
    def send(self, message: EmailMessage) -> str: ...
