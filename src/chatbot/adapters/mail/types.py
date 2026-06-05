from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmailMessage:
    to_addr: str
    subject: str
    body_text: str
    from_addr: str
