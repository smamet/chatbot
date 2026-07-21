from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Attachment:
    mime_type: str
    data: bytes
    filename: str | None = None
