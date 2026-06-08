from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OutboundAttachment:
    filename: str
    data: bytes
    mime_type: str = "application/pdf"
