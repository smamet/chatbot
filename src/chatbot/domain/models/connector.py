from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class ConnectorDirection(StrEnum):
    IN = "in"
    OUT = "out"
    BOTH = "both"


class ConnectorType(StrEnum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    MESSENGER = "messenger"
    INSTAGRAM = "instagram"
    CHAT = "chat"
    IG = "ig"


class ConnectorMode(StrEnum):
    DIRECT = "direct"
    VALIDATION = "validation"


@dataclass(frozen=True)
class Connector:
    id: int
    tenant_id: int
    direction: ConnectorDirection
    type: ConnectorType
    mode: ConnectorMode
    config: dict[str, Any]
    active: bool
    created_at: datetime
    updated_at: datetime
