from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any


class IntegrationType(StrEnum):
    ERPNEXT = "erpnext"
    QUICKBOOKS = "quickbooks"
    CAC40_BACKTEST = "cac40_backtest"


@dataclass(frozen=True)
class Integration:
    id: int
    tenant_id: int
    type: IntegrationType
    config: dict[str, Any]
    active: bool
    created_at: datetime
    updated_at: datetime
