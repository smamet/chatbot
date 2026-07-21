from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    LIMIT = "LIMIT"
    STOP = "STOP"
    MARKET = "MARKET"


class OrderPurpose(StrEnum):
    ENTRY = "entry"
    TP = "tp"
    HEDGE_COVER = "hedge_cover"
    CLOSE = "close"


class LegRole(StrEnum):
    PRIMARY = "primary"
    HEDGE = "hedge"
    HEDGE_COVER = "hedge_cover"


@dataclass
class WorkingOrder:
    id: str
    type: OrderType
    side: Side
    level: float
    size: float
    purpose: OrderPurpose
    position_id: str | None = None  # linked leg for TP/close
    client_ref: str = ""
    deal_id: str = ""  # IG dealId after confirm (live)
    active_from_bar: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "side": self.side.value,
            "level": self.level,
            "size": self.size,
            "purpose": self.purpose.value,
            "position_id": self.position_id,
            "client_ref": self.client_ref,
            "deal_id": self.deal_id,
            "active_from_bar": self.active_from_bar,
        }


@dataclass
class PositionLeg:
    id: str
    side: Side
    size: float
    entry: float
    role: LegRole
    opened_bar: int = 0
    opened_at: str = ""
    upl: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "side": self.side.value,
            "size": self.size,
            "entry": self.entry,
            "role": self.role.value,
            "upl": self.upl,
        }


@dataclass
class ClosedTrade:
    id: str
    side: Side
    size: float
    entry: float
    exit: float
    role: LegRole
    realized_pnl: float
    opened_at: str
    closed_at: str
    bars_held: int


@dataclass
class MarketSnapshot:
    symbol: str
    last_price: float
    positions: list[PositionLeg] = field(default_factory=list)
    working_orders: list[WorkingOrder] = field(default_factory=list)
    account_upl: float = 0.0
    last_levels: dict[str, Any] = field(default_factory=dict)
    phase: str = "Flat"
    bar_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "positions": [p.to_dict() for p in self.positions],
            "working_orders": [o.to_dict() for o in self.working_orders],
            "account_upl": self.account_upl,
            "last_levels": self.last_levels,
            "phase": self.phase,
        }


@dataclass
class LlmAnalysis:
    support: float | None = None
    resistance: float | None = None
    bias: str = "hold"
    rsi_note: str = ""
    pivot_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LlmAction:
    op: str
    side: str | None = None
    level: float | None = None
    size: float | None = None
    purpose: str | None = None
    order_id: str | None = None
    position_id: str | None = None
    reason: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LlmAction:
        return cls(
            op=str(data.get("op", "")),
            side=data.get("side"),
            level=float(data["level"]) if data.get("level") is not None else None,
            size=float(data["size"]) if data.get("size") is not None else None,
            purpose=data.get("purpose"),
            order_id=data.get("order_id"),
            position_id=data.get("position_id"),
            reason=str(data.get("reason", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LlmDecision:
    analysis: LlmAnalysis
    actions: list[LlmAction] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
        }
