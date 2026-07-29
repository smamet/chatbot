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


def attached_deal_id(parent_deal: str, purpose: OrderPurpose | str) -> str:
    """Build ``attached:{parentDeal}:{purpose}`` sentinel for IG stop/limit on a deal."""
    parent = (parent_deal or "").strip()
    if isinstance(purpose, OrderPurpose):
        purpose_s = purpose.value
    else:
        purpose_s = str(purpose or "").strip().lower() or OrderPurpose.TP.value
    if purpose_s == OrderPurpose.CLOSE.value:
        purpose_s = OrderPurpose.TP.value
    return f"attached:{parent}:{purpose_s}"


def parse_attached_deal_id(deal_id: str) -> tuple[str, str] | None:
    """
    Parse ``attached:{parent}[:purpose]``.

    Legacy 2-part ``attached:{deal}`` is treated as purpose ``tp`` (limit attach).
    """
    raw = (deal_id or "").strip()
    if not raw.startswith("attached:"):
        return None
    parts = raw.split(":")
    if len(parts) < 2:
        return None
    parent = (parts[1] or "").strip()
    if not parent:
        return None
    if len(parts) == 2:
        return parent, OrderPurpose.TP.value
    purpose = (parts[2] or "").strip().lower() or OrderPurpose.TP.value
    if purpose == OrderPurpose.CLOSE.value:
        purpose = OrderPurpose.TP.value
    return parent, purpose


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
    parent_order_id: str | None = None  # dormant until parent entry fills
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
            "parent_order_id": self.parent_order_id,
            "client_ref": self.client_ref,
            "deal_id": self.deal_id,
            "active_from_bar": self.active_from_bar,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> WorkingOrder:
        raw = dict(data or {})
        return cls(
            id=str(raw.get("id") or ""),
            type=OrderType(str(raw.get("type") or OrderType.LIMIT.value)),
            side=Side(str(raw.get("side") or Side.BUY.value)),
            level=float(raw.get("level") or 0.0),
            size=float(raw.get("size") or 0.0),
            purpose=OrderPurpose(str(raw.get("purpose") or OrderPurpose.ENTRY.value)),
            position_id=(str(raw["position_id"]) if raw.get("position_id") else None),
            parent_order_id=(
                str(raw["parent_order_id"]) if raw.get("parent_order_id") else None
            ),
            client_ref=str(raw.get("client_ref") or ""),
            deal_id=str(raw.get("deal_id") or ""),
            active_from_bar=int(raw.get("active_from_bar") or 0),
        )


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
    deal_id: str = ""  # IG dealId when imported / mirrored

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "side": self.side.value,
            "size": self.size,
            "entry": self.entry,
            "role": self.role.value,
            "opened_bar": self.opened_bar,
            "opened_at": self.opened_at,
            "upl": self.upl,
            "deal_id": self.deal_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PositionLeg:
        raw = dict(data or {})
        return cls(
            id=str(raw.get("id") or ""),
            side=Side(str(raw.get("side") or Side.BUY.value)),
            size=float(raw.get("size") or 0.0),
            entry=float(raw.get("entry") or 0.0),
            role=LegRole(str(raw.get("role") or LegRole.PRIMARY.value)),
            opened_bar=int(raw.get("opened_bar") or 0),
            opened_at=str(raw.get("opened_at") or ""),
            upl=float(raw.get("upl") or 0.0),
            deal_id=str(raw.get("deal_id") or ""),
        )


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
    deal_id: str = ""  # IG dealId of the closed leg (live)
    phantom: bool = False  # local close not confirmed on IG / later reopened
    ig_confirmed: bool = False  # True when closed because deal vanished from IG

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "side": self.side.value,
            "size": self.size,
            "entry": self.entry,
            "exit": self.exit,
            "role": self.role.value,
            "realized_pnl": self.realized_pnl,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "bars_held": self.bars_held,
            "deal_id": self.deal_id,
            "phantom": self.phantom,
            "ig_confirmed": self.ig_confirmed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ClosedTrade:
        raw = dict(data or {})
        return cls(
            id=str(raw.get("id") or ""),
            side=Side(str(raw.get("side") or Side.BUY.value)),
            size=float(raw.get("size") or 0.0),
            entry=float(raw.get("entry") or 0.0),
            exit=float(raw.get("exit") or 0.0),
            role=LegRole(str(raw.get("role") or LegRole.PRIMARY.value)),
            realized_pnl=float(raw.get("realized_pnl") or 0.0),
            opened_at=str(raw.get("opened_at") or ""),
            closed_at=str(raw.get("closed_at") or ""),
            bars_held=int(raw.get("bars_held") or 0),
            deal_id=str(raw.get("deal_id") or ""),
            phantom=bool(raw.get("phantom") or False),
            ig_confirmed=bool(raw.get("ig_confirmed") or False),
        )


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

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LlmAnalysis:
        data = data or {}
        return cls(
            support=float(data["support"]) if data.get("support") is not None else None,
            resistance=(
                float(data["resistance"]) if data.get("resistance") is not None else None
            ),
            bias=str(data.get("bias") or "hold"),
            rsi_note=str(data.get("rsi_note") or ""),
            pivot_note=str(data.get("pivot_note") or ""),
        )

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

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> LlmDecision:
        data = data if isinstance(data, dict) else {}
        actions_raw = data.get("actions") or []
        actions = [
            LlmAction.from_dict(a) for a in actions_raw if isinstance(a, dict)
        ]
        return cls(
            analysis=LlmAnalysis.from_dict(
                data.get("analysis") if isinstance(data.get("analysis"), dict) else {}
            ),
            actions=actions,
            raw=dict(data),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis": self.analysis.to_dict(),
            "actions": [a.to_dict() for a in self.actions],
        }
