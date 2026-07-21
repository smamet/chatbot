from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chatbot.cac40.config import Cac40Config, LastLevels
from chatbot.cac40.hedge_ledger import HedgeLedger, exit_would_lose
from chatbot.cac40.models import (
    LlmAction,
    LlmDecision,
    OrderPurpose,
    OrderType,
    PositionLeg,
    Side,
    WorkingOrder,
)

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    executed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


class RiskGate:
    """LLM proposes; code authorizes. Prefer continuity over stacking entries."""

    def __init__(self, config: Cac40Config, ledger: HedgeLedger) -> None:
        self.config = config
        self.ledger = ledger

    def apply(self, decision: LlmDecision) -> GateResult:
        result = GateResult()
        analysis = decision.analysis

        if analysis.support is not None and analysis.resistance is not None:
            self.ledger.last_levels = LastLevels(
                support=analysis.support,
                resistance=analysis.resistance,
                source="llm",
            )

        for action in decision.actions:
            try:
                self._apply_one(action, result)
            except Exception as exc:  # pragma: no cover
                result.rejected.append(f"{action.op}:error:{exc}")
                logger.exception("RiskGate action failed: %s", action)
        return result

    def _clamp_size(self, action: LlmAction) -> float:
        return float(self.config.order_size)

    def _has_entry_working(self, side: Side) -> bool:
        for order in self.ledger.working_orders.values():
            if order.purpose == OrderPurpose.ENTRY and order.side == side:
                return True
        return False

    def _has_hedge_for(self, position_id: str | None) -> bool:
        if not position_id:
            return False
        for order in self.ledger.working_orders.values():
            if order.purpose == OrderPurpose.HEDGE_COVER and order.position_id == position_id:
                return True
        return False

    def _resolve_position_id(self, action: LlmAction) -> str | None:
        if action.position_id:
            return action.position_id
        if len(self.ledger.positions) == 1:
            return next(iter(self.ledger.positions))
        return None

    def _loss_exit_blocked(self, leg: PositionLeg, exit_price: float) -> bool:
        if not self.config.prevent_loss_exits:
            return False
        return exit_would_lose(leg, exit_price, self.config.point_value)

    def _apply_one(self, action: LlmAction, result: GateResult) -> None:
        op = action.op
        purpose = (action.purpose or "").strip().lower()

        if op in ("market_open", "market_close") and not self.config.allow_market_orders:
            result.rejected.append(f"{op}:market_disabled")
            return

        if op == "place_limit":
            if analysis_missing_sr(self.ledger) and purpose in ("entry", "tp"):
                if self.ledger.last_levels.support is None or self.ledger.last_levels.resistance is None:
                    result.rejected.append("place_limit:missing_sr")
                    return
            if not action.side or action.level is None:
                result.rejected.append("place_limit:invalid")
                return
            side = Side(action.side)
            if purpose == "entry":
                if self.ledger.legs_count() >= self.config.max_open_positions:
                    result.rejected.append("place_limit:max_positions")
                    return
                if self._has_entry_working(side):
                    result.rejected.append("place_limit:duplicate_entry")
                    return
            if purpose in ("tp", "close"):
                pid = self._resolve_position_id(action)
                if not pid:
                    result.rejected.append("place_limit:missing_position_id")
                    return
                action.position_id = pid
                leg = self.ledger.positions.get(pid)
                if leg is not None:
                    exit_px = self.ledger.estimate_exit_fill(side, float(action.level), order_type=OrderType.LIMIT)
                    if self._loss_exit_blocked(leg, exit_px):
                        result.rejected.append("place_limit:loss_exit_blocked")
                        return
            if purpose == "hedge_cover":
                if not self.ledger.positions:
                    result.rejected.append("place_limit:hedge_without_primary")
                    return
                pid = self._resolve_position_id(action)
                if pid and self._has_hedge_for(pid):
                    result.rejected.append("place_limit:duplicate_hedge")
                    return
                if pid:
                    action.position_id = pid
            order = WorkingOrder(
                id="",
                type=OrderType.LIMIT,
                side=side,
                level=float(action.level),
                size=self._clamp_size(action),
                purpose=OrderPurpose(purpose or "entry"),
                position_id=action.position_id,
            )
            placed = self.ledger.place_order(order)
            result.executed.append(f"place_limit:{placed.id}@{placed.level}")
            return

        if op == "place_stop":
            if not action.side or action.level is None:
                result.rejected.append("place_stop:invalid")
                return
            side = Side(action.side)
            if purpose == "entry":
                if self.ledger.legs_count() >= self.config.max_open_positions:
                    result.rejected.append("place_stop:max_positions")
                    return
                if self._has_entry_working(side):
                    result.rejected.append("place_stop:duplicate_entry")
                    return
            if purpose in ("tp", "close"):
                pid = self._resolve_position_id(action)
                if not pid:
                    result.rejected.append("place_stop:missing_position_id")
                    return
                action.position_id = pid
                leg = self.ledger.positions.get(pid)
                if leg is not None:
                    exit_px = self.ledger.estimate_exit_fill(side, float(action.level), order_type=OrderType.STOP)
                    if self._loss_exit_blocked(leg, exit_px):
                        result.rejected.append("place_stop:loss_exit_blocked")
                        return
            if purpose == "hedge_cover" or not purpose:
                if not self.ledger.positions:
                    result.rejected.append("place_stop:hedge_without_primary")
                    return
                pid = self._resolve_position_id(action)
                if pid and self._has_hedge_for(pid):
                    result.rejected.append("place_stop:duplicate_hedge")
                    return
                if pid:
                    action.position_id = pid
                purpose = purpose or "hedge_cover"
            order = WorkingOrder(
                id="",
                type=OrderType.STOP,
                side=side,
                level=float(action.level),
                size=self._clamp_size(action),
                purpose=OrderPurpose(purpose or "hedge_cover"),
                position_id=action.position_id,
            )
            placed = self.ledger.place_order(order)
            result.executed.append(f"place_stop:{placed.id}@{placed.level}")
            return

        if op == "amend_order":
            if not action.order_id or action.level is None:
                result.rejected.append("amend_order:invalid")
                return
            existing = self.ledger.working_orders.get(action.order_id)
            if existing is None:
                result.rejected.append(f"amend_order:unknown:{action.order_id}")
                return
            if existing.purpose in (OrderPurpose.TP, OrderPurpose.CLOSE) and existing.position_id:
                leg = self.ledger.positions.get(existing.position_id)
                if leg is not None:
                    exit_px = self.ledger.estimate_exit_fill(
                        existing.side, float(action.level), order_type=existing.type
                    )
                    if self._loss_exit_blocked(leg, exit_px):
                        result.rejected.append("amend_order:loss_exit_blocked")
                        return
            self.ledger.amend_order(action.order_id, level=float(action.level))
            result.executed.append(f"amend_order:{action.order_id}->{action.level}")
            return

        if op == "cancel_order":
            if not action.order_id:
                result.rejected.append("cancel_order:invalid")
                return
            self.ledger.cancel_order(action.order_id)
            result.executed.append(f"cancel_order:{action.order_id}")
            return

        if op == "market_open":
            if self.ledger.legs_count() >= self.config.max_open_positions:
                result.rejected.append("market_open:max_positions")
                return
            if not action.side:
                result.rejected.append("market_open:invalid")
                return
            pid = self.ledger.market_open(Side(action.side), self._clamp_size(action))
            result.executed.append(f"market_open:{pid}")
            return

        if op == "market_close":
            if not action.position_id:
                result.rejected.append("market_close:invalid")
                return
            leg = self.ledger.positions.get(action.position_id)
            if leg is not None:
                exit_px = self.ledger.market_close_fill_price(leg)
                if self._loss_exit_blocked(leg, exit_px):
                    result.rejected.append("market_close:loss_exit_blocked")
                    return
            self.ledger.market_close(action.position_id)
            result.executed.append(f"market_close:{action.position_id}")
            return

        result.rejected.append(f"{op}:unknown_op")


def analysis_missing_sr(ledger: HedgeLedger) -> bool:
    return ledger.last_levels.support is None or ledger.last_levels.resistance is None
