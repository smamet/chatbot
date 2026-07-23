from __future__ import annotations

import logging
from dataclasses import dataclass, field

from chatbot.cac40.config import Cac40Config, LastLevels
from chatbot.cac40.hedge_ledger import HedgeLedger, exit_would_lose
from chatbot.cac40.models import (
    LegRole,
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

    def __init__(
        self,
        config: Cac40Config,
        ledger: HedgeLedger,
        *,
        flatten_active: bool = False,
        broker: object | None = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.flatten_active = bool(flatten_active)
        # Live: IgConnector with real market_open/close. Paper/backtest: None → ledger.
        self.broker = broker

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

    def _exposure_with_entries_and_working_hedges(self) -> float:
        """Signed exposure: positions + working ENTRY + working HEDGE_COVER (+BUY / −SELL)."""
        exp = float(self.ledger.net_size())
        for order in self.ledger.working_orders.values():
            if order.purpose not in (OrderPurpose.ENTRY, OrderPurpose.HEDGE_COVER):
                continue
            exp += float(order.size) if order.side == Side.BUY else -float(order.size)
        return exp

    def _hedge_cover_size(self, hedge_side: Side) -> float:
        """Size hedge to cover residual unprotected exposure, not just order_size.

        Example: two BUY legs → SELL hedge_cover size 2. If a 1-lot SELL hedge already
        rests, a further SELL hedge is sized to the residual 1 (not another full 2).
        Filled opposing legs reduce net_size(); working hedges are counted too.
        """
        exp = self._exposure_with_entries_and_working_hedges()
        need = exp if hedge_side == Side.SELL else -exp
        if need > 0:
            return float(need)
        # Bracket hedge before net flips (entry just placed, flat residual): use entry size.
        entry = self._find_working_entry(opposite_of=hedge_side)
        if entry is not None:
            return float(entry.size)
        return float(self.config.order_size)

    def _flatten_hedge_size(self, action: LlmAction) -> float:
        """Use requested size (or |net|) during weekend flatten — not order_size clamp."""
        if action.size is not None and float(action.size) > 0:
            return abs(float(action.size))
        net = abs(float(self.ledger.net_size()))
        return net if net > 0 else float(self.config.order_size)

    def _size_for_place(self, action: LlmAction, *, side: Side, purpose: str) -> float:
        if purpose == "hedge_cover":
            return self._hedge_cover_size(side)
        return self._clamp_size(action)

    def _has_entry_working(self, side: Side) -> bool:
        for order in self.ledger.working_orders.values():
            if order.purpose == OrderPurpose.ENTRY and order.side == side:
                return True
        return False

    def _same_level_primary(self, side: Side, level: float) -> bool:
        """True if an open same-side leg sits within llm_level_band_points of level."""
        band = abs(float(self.config.llm_level_band_points or 15.0))
        for leg in self.ledger.positions.values():
            if leg.side != side:
                continue
            if abs(float(leg.entry) - float(level)) <= band:
                return True
        return False

    def _unprotected_open_size(self, side: Side) -> float:
        """Open lots on ``side`` not covered by opposing filled legs or working hedges."""
        open_sz = sum(
            float(leg.size)
            for leg in self.ledger.positions.values()
            if leg.side == side
        )
        if open_sz <= 0:
            return 0.0
        hedge_side = Side.SELL if side == Side.BUY else Side.BUY
        covered = sum(
            float(leg.size)
            for leg in self.ledger.positions.values()
            if leg.side == hedge_side
        )
        covered += sum(
            float(order.size)
            for order in self.ledger.working_orders.values()
            if order.purpose == OrderPurpose.HEDGE_COVER and order.side == hedge_side
        )
        return max(0.0, open_sz - covered)

    def _has_unprotected_open_book(self) -> bool:
        """True if any open BUY or SELL legs lack full hedge cover."""
        return (
            self._unprotected_open_size(Side.BUY) > 1e-9
            or self._unprotected_open_size(Side.SELL) > 1e-9
        )

    def _find_working_entry(self, *, opposite_of: Side | None = None) -> WorkingOrder | None:
        """Latest ENTRY working order, optionally opposite the child side (TP/hedge)."""
        entries = [
            o
            for o in self.ledger.working_orders.values()
            if o.purpose == OrderPurpose.ENTRY
        ]
        if not entries:
            return None
        if opposite_of is not None:
            opp = [o for o in entries if o.side != opposite_of]
            if opp:
                return opp[-1]
            return None
        return entries[-1]

    def _has_hedge_for(self, position_id: str | None) -> bool:
        if not position_id:
            return False
        for order in self.ledger.working_orders.values():
            if order.purpose == OrderPurpose.HEDGE_COVER and order.position_id == position_id:
                return True
        return False

    def _has_hedge_for_parent(self, parent_order_id: str | None) -> bool:
        if not parent_order_id:
            return False
        for order in self.ledger.working_orders.values():
            if (
                order.purpose == OrderPurpose.HEDGE_COVER
                and order.parent_order_id == parent_order_id
            ):
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

    @staticmethod
    def _hedge_beyond_entry(hedge_side: Side, hedge_level: float, entry: WorkingOrder) -> bool:
        """BUY stop must sit at/above entry; SELL stop at/below — price crosses entry first."""
        if hedge_side == Side.BUY:
            return float(hedge_level) >= float(entry.level)
        return float(hedge_level) <= float(entry.level)

    def _apply_one(self, action: LlmAction, result: GateResult) -> None:
        op = action.op
        purpose = (action.purpose or "").strip().lower()

        if op in ("market_open", "market_close") and not self.config.allow_market_orders:
            # Exceptions when market orders are off:
            # - weekend/holiday flatten may market-hedge
            # - profitable market_close (hedge→new S/R rotation)
            purpose_l = (action.purpose or "").strip().lower()
            flatten_ok = (
                self.flatten_active
                and op == "market_open"
                and purpose_l in ("hedge_cover", "hedge", "")
            )
            profit_close_ok = False
            if op == "market_close" and action.position_id:
                leg = self.ledger.positions.get(action.position_id)
                if leg is not None:
                    exit_px = self.ledger.market_close_fill_price(leg)
                    profit_close_ok = not exit_would_lose(
                        leg, exit_px, self.config.point_value
                    )
            if not (flatten_ok or profit_close_ok):
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
            parent_order_id: str | None = None
            if purpose == "entry":
                if self.ledger.legs_count() >= self.config.max_open_positions:
                    result.rejected.append("place_limit:max_positions")
                    return
                if self._has_entry_working(side):
                    result.rejected.append("place_limit:duplicate_entry")
                    return
                if self._same_level_primary(side, float(action.level)):
                    result.rejected.append("place_limit:same_level_primary")
                    return
                # Do not add new directional risk while existing legs are naked.
                if self.ledger.positions and self._has_unprotected_open_book():
                    result.rejected.append("place_limit:unhedged_open_book")
                    return
            if purpose in ("tp", "close"):
                # Prefer explicit position_id; else bracket a working entry; else auto-link
                # a single open leg. Do NOT auto-link a leg when a new entry is waiting —
                # that would steal the TP from the bracket.
                if action.position_id:
                    pid = action.position_id
                    action.position_id = pid
                    leg = self.ledger.positions.get(pid)
                    if leg is not None:
                        exit_px = self.ledger.estimate_exit_fill(
                            side, float(action.level), order_type=OrderType.LIMIT
                        )
                        if self._loss_exit_blocked(leg, exit_px):
                            result.rejected.append("place_limit:loss_exit_blocked")
                            return
                else:
                    entry = self._find_working_entry(opposite_of=side)
                    if entry is not None:
                        synth = PositionLeg(
                            id="__pending__",
                            side=entry.side,
                            size=entry.size,
                            entry=float(entry.level),
                            role=LegRole.PRIMARY,
                        )
                        exit_px = self.ledger.estimate_exit_fill(
                            side, float(action.level), order_type=OrderType.LIMIT
                        )
                        if self._loss_exit_blocked(synth, exit_px):
                            result.rejected.append("place_limit:loss_exit_blocked")
                            return
                        parent_order_id = entry.id
                    else:
                        pid = self._resolve_position_id(action)
                        if not pid:
                            result.rejected.append("place_limit:missing_position_id")
                            return
                        action.position_id = pid
                        leg = self.ledger.positions.get(pid)
                        if leg is not None:
                            exit_px = self.ledger.estimate_exit_fill(
                                side, float(action.level), order_type=OrderType.LIMIT
                            )
                            if self._loss_exit_blocked(leg, exit_px):
                                result.rejected.append("place_limit:loss_exit_blocked")
                                return
            if purpose == "hedge_cover":
                if action.position_id:
                    pid = action.position_id
                    if self._has_hedge_for(pid):
                        result.rejected.append("place_limit:duplicate_hedge")
                        return
                    action.position_id = pid
                else:
                    entry = self._find_working_entry(opposite_of=side)
                    if entry is not None:
                        if not self._hedge_beyond_entry(side, float(action.level), entry):
                            result.rejected.append("place_limit:hedge_not_beyond_entry")
                            return
                        if self._has_hedge_for_parent(entry.id):
                            result.rejected.append("place_limit:duplicate_hedge")
                            return
                        parent_order_id = entry.id
                    elif not self.ledger.positions:
                        result.rejected.append("place_limit:hedge_without_primary")
                        return
                    else:
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
                size=self._size_for_place(action, side=side, purpose=purpose or "entry"),
                purpose=OrderPurpose(purpose or "entry"),
                position_id=action.position_id,
                parent_order_id=parent_order_id,
            )
            placed = self.ledger.place_order(order)
            result.executed.append(f"place_limit:{placed.id}@{placed.level}")
            return

        if op == "place_stop":
            if not action.side or action.level is None:
                result.rejected.append("place_stop:invalid")
                return
            side = Side(action.side)
            parent_order_id = None
            if purpose == "entry":
                if self.ledger.legs_count() >= self.config.max_open_positions:
                    result.rejected.append("place_stop:max_positions")
                    return
                if self._has_entry_working(side):
                    result.rejected.append("place_stop:duplicate_entry")
                    return
                if self._same_level_primary(side, float(action.level)):
                    result.rejected.append("place_stop:same_level_primary")
                    return
                if self.ledger.positions and self._has_unprotected_open_book():
                    result.rejected.append("place_stop:unhedged_open_book")
                    return
            if purpose in ("tp", "close"):
                if action.position_id:
                    pid = action.position_id
                    action.position_id = pid
                    leg = self.ledger.positions.get(pid)
                    if leg is not None:
                        exit_px = self.ledger.estimate_exit_fill(
                            side, float(action.level), order_type=OrderType.STOP
                        )
                        if self._loss_exit_blocked(leg, exit_px):
                            result.rejected.append("place_stop:loss_exit_blocked")
                            return
                else:
                    entry = self._find_working_entry(opposite_of=side)
                    if entry is not None:
                        synth = PositionLeg(
                            id="__pending__",
                            side=entry.side,
                            size=entry.size,
                            entry=float(entry.level),
                            role=LegRole.PRIMARY,
                        )
                        exit_px = self.ledger.estimate_exit_fill(
                            side, float(action.level), order_type=OrderType.STOP
                        )
                        if self._loss_exit_blocked(synth, exit_px):
                            result.rejected.append("place_stop:loss_exit_blocked")
                            return
                        parent_order_id = entry.id
                    else:
                        pid = self._resolve_position_id(action)
                        if not pid:
                            result.rejected.append("place_stop:missing_position_id")
                            return
                        action.position_id = pid
                        leg = self.ledger.positions.get(pid)
                        if leg is not None:
                            exit_px = self.ledger.estimate_exit_fill(
                                side, float(action.level), order_type=OrderType.STOP
                            )
                            if self._loss_exit_blocked(leg, exit_px):
                                result.rejected.append("place_stop:loss_exit_blocked")
                                return
            if purpose == "hedge_cover" or not purpose:
                purpose = purpose or "hedge_cover"
                if action.position_id:
                    pid = action.position_id
                    if self._has_hedge_for(pid):
                        result.rejected.append("place_stop:duplicate_hedge")
                        return
                    action.position_id = pid
                else:
                    entry = self._find_working_entry(opposite_of=side)
                    if entry is not None:
                        if not self._hedge_beyond_entry(side, float(action.level), entry):
                            result.rejected.append("place_stop:hedge_not_beyond_entry")
                            return
                        if self._has_hedge_for_parent(entry.id):
                            result.rejected.append("place_stop:duplicate_hedge")
                            return
                        parent_order_id = entry.id
                    elif not self.ledger.positions:
                        result.rejected.append("place_stop:hedge_without_primary")
                        return
                    else:
                        pid = self._resolve_position_id(action)
                        if pid and self._has_hedge_for(pid):
                            result.rejected.append("place_stop:duplicate_hedge")
                            return
                        if pid:
                            action.position_id = pid
            order = WorkingOrder(
                id="",
                type=OrderType.STOP,
                side=side,
                level=float(action.level),
                size=self._size_for_place(
                    action, side=side, purpose=purpose or "hedge_cover"
                ),
                purpose=OrderPurpose(purpose or "hedge_cover"),
                position_id=action.position_id,
                parent_order_id=parent_order_id,
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
            if not action.side:
                result.rejected.append("market_open:invalid")
                return
            purpose_l = (action.purpose or "").strip().lower()
            is_flatten_hedge = self.flatten_active and purpose_l in (
                "hedge_cover",
                "hedge",
                "",
            )
            if (
                not is_flatten_hedge
                and self.ledger.legs_count() >= self.config.max_open_positions
            ):
                result.rejected.append("market_open:max_positions")
                return
            if is_flatten_hedge:
                size = self._flatten_hedge_size(action)
                role = LegRole.HEDGE
                side = Side(action.side)
            else:
                size = self._clamp_size(action)
                role = LegRole.PRIMARY
                side = Side(action.side)
            try:
                if self.broker is not None:
                    pid = self.broker.market_open(side, size, role=role)
                else:
                    pid = self.ledger.market_open(side, size, role=role)
            except Exception as exc:
                logger.exception("market_open failed")
                result.rejected.append(f"market_open:broker:{exc}")
                return
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
            try:
                if self.broker is not None:
                    self.broker.market_close(action.position_id)
                else:
                    self.ledger.market_close(action.position_id)
            except Exception as exc:
                logger.exception("market_close failed")
                result.rejected.append(f"market_close:broker:{exc}")
                return
            result.executed.append(f"market_close:{action.position_id}")
            return

        result.rejected.append(f"{op}:unknown_op")


def analysis_missing_sr(ledger: HedgeLedger) -> bool:
    return ledger.last_levels.support is None or ledger.last_levels.resistance is None
