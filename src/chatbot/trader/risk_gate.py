from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from chatbot.trader.config import TraderConfig, LastLevels
from chatbot.trader.hedge_ledger import HedgeLedger, exit_would_lose
from chatbot.trader.models import (
    LegRole,
    LlmAction,
    LlmDecision,
    OrderPurpose,
    OrderType,
    PositionLeg,
    Side,
    WorkingOrder,
)
from chatbot.trader.point_size import infer_point_size

logger = logging.getLogger(__name__)


def _hedge_nudge_note(before: float, after: float, point_size: float) -> str:
    """Format nudge note with enough decimals for the instrument."""
    decimals = 5 if point_size < 0.001 else (2 if point_size < 1 else 1)
    fmt = f"{{:.{decimals}f}}"
    return f"hedge_nudged:{fmt.format(before)}->{fmt.format(after)}"


@dataclass
class GateResult:
    executed: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class RiskGate:
    """LLM proposes; code authorizes. Prefer continuity over stacking entries."""

    def __init__(
        self,
        config: TraderConfig,
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

    def _exposure_with_entries_and_working_hedges(
        self, *, exclude_order_id: str | None = None
    ) -> float:
        """Signed exposure: positions + working ENTRY + working HEDGE_COVER (+BUY / −SELL)."""
        exp = float(self.ledger.net_size())
        for order in self.ledger.working_orders.values():
            if exclude_order_id and order.id == exclude_order_id:
                continue
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

    def _amend_size_for(self, existing: WorkingOrder, action: LlmAction) -> float | None:
        """Return new size for amend, or None to leave size unchanged."""
        purpose = (
            (action.purpose or "").strip().lower()
            or (existing.purpose.value if existing.purpose else "")
        )
        if purpose == "hedge_cover":
            # Full cover as if this WO were absent — never residual-after-self.
            exp = self._exposure_with_entries_and_working_hedges(
                exclude_order_id=existing.id
            )
            need = exp if existing.side == Side.SELL else -exp
            if need > 0:
                return float(need)
            return None
        if action.size is None or float(action.size) <= 0:
            return None
        requested = abs(float(action.size))
        if purpose in ("entry", "tp", "close"):
            return min(requested, float(self.config.order_size))
        return requested

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

    def _known_leg_id(self, position_id: str | None) -> str | None:
        """Return position_id only if it names an open leg (ignore LLM placeholders)."""
        pid = str(position_id or "").strip()
        if not pid:
            return None
        return pid if pid in self.ledger.positions else None

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

    def _same_level_hedge(self, side: Side, level: float) -> bool:
        """True if a same-side working/filled hedge already sits at ~the same price.

        Uses ``hedge_beyond_entry_points`` (not the wider entry band) so a residual
        cover a few points away from an existing hedge is still allowed.
        """
        points = max(1.0, abs(float(self.config.hedge_beyond_entry_points or 2.0)))
        pip = infer_point_size(float(self.ledger.last_price or level or 0.0))
        band = points * pip if pip > 0 else points
        lvl = float(level)
        for order in self.ledger.working_orders.values():
            if order.purpose != OrderPurpose.HEDGE_COVER or order.side != side:
                continue
            if abs(float(order.level) - lvl) <= band + 1e-12:
                return True
        for leg in self.ledger.positions.values():
            if leg.side != side or leg.role != LegRole.HEDGE:
                continue
            if abs(float(leg.entry) - lvl) <= band + 1e-12:
                return True
        return False

    def _hedge_cover_place_blocked(
        self,
        *,
        side: Side,
        level: float,
        position_id: str | None,
        parent_order_id: str | None,
    ) -> str | None:
        """Return reject suffix (no op prefix) if hedge_cover must not place.

        Allows unlinked hedges when residual unprotected exposure remains (sized
        via ``_hedge_cover_size``). Blocks same-level stacks and no-residual
        orphans — the 1.1535 triple-STOP gap.
        """
        if self._same_level_hedge(side, level):
            return "same_level_hedge"
        # Brackets may place before the entry fills while open-book residual is flat.
        if parent_order_id or self._find_working_entry(opposite_of=side) is not None:
            return None
        covered_side = Side.SELL if side == Side.BUY else Side.BUY
        if self._unprotected_open_size(covered_side) > 1e-9:
            return None
        if not position_id and not parent_order_id:
            return "orphan_hedge"
        return "exposure_already_covered"

    def _resolve_position_id(self, action: LlmAction) -> str | None:
        known = self._known_leg_id(action.position_id)
        if known:
            return known
        if len(self.ledger.positions) == 1:
            return next(iter(self.ledger.positions))
        return None

    def _loss_exit_blocked(self, leg: PositionLeg, exit_price: float) -> bool:
        if not self.config.prevent_loss_exits:
            return False
        return exit_would_lose(leg, exit_price, self.config.point_value)

    @staticmethod
    def _hedge_wrong_side(hedge_side: Side, hedge_level: float, anchor: float) -> bool:
        """True when the stop would fire without price first crossing the entry/fill."""
        if hedge_side == Side.BUY:
            return float(hedge_level) < float(anchor)
        return float(hedge_level) > float(anchor)

    @staticmethod
    def _hedge_beyond_entry(hedge_side: Side, hedge_level: float, entry: WorkingOrder) -> bool:
        """BUY stop must sit at/above entry; SELL stop at/below — price crosses entry first."""
        return not RiskGate._hedge_wrong_side(hedge_side, hedge_level, float(entry.level))

    def _hedge_step_and_min_dist(self, anchor: float) -> tuple[float, float, float]:
        """Return ``(step, min_dist, pip)`` in price units for hedge clearance."""
        points = max(0.0, float(self.config.hedge_beyond_entry_points or 0.0))
        pip = infer_point_size(float(self.ledger.last_price or anchor or 0.0))
        min_dist = points * pip
        step = 0.0
        broker = self.broker
        if broker is not None:
            try:
                resolve_step = getattr(broker, "resolve_price_step", None)
                if callable(resolve_step):
                    step = float(resolve_step() or 0.0)
            except Exception:
                step = 0.0
            try:
                resolve_min = getattr(broker, "resolve_min_stop_or_limit_distance", None)
                if callable(resolve_min):
                    rules_min = float(resolve_min() or 0.0)
                    if rules_min > 0:
                        min_dist = max(min_dist, rules_min)
            except Exception:
                pass
        # FX without broker: pipette grid (matches IgConnector.resolve_price_step).
        if step <= 0 and pip > 0 and pip <= 0.0001 + 1e-15:
            step = pip / 10.0
        return step, min_dist, pip

    @staticmethod
    def _hedge_target_beyond(
        hedge_side: Side, anchor: float, *, min_dist: float, step: float
    ) -> float:
        anchor_f = float(anchor)
        raw = anchor_f + float(min_dist) if hedge_side == Side.BUY else anchor_f - float(min_dist)
        if step <= 0:
            return raw
        # Snap onto the price grid *away* from the entry (never back onto it).
        n = max(1, int(math.ceil((float(min_dist) - 1e-15) / step)))
        delta = n * step
        if hedge_side == Side.BUY:
            return anchor_f + delta
        return anchor_f - delta

    def _ensure_hedge_clearance(
        self, hedge_side: Side, hedge_level: float, anchor: float
    ) -> tuple[float | None, str | None]:
        """
        Keep hedge on the correct side of ``anchor`` by at least
        ``hedge_beyond_entry_points`` (and IG min stop when known).

        Returns ``(level, note)``. ``level`` is None when the hedge is on the
        wrong side (caller should reject). Same-level / too-close levels are
        nudged outward — never rejected for distance alone.
        """
        if self._hedge_wrong_side(hedge_side, hedge_level, anchor):
            return None, None
        points = max(0.0, float(self.config.hedge_beyond_entry_points or 0.0))
        lvl = float(hedge_level)
        anchor_f = float(anchor)
        if points <= 0:
            return lvl, None
        step, min_dist, pip = self._hedge_step_and_min_dist(anchor_f)
        target = self._hedge_target_beyond(
            hedge_side, anchor_f, min_dist=min_dist, step=step
        )
        if hedge_side == Side.BUY:
            if lvl + 1e-12 < target:
                return target, _hedge_nudge_note(lvl, target, pip)
            return lvl, None
        if lvl > target + 1e-12:
            return target, _hedge_nudge_note(lvl, target, pip)
        return lvl, None

    def _apply_hedge_clearance(
        self,
        *,
        side: Side,
        level: float,
        anchor: float,
        result: GateResult,
        reject_prefix: str,
    ) -> float | None:
        """Return level to place (possibly nudged). None = wrong-side reject.

        Does **not** mutate the LLM action — journal must keep the model output.
        """
        nudged, note = self._ensure_hedge_clearance(side, float(level), anchor)
        if nudged is None:
            result.rejected.append(f"{reject_prefix}:hedge_not_beyond_entry")
            return None
        if note:
            result.notes.append(note)
        return float(nudged)

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
            order_level = float(action.level)
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
                # Prefer explicit *real* position_id; else bracket a working entry;
                # else auto-link a single open leg. Ignore LLM placeholders like
                # "SELL_1.1386_entry" (not in ledger.positions) so TP still gets
                # parent_order_id → IG limitDistance attach.
                # Do NOT auto-link a leg when a new entry is waiting — that would
                # steal the TP from the bracket.
                pid = self._known_leg_id(action.position_id)
                if pid:
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
                    action.position_id = None
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
                pid = self._known_leg_id(action.position_id)
                if pid:
                    if self._has_hedge_for(pid):
                        result.rejected.append("place_limit:duplicate_hedge")
                        return
                    action.position_id = pid
                    leg = self.ledger.positions.get(pid)
                    if leg is not None:
                        cleared = self._apply_hedge_clearance(
                            side=side,
                            level=order_level,
                            anchor=float(leg.entry),
                            result=result,
                            reject_prefix="place_limit",
                        )
                        if cleared is None:
                            return
                        order_level = cleared
                else:
                    action.position_id = None
                    entry = self._find_working_entry(opposite_of=side)
                    if entry is not None:
                        cleared = self._apply_hedge_clearance(
                            side=side,
                            level=order_level,
                            anchor=float(entry.level),
                            result=result,
                            reject_prefix="place_limit",
                        )
                        if cleared is None:
                            return
                        order_level = cleared
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
                            leg = self.ledger.positions.get(pid)
                            if leg is not None:
                                cleared = self._apply_hedge_clearance(
                                    side=side,
                                    level=order_level,
                                    anchor=float(leg.entry),
                                    result=result,
                                    reject_prefix="place_limit",
                                )
                                if cleared is None:
                                    return
                                order_level = cleared
                blocked = self._hedge_cover_place_blocked(
                    side=side,
                    level=order_level,
                    position_id=action.position_id,
                    parent_order_id=parent_order_id,
                )
                if blocked:
                    result.rejected.append(f"place_limit:{blocked}")
                    return
            order = WorkingOrder(
                id="",
                type=OrderType.LIMIT,
                side=side,
                level=order_level,
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
            order_level = float(action.level)
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
                pid = self._known_leg_id(action.position_id)
                if pid:
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
                    action.position_id = None
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
                pid = self._known_leg_id(action.position_id)
                if pid:
                    if self._has_hedge_for(pid):
                        result.rejected.append("place_stop:duplicate_hedge")
                        return
                    action.position_id = pid
                    leg = self.ledger.positions.get(pid)
                    if leg is not None:
                        cleared = self._apply_hedge_clearance(
                            side=side,
                            level=order_level,
                            anchor=float(leg.entry),
                            result=result,
                            reject_prefix="place_stop",
                        )
                        if cleared is None:
                            return
                        order_level = cleared
                else:
                    action.position_id = None
                    entry = self._find_working_entry(opposite_of=side)
                    if entry is not None:
                        cleared = self._apply_hedge_clearance(
                            side=side,
                            level=order_level,
                            anchor=float(entry.level),
                            result=result,
                            reject_prefix="place_stop",
                        )
                        if cleared is None:
                            return
                        order_level = cleared
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
                            leg = self.ledger.positions.get(pid)
                            if leg is not None:
                                cleared = self._apply_hedge_clearance(
                                    side=side,
                                    level=order_level,
                                    anchor=float(leg.entry),
                                    result=result,
                                    reject_prefix="place_stop",
                                )
                                if cleared is None:
                                    return
                                order_level = cleared
                blocked = self._hedge_cover_place_blocked(
                    side=side,
                    level=order_level,
                    position_id=action.position_id,
                    parent_order_id=parent_order_id,
                )
                if blocked:
                    result.rejected.append(f"place_stop:{blocked}")
                    return
            order = WorkingOrder(
                id="",
                type=OrderType.STOP,
                side=side,
                level=order_level,
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
            old_size = float(existing.size)
            new_size = self._amend_size_for(existing, action)
            amended = self.ledger.amend_order(
                action.order_id, level=float(action.level), size=new_size
            )
            if new_size is not None and abs(float(amended.size) - old_size) > 1e-9:
                result.executed.append(
                    f"amend_order:{action.order_id}->{action.level}x{amended.size}"
                )
            else:
                result.executed.append(f"amend_order:{action.order_id}->{action.level}")
            return

        if op == "cancel_order":
            if not action.order_id:
                result.rejected.append("cancel_order:invalid")
                return
            if action.order_id not in self.ledger.working_orders:
                # Stale LLM / replay ids must not look like a successful cancel.
                result.rejected.append(f"cancel_order:unknown:{action.order_id}")
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
