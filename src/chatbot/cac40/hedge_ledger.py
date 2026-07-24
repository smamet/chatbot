from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from chatbot.cac40.config import Cac40Config, LastLevels
from chatbot.cac40.fill_engine import evaluate_order_fill, resolve_intrabar_conflict
from chatbot.cac40.models import (
    ClosedTrade,
    LegRole,
    MarketSnapshot,
    OrderPurpose,
    OrderType,
    PositionLeg,
    Side,
    WorkingOrder,
)


def realized_exit_pnl(leg: PositionLeg, exit_price: float, point_value: float = 1.0) -> float:
    """Same economics as HedgeLedger.close_position."""
    direction = 1.0 if leg.side == Side.BUY else -1.0
    return (exit_price - leg.entry) * direction * leg.size * point_value


def exit_would_lose(leg: PositionLeg, exit_price: float, point_value: float = 1.0) -> bool:
    return realized_exit_pnl(leg, exit_price, point_value) <= 0


@dataclass
class HedgeLedger:
    """Individual hedge-mode position book. Never nets opposing legs."""

    config: Cac40Config
    symbol: str = "CAC40"
    cash: float = 0.0
    realized_session: float = 0.0
    bar_index: int = 0
    last_price: float = 0.0
    last_levels: LastLevels = field(default_factory=LastLevels)
    phase: str = "Flat"
    positions: dict[str, PositionLeg] = field(default_factory=dict)
    working_orders: dict[str, WorkingOrder] = field(default_factory=dict)
    closed_trades: list[ClosedTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    _id_seq: itertools.count = field(default_factory=lambda: itertools.count(1))

    def _next_id(self, prefix: str) -> str:
        return f"{prefix}{next(self._id_seq)}"

    def legs_count(self) -> int:
        return len(self.positions)

    def net_size(self) -> float:
        """Signed exposure: +BUY size, −SELL size (0 = directionally flat)."""
        net = 0.0
        for leg in self.positions.values():
            net += float(leg.size) if leg.side == Side.BUY else -float(leg.size)
        return net

    def entry_order_ids(self) -> list[str]:
        return [
            oid
            for oid, order in self.working_orders.items()
            if order.purpose == OrderPurpose.ENTRY
        ]

    def mark_to_market(self, last_price: float) -> float:
        self.last_price = last_price
        net = 0.0
        pv = self.config.point_value
        for leg in self.positions.values():
            direction = 1.0 if leg.side == Side.BUY else -1.0
            leg.upl = (last_price - leg.entry) * direction * leg.size * pv
            net += leg.upl
        return net

    def gross_upl(self) -> float:
        return sum(abs(p.upl) for p in self.positions.values())

    def infer_phase(self) -> str:
        if not self.positions:
            self.phase = "Flat"
            return self.phase
        sides = {p.side for p in self.positions.values()}
        if len(sides) > 1:
            self.phase = "Covered"
        elif Side.BUY in sides:
            self.phase = "LongAtSupport"
        else:
            self.phase = "ShortAtResistance"
        return self.phase

    def position_summary(self) -> str:
        if not self.positions:
            return "-"
        sides = {p.side for p in self.positions.values()}
        if len(sides) > 1:
            return "hedged"
        return "long" if Side.BUY in sides else "short"

    def get_snapshot(self) -> MarketSnapshot:
        net = self.mark_to_market(self.last_price) if self.last_price else 0.0
        self.infer_phase()
        return MarketSnapshot(
            symbol=self.symbol,
            last_price=self.last_price,
            positions=list(self.positions.values()),
            working_orders=list(self.working_orders.values()),
            account_upl=net,
            last_levels=self.last_levels.to_dict(),
            phase=self.phase,
            bar_index=self.bar_index,
        )

    def place_order(self, order: WorkingOrder) -> WorkingOrder:
        if not order.id:
            order.id = self._next_id("o")
        order.active_from_bar = self.bar_index + 1  # causal: next bar
        self.working_orders[order.id] = order
        return order

    def amend_order(
        self, order_id: str, *, level: float, size: float | None = None
    ) -> WorkingOrder:
        order = self.working_orders[order_id]
        order.level = level
        if size is not None and float(size) > 0:
            order.size = float(size)
        return order

    def cancel_order(self, order_id: str) -> None:
        self.working_orders.pop(order_id, None)
        # Cascade-cancel dormant bracket children linked to this entry.
        for oid, order in list(self.working_orders.items()):
            if order.parent_order_id == order_id:
                self.working_orders.pop(oid, None)

    def _is_dormant_child(self, order: WorkingOrder) -> bool:
        """True while parent entry is still working (not yet filled/cancelled)."""
        if not order.parent_order_id:
            return False
        return order.parent_order_id in self.working_orders

    def _arm_children(self, parent_id: str, leg_id: str) -> list[WorkingOrder]:
        armed: list[WorkingOrder] = []
        for order in self.working_orders.values():
            if order.parent_order_id == parent_id:
                order.position_id = leg_id
                armed.append(order)
        return armed

    def _open_leg(
        self,
        side: Side,
        size: float,
        entry: float,
        role: LegRole,
        *,
        opened_at: str = "",
        deal_id: str = "",
    ) -> PositionLeg:
        leg = PositionLeg(
            id=self._next_id("p"),
            side=side,
            size=size,
            entry=entry,
            role=role,
            opened_bar=self.bar_index,
            opened_at=opened_at,
            deal_id=(deal_id or "").strip(),
        )
        self.positions[leg.id] = leg
        return leg

    def close_position(
        self,
        position_id: str,
        exit_price: float,
        *,
        closed_at: str = "",
        ig_confirmed: bool = False,
    ) -> ClosedTrade | None:
        leg = self.positions.pop(position_id, None)
        if leg is None:
            return None
        pnl = realized_exit_pnl(leg, exit_price, self.config.point_value)
        self.realized_session += pnl
        self.cash += pnl
        trade = ClosedTrade(
            id=leg.id,
            side=leg.side,
            size=leg.size,
            entry=leg.entry,
            exit=exit_price,
            role=leg.role,
            realized_pnl=pnl,
            opened_at=leg.opened_at,
            closed_at=closed_at,
            bars_held=max(0, self.bar_index - leg.opened_bar),
            deal_id=(leg.deal_id or "").strip(),
            ig_confirmed=bool(ig_confirmed),
        )
        self.closed_trades.append(trade)
        # Cancel linked working orders
        for oid, order in list(self.working_orders.items()):
            if order.position_id == position_id:
                self.working_orders.pop(oid, None)
        return trade

    def quarantine_phantom_closes(self, deal_ids: set[str]) -> list[str]:
        """
        Mark closed trades as phantom when their IG dealId is open again.

        Reverses cash / realized_session for newly quarantined rows.
        """
        flagged: list[str] = []
        want = {d.strip() for d in deal_ids if d and str(d).strip()}
        if not want:
            return flagged
        for trade in self.closed_trades:
            did = (trade.deal_id or "").strip()
            if not did or did not in want or trade.phantom:
                continue
            trade.phantom = True
            self.realized_session -= trade.realized_pnl
            self.cash -= trade.realized_pnl
            flagged.append(trade.id)
        return flagged

    def trusted_closed_trades(self) -> list[ClosedTrade]:
        return [t for t in self.closed_trades if not t.phantom]

    def market_close_fill_price(self, leg: PositionLeg) -> float:
        half = abs(self.config.spread_points) / 2.0
        return self.last_price - half if leg.side == Side.BUY else self.last_price + half

    def estimate_exit_fill(self, close_side: Side, level: float, *, order_type: OrderType = OrderType.LIMIT) -> float:
        """Optimistic touch fill at level (same convention as fill_engine)."""
        if order_type == OrderType.STOP:
            slip = abs(self.config.slippage_points)
            return level + slip if close_side == Side.BUY else level - slip
        half = abs(self.config.spread_points) / 2.0
        return level + half if close_side == Side.BUY else level - half

    def market_open(self, side: Side, size: float, *, role: LegRole = LegRole.PRIMARY) -> str:
        price = self.last_price
        half = abs(self.config.spread_points) / 2.0
        fill = price + half if side == Side.BUY else price - half
        leg = self._open_leg(side, size, fill, role)
        return leg.id

    def market_close(self, position_id: str) -> None:
        leg = self.positions.get(position_id)
        if not leg:
            return
        self.close_position(position_id, self.market_close_fill_price(leg))

    def apply_overnight_funding(self) -> float:
        if not self.positions or not self.last_price:
            return 0.0
        charge = 0.0
        rate = self.config.overnight_funding_rate
        for leg in self.positions.values():
            notional = abs(self.last_price * leg.size * self.config.point_value)
            fee = notional * rate
            charge += fee
            self.cash -= fee
            self.realized_session -= fee
        return charge

    def process_bar(
        self, bar: dict, *, ts: str = "", apply_fills: bool = True
    ) -> list[dict]:
        """
        Advance bar / MTM. When ``apply_fills`` (paper/backtest), match working
        orders against OHLC. Live must pass ``apply_fills=False`` and rely on IG.
        """
        self.bar_index += 1
        self.last_price = float(bar["close"])
        events: list[dict] = []

        if apply_fills:
            candidates: list[tuple[WorkingOrder, object]] = []
            for order in list(self.working_orders.values()):
                if order.active_from_bar > self.bar_index:
                    continue
                if self._is_dormant_child(order):
                    continue
                fill = evaluate_order_fill(order, bar, self.config)
                if fill:
                    candidates.append((order, fill))

            events.extend(
                self._apply_fills(
                    candidates,
                    bar=bar,
                    ts=ts,
                    allow_arm_children=True,
                )
            )

        net = self.mark_to_market(self.last_price)
        self.infer_phase()
        self.equity_curve.append(
            {
                "bar": self.bar_index,
                "ts": ts,
                "price": self.last_price,
                "net_upl": net,
                "realized": self.realized_session,
                "equity": self.cash + net,
                "legs": self.legs_count(),
            }
        )
        return events

    def _apply_fills(
        self,
        candidates: list[tuple[WorkingOrder, object]],
        *,
        bar: dict,
        ts: str,
        allow_arm_children: bool,
    ) -> list[dict]:
        events: list[dict] = []
        for order, fill in resolve_intrabar_conflict(
            candidates, pessimistic=self.config.intrabar_pessimistic
        ):
            if order.id not in self.working_orders:
                continue
            self.working_orders.pop(order.id, None)

            # TP/CLOSE without position_id must never open a new leg.
            if order.purpose in (OrderPurpose.TP, OrderPurpose.CLOSE):
                if not order.position_id:
                    events.append(
                        {
                            "type": "rejected_fill",
                            "reason": "tp_missing_position_id",
                            "order": order.to_dict(),
                            "fill": fill.fill_price,
                        }
                    )
                    continue
                leg = self.positions.get(order.position_id)
                if (
                    self.config.prevent_loss_exits
                    and leg is not None
                    and exit_would_lose(leg, fill.fill_price, self.config.point_value)
                ):
                    events.append(
                        {
                            "type": "rejected_fill",
                            "reason": "loss_exit_blocked",
                            "order": order.to_dict(),
                            "fill": fill.fill_price,
                        }
                    )
                    continue
                trade = self.close_position(order.position_id, fill.fill_price, closed_at=ts)
                events.append(
                    {
                        "type": "close",
                        "order": order.to_dict(),
                        "fill": fill.fill_price,
                        "trade": trade,
                    }
                )
                continue

            # Hedge cover before any primary would open a rogue opposing leg.
            if order.purpose == OrderPurpose.HEDGE_COVER and not self.positions:
                events.append(
                    {
                        "type": "rejected_fill",
                        "reason": "hedge_without_primary",
                        "order": order.to_dict(),
                        "fill": fill.fill_price,
                    }
                )
                continue

            if self.legs_count() >= self.config.max_open_positions:
                events.append(
                    {
                        "type": "rejected_fill",
                        "reason": "max_positions",
                        "order": order.to_dict(),
                        "fill": fill.fill_price,
                    }
                )
                continue

            role = LegRole.PRIMARY
            if order.purpose == OrderPurpose.HEDGE_COVER:
                role = LegRole.HEDGE_COVER if self.positions else LegRole.PRIMARY
                if self.positions:
                    existing = next(iter(self.positions.values()))
                    role = (
                        LegRole.HEDGE
                        if order.side != existing.side
                        else LegRole.HEDGE_COVER
                    )
            elif order.purpose == OrderPurpose.ENTRY:
                role = LegRole.PRIMARY

            leg = self._open_leg(
                order.side, order.size, fill.fill_price, role, opened_at=ts
            )
            events.append(
                {
                    "type": "open",
                    "order": order.to_dict(),
                    "fill": fill.fill_price,
                    "leg": leg.to_dict(),
                }
            )

            if allow_arm_children and order.purpose == OrderPurpose.ENTRY:
                armed = self._arm_children(order.id, leg.id)
                if armed:
                    child_cands: list[tuple[WorkingOrder, object]] = []
                    for child in armed:
                        # Eligible on this bar even if active_from_bar was next.
                        child.active_from_bar = min(child.active_from_bar, self.bar_index)
                        cfill = evaluate_order_fill(child, bar, self.config)
                        if cfill:
                            child_cands.append((child, cfill))
                    if child_cands:
                        events.extend(
                            self._apply_fills(
                                child_cands,
                                bar=bar,
                                ts=ts,
                                allow_arm_children=False,
                            )
                        )
        return events

    def pnl_payload(self) -> dict:
        net = sum(p.upl for p in self.positions.values())
        trusted = self.trusted_closed_trades()
        return {
            "net_upl": net,
            "gross_upl": self.gross_upl(),
            "realized_session": self.realized_session,
            "legs_count": self.legs_count(),
            "working_orders_count": len(self.working_orders),
            "closed_trades_trusted": len(trusted),
            "closed_trades_phantom": sum(1 for t in self.closed_trades if t.phantom),
        }

    def to_state_dict(self) -> dict:
        """Persistable ledger snapshot for live paper/live continuity."""
        max_id = 0
        for key in list(self.positions) + list(self.working_orders):
            digits = "".join(ch for ch in key if ch.isdigit())
            if digits.isdigit():
                max_id = max(max_id, int(digits))
        for trade in self.closed_trades:
            digits = "".join(ch for ch in trade.id if ch.isdigit())
            if digits.isdigit():
                max_id = max(max_id, int(digits))
        return {
            "symbol": self.symbol,
            "cash": self.cash,
            "realized_session": self.realized_session,
            "bar_index": self.bar_index,
            "last_price": self.last_price,
            "last_levels": self.last_levels.to_dict(),
            "phase": self.phase,
            "positions": [p.to_dict() for p in self.positions.values()],
            "working_orders": [o.to_dict() for o in self.working_orders.values()],
            "closed_trades": [t.to_dict() for t in self.closed_trades],
            "equity_curve": list(self.equity_curve),
            "id_seq": max_id,
        }

    @classmethod
    def from_state_dict(cls, config: Cac40Config, data: dict | None) -> HedgeLedger:
        raw = dict(data or {})
        ledger = cls(config=config, symbol=str(raw.get("symbol") or config.symbol or "CAC40"))
        ledger.cash = float(raw.get("cash") or 0.0)
        ledger.realized_session = float(raw.get("realized_session") or 0.0)
        ledger.bar_index = int(raw.get("bar_index") or 0)
        ledger.last_price = float(raw.get("last_price") or 0.0)
        ledger.last_levels = LastLevels.from_dict(raw.get("last_levels") or {})
        ledger.phase = str(raw.get("phase") or "Flat")
        ledger.positions = {
            p.id: p
            for p in (PositionLeg.from_dict(row) for row in (raw.get("positions") or []))
            if p.id
        }
        ledger.working_orders = {
            o.id: o
            for o in (WorkingOrder.from_dict(row) for row in (raw.get("working_orders") or []))
            if o.id
        }
        ledger.closed_trades = [
            ClosedTrade.from_dict(row) for row in (raw.get("closed_trades") or [])
        ]
        ledger.equity_curve = list(raw.get("equity_curve") or [])
        start = int(raw.get("id_seq") or 0) + 1
        ledger._id_seq = itertools.count(max(1, start))
        return ledger
