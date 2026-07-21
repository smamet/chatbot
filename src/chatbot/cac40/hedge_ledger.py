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
    PositionLeg,
    Side,
    WorkingOrder,
)


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

    def amend_order(self, order_id: str, *, level: float) -> WorkingOrder:
        order = self.working_orders[order_id]
        order.level = level
        return order

    def cancel_order(self, order_id: str) -> None:
        self.working_orders.pop(order_id, None)

    def _open_leg(
        self,
        side: Side,
        size: float,
        entry: float,
        role: LegRole,
        *,
        opened_at: str = "",
    ) -> PositionLeg:
        leg = PositionLeg(
            id=self._next_id("p"),
            side=side,
            size=size,
            entry=entry,
            role=role,
            opened_bar=self.bar_index,
            opened_at=opened_at,
        )
        self.positions[leg.id] = leg
        return leg

    def close_position(self, position_id: str, exit_price: float, *, closed_at: str = "") -> ClosedTrade | None:
        leg = self.positions.pop(position_id, None)
        if leg is None:
            return None
        direction = 1.0 if leg.side == Side.BUY else -1.0
        pnl = (exit_price - leg.entry) * direction * leg.size * self.config.point_value
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
        )
        self.closed_trades.append(trade)
        # Cancel linked working orders
        for oid, order in list(self.working_orders.items()):
            if order.position_id == position_id:
                self.working_orders.pop(oid, None)
        return trade

    def market_open(self, side: Side, size: float, *, role: LegRole = LegRole.PRIMARY) -> str:
        price = self.last_price
        half = abs(self.config.spread_points) / 2.0
        fill = price + half if side == Side.BUY else price - half
        leg = self._open_leg(side, size, fill, role)
        return leg.id

    def market_close(self, position_id: str) -> None:
        half = abs(self.config.spread_points) / 2.0
        leg = self.positions.get(position_id)
        if not leg:
            return
        fill = self.last_price - half if leg.side == Side.BUY else self.last_price + half
        self.close_position(position_id, fill)

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

    def process_bar(self, bar: dict, *, ts: str = "") -> list[dict]:
        """Match working orders against OHLC bar; update MTM. Returns fill events."""
        self.bar_index += 1
        self.last_price = float(bar["close"])
        events: list[dict] = []

        candidates: list[tuple[WorkingOrder, object]] = []
        for order in list(self.working_orders.values()):
            if order.active_from_bar > self.bar_index:
                continue
            fill = evaluate_order_fill(order, bar, self.config)
            if fill:
                candidates.append((order, fill))

        for order, fill in resolve_intrabar_conflict(
            candidates, pessimistic=self.config.intrabar_pessimistic
        ):
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
                    role = LegRole.HEDGE if order.side != existing.side else LegRole.HEDGE_COVER
            elif order.purpose == OrderPurpose.ENTRY:
                role = LegRole.PRIMARY

            leg = self._open_leg(order.side, order.size, fill.fill_price, role, opened_at=ts)
            events.append(
                {"type": "open", "order": order.to_dict(), "fill": fill.fill_price, "leg": leg.to_dict()}
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

    def pnl_payload(self) -> dict:
        net = sum(p.upl for p in self.positions.values())
        return {
            "net_upl": net,
            "gross_upl": self.gross_upl(),
            "realized_session": self.realized_session,
            "legs_count": self.legs_count(),
            "working_orders_count": len(self.working_orders),
        }
