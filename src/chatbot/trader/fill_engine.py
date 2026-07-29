from __future__ import annotations

from dataclasses import dataclass

from chatbot.trader.config import TraderConfig
from chatbot.trader.models import OrderType, Side, WorkingOrder


@dataclass(frozen=True)
class FillResult:
    order_id: str
    fill_price: float
    reason: str


def _half_spread(cfg: TraderConfig) -> float:
    return abs(cfg.spread_points) / 2.0


def evaluate_limit_fill(order: WorkingOrder, bar: dict, cfg: TraderConfig) -> FillResult | None:
    """Limit fill if bar touches level. Fill at level +/- half spread (never better)."""
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    level = order.level
    half = _half_spread(cfg)

    if order.side == Side.BUY:
        if l <= level:
            # Gap through: open already below level → fill at open (worse)
            if o < level:
                price = o + half
            else:
                price = level + half
            return FillResult(order.id, price, "limit_buy_touch")
        return None

    if h >= level:
        if o > level:
            price = o - half
        else:
            price = level - half
        return FillResult(order.id, price, "limit_sell_touch")
    return None


def evaluate_stop_fill(order: WorkingOrder, bar: dict, cfg: TraderConfig) -> FillResult | None:
    """
    Stop fill on breach; gap open fills at open + slippage.

    - Standard BUY STOP: triggers on rise (high >= level)
    - Standard SELL STOP: triggers on fall (low <= level)
    - HEDGE_COVER breakout entries reverse the conventional side placement:
      SELL hedge_cover above market triggers on rise; BUY hedge_cover below on fall.
    """
    from chatbot.trader.models import OrderPurpose

    o, h, l = bar["open"], bar["high"], bar["low"]
    level = order.level
    slip = abs(cfg.slippage_points)
    breakout = order.purpose == OrderPurpose.HEDGE_COVER

    if order.side == Side.BUY:
        # Buy stop (standard above) OR buy hedge on breakdown (below → still low trigger)
        if breakout:
            if o <= level:
                return FillResult(order.id, o + slip, "stop_buy_hedge_gap")
            if l <= level:
                return FillResult(order.id, level + slip, "stop_buy_hedge_touch")
            return None
        if o >= level:
            return FillResult(order.id, o + slip, "stop_buy_gap_open")
        if h >= level:
            return FillResult(order.id, level + slip, "stop_buy_touch")
        return None

    # SELL stop
    if breakout:
        # Open short on upside breakout
        if o >= level:
            return FillResult(order.id, o - slip, "stop_sell_hedge_gap")
        if h >= level:
            return FillResult(order.id, level - slip, "stop_sell_hedge_touch")
        return None
    if o <= level:
        return FillResult(order.id, o - slip, "stop_sell_gap_open")
    if l <= level:
        return FillResult(order.id, level - slip, "stop_sell_touch")
    return None


def evaluate_order_fill(order: WorkingOrder, bar: dict, cfg: TraderConfig) -> FillResult | None:
    if order.type == OrderType.LIMIT:
        return evaluate_limit_fill(order, bar, cfg)
    if order.type == OrderType.STOP:
        return evaluate_stop_fill(order, bar, cfg)
    if order.type == OrderType.MARKET:
        mid = (bar["open"] + bar["close"]) / 2.0
        half = _half_spread(cfg)
        price = mid + half if order.side == Side.BUY else mid - half
        return FillResult(order.id, price, "market")
    return None


def resolve_intrabar_conflict(
    fills: list[tuple[WorkingOrder, FillResult]],
    *,
    pessimistic: bool = True,
) -> list[tuple[WorkingOrder, FillResult]]:
    """
    If TP (limit close) and protective stop both fill in same bar for same position,
    keep stop first when pessimistic=True.
    """
    by_pos: dict[str, list[tuple[WorkingOrder, FillResult]]] = {}
    others: list[tuple[WorkingOrder, FillResult]] = []
    for order, fill in fills:
        if order.position_id:
            by_pos.setdefault(order.position_id, []).append((order, fill))
        else:
            others.append((order, fill))

    resolved: list[tuple[WorkingOrder, FillResult]] = list(others)
    for _pid, group in by_pos.items():
        if len(group) == 1:
            resolved.extend(group)
            continue
        stops = [g for g in group if g[0].type == OrderType.STOP]
        limits = [g for g in group if g[0].type == OrderType.LIMIT]
        if pessimistic and stops:
            resolved.append(stops[0])
        elif limits:
            resolved.append(limits[0])
        else:
            resolved.append(group[0])
    return resolved
