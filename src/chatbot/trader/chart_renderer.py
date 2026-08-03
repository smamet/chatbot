from __future__ import annotations

import io
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
except ImportError:  # pragma: no cover
    plt = None  # type: ignore


# TradingView-style Traditional pivot colors (gold family).
_PIVOT_STYLE: dict[str, dict[str, Any]] = {
    "P": {"color": "#c9a227", "lw": 1.4},
    "R1": {"color": "#d97706", "lw": 1.0},
    "R2": {"color": "#ea580c", "lw": 1.0},
    "R3": {"color": "#dc2626", "lw": 0.9},
    "S1": {"color": "#ca8a04", "lw": 1.0},
    "S2": {"color": "#a16207", "lw": 1.0},
    "S3": {"color": "#854d0e", "lw": 0.9},
}

# Pivot *session* period (TradingView "Traditional, Daily|Weekly|Monthly").
PIVOT_PERIODS: dict[str, str] = {
    "D": "Daily",
    "W": "Weekly",
    "M": "Monthly",
}
_PIVOT_RESAMPLE: dict[str, str] = {
    "D": "1D",
    "W": "W-MON",
    "M": "MS",
}
# Chart timeframes that never draw pivots (daily candles = the pivot session itself).
_PIVOT_SKIP_TF = frozenset({"1d", "1D", "d", "D", "daily", "Daily"})


def normalize_pivot_period(period: str | None) -> str:
    key = (period or "D").strip().upper()
    if key in ("DAY", "DAILY", "1D"):
        return "D"
    if key in ("WEEK", "WEEKLY", "1W"):
        return "W"
    if key in ("MONTH", "MONTHLY", "1M"):
        return "M"
    if key not in PIVOT_PERIODS:
        return "D"
    return key


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, 1e-12)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def traditional_pivots(high: float, low: float, close: float) -> dict[str, float]:
    """TradingView Traditional pivot levels from the previous session H/L/C."""
    h, l, c = float(high), float(low), float(close)
    pp = (h + l + c) / 3.0
    return {
        "P": pp,
        "R1": 2.0 * pp - l,
        "S1": 2.0 * pp - h,
        "R2": pp + (h - l),
        "S2": pp - (h - l),
        "R3": h + 2.0 * (pp - l),
        "S3": l - 2.0 * (h - pp),
    }


def _session_key(ts: pd.Timestamp, pivot_period: str) -> date:
    """Bucket a bar into its pivot session (day / week-start / month-start)."""
    t = pd.Timestamp(ts)
    period = normalize_pivot_period(pivot_period)
    if period == "W":
        # Period conversion drops tz; normalize first to avoid warn + ambiguity.
        naive = t.tz_convert("UTC").tz_localize(None) if t.tzinfo is not None else t
        return naive.to_period("W-MON").start_time.date()
    if period == "M":
        return date(t.year, t.month, 1)
    return t.date()


def pivot_map(df: pd.DataFrame, pivot_period: str = "D") -> dict[date, dict[str, float]]:
    """Map each session start date → Traditional pivots from the prior session."""
    if df.empty:
        return {}
    period = normalize_pivot_period(pivot_period)
    rule = _PIVOT_RESAMPLE[period]
    sessions = (
        df.resample(rule)
        .agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna(how="any")
    )
    out: dict[date, dict[str, float]] = {}
    for i in range(1, len(sessions)):
        prev = sessions.iloc[i - 1]
        sess_ts = sessions.index[i]
        key = _session_key(pd.Timestamp(sess_ts), period)
        out[key] = traditional_pivots(prev["high"], prev["low"], prev["close"])
    return out


def daily_pivot_map(df: pd.DataFrame) -> dict[date, dict[str, float]]:
    """Backward-compatible alias for daily Traditional pivots."""
    return pivot_map(df, "D")


def _format_price_label(price: float) -> str:
    """Adaptive decimals so FX (~1.15) and indices (~7000) both stay readable."""
    p = abs(float(price))
    if p >= 100:
        return f"{price:.1f}"
    if p >= 10:
        return f"{price:.2f}"
    if p >= 1:
        return f"{price:.4f}"
    return f"{price:.5f}"


def min_candle_body_height(data: pd.DataFrame) -> float:
    """Scale-aware doji body — never use a fixed absolute price (0.1 wrecks FX)."""
    if data is None or data.empty:
        return 1e-6
    highs = data["high"].astype(float)
    lows = data["low"].astype(float)
    ranges = (highs - lows).abs()
    positive = ranges[ranges > 0]
    if len(positive):
        ref = float(positive.median())
    else:
        mid = float(data["close"].astype(float).abs().median() or 0.0)
        ref = mid * 1e-4 if mid > 0 else 1e-6
    return max(ref * 0.02, 1e-12)


def candle_body_height(open_: float, high: float, low: float, close: float, *, min_height: float) -> float:
    """Visible body height; zero-range bars become a thin doji at series scale."""
    height = abs(float(close) - float(open_)) or (float(high) - float(low)) * 0.01
    if height <= 0:
        return float(min_height)
    return float(height)


def _draw_pivots(
    ax: Any,
    data: pd.DataFrame,
    pivots_by_session: dict[date, dict[str, float]],
    pivot_period: str,
) -> None:
    if not pivots_by_session or data.empty:
        return
    period = normalize_pivot_period(pivot_period)
    session_ranges: dict[date, list[int]] = {}
    for i, ts in enumerate(data.index):
        key = _session_key(pd.Timestamp(ts), period)
        session_ranges.setdefault(key, []).append(i)

    for sess, idxs in session_ranges.items():
        levels = pivots_by_session.get(sess)
        if not levels or not idxs:
            continue
        x0, x1 = idxs[0], idxs[-1]
        for name, price in levels.items():
            style = _PIVOT_STYLE.get(name, {"color": "#c9a227", "lw": 1.0})
            ax.hlines(
                price,
                x0 - 0.4,
                x1 + 0.4,
                colors=style["color"],
                linewidth=style["lw"],
                alpha=0.85,
            )
            ax.text(
                x1 + 0.45,
                price,
                f"{name} {_format_price_label(float(price))}",
                color=style["color"],
                fontsize=6.5,
                va="center",
                ha="left",
                clip_on=False,
            )


def render_ohlc_chart(
    df: pd.DataFrame,
    *,
    title: str,
    support: float | None = None,
    resistance: float | None = None,
    out_path: Path | None = None,
    rsi_period: int = 14,
    display_bars: int | None = None,
    show_rsi: bool = True,
    show_pivots: bool = True,
    pivot_period: str = "D",
) -> bytes:
    """Render candlestick chart with optional RSI and Traditional pivots.

    Pass extra bars before the visible window so RSI/pivots can warm up; set
    ``display_bars`` to trim candles to the last N rows after computing indicators.
    ``pivot_period`` is D (Daily), W (Weekly), or M (Monthly).
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for chart rendering")
    if df.empty:
        raise ValueError("empty OHLC dataframe")

    data_full = df.copy()
    if not isinstance(data_full.index, pd.DatetimeIndex):
        data_full.index = pd.to_datetime(data_full.index)

    pivot_period = normalize_pivot_period(pivot_period)
    rsi_len = max(2, int(rsi_period or 14))
    rsi_full = _rsi(data_full["close"], period=rsi_len) if show_rsi else None
    pivots = pivot_map(data_full, pivot_period) if show_pivots else {}

    if display_bars is not None and display_bars > 0 and len(data_full) > display_bars:
        data = data_full.iloc[-display_bars:]
        rsi = rsi_full.iloc[-display_bars:] if rsi_full is not None else None
    else:
        data = data_full
        rsi = rsi_full

    if show_rsi:
        fig, (ax, ax_rsi) = plt.subplots(
            2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
        ax_rsi = None

    width = 0.6
    min_body = min_candle_body_height(data)
    for i, (_ts, row) in enumerate(data.iterrows()):
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color = "#16a34a" if c >= o else "#dc2626"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        bottom = min(o, c)
        height = candle_body_height(o, h, l, c, min_height=min_body)
        ax.add_patch(Rectangle((i - width / 2, bottom), width, height, facecolor=color, edgecolor=color))

    if show_pivots:
        _draw_pivots(ax, data, pivots, pivot_period)

    if support is not None:
        ax.axhline(support, color="#2563eb", linestyle="--", linewidth=1, label=f"S {support}")
    if resistance is not None:
        ax.axhline(resistance, color="#ea580c", linestyle="--", linewidth=1, label=f"R {resistance}")

    subtitle = []
    if show_pivots:
        subtitle.append(f"Pivots Traditional {PIVOT_PERIODS[pivot_period]}")
    if show_rsi:
        subtitle.append(f"RSI({rsi_len})")
    ax.set_title(title + (f" — {', '.join(subtitle)}" if subtitle else ""))
    ax.grid(True, alpha=0.3)
    if support is not None or resistance is not None:
        ax.legend(loc="upper left", fontsize=8)

    step = max(1, len(data) // 6)
    ticks = list(range(0, len(data), step))
    tick_labels = [data.index[i].strftime("%m-%d %H:%M") for i in ticks]

    if ax_rsi is not None and rsi is not None:
        xs = list(range(len(data)))
        ax_rsi.plot(xs, rsi.to_numpy(dtype=float), color="#7c3aed", linewidth=1)
        ax_rsi.axhline(70, color="gray", linestyle=":", linewidth=0.8)
        ax_rsi.axhline(30, color="gray", linestyle=":", linewidth=0.8)
        ax_rsi.set_ylabel(f"RSI({rsi_len})")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.set_xticks(ticks)
        ax_rsi.set_xticklabels(tick_labels, rotation=30, ha="right")
    else:
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    png = buf.getvalue()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
    return png


def render_multi_timeframe(
    ohlc: dict[str, pd.DataFrame],
    *,
    last_levels: dict[str, Any] | None = None,
    out_dir: Path | None = None,
    rsi_period: int = 14,
    display_bars: dict[str, int] | int | None = None,
    show_rsi: bool = True,
    show_pivots: bool = True,
    pivot_period: str = "D",
    symbol: str | None = None,
) -> dict[str, bytes]:
    levels = last_levels or {}
    support = levels.get("support")
    resistance = levels.get("resistance")
    pivot_period = normalize_pivot_period(pivot_period)
    label = (symbol or "").strip() or "OHLC"
    images: dict[str, bytes] = {}
    for tf, df in ohlc.items():
        if df is None or df.empty:
            continue
        path = out_dir / f"chart_{tf}.png" if out_dir else None
        if isinstance(display_bars, dict):
            n_display = display_bars.get(tf)
        else:
            n_display = display_bars
        # Never overlay session pivots on the daily chart.
        tf_pivots = bool(show_pivots) and str(tf) not in _PIVOT_SKIP_TF
        images[tf] = render_ohlc_chart(
            df,
            title=f"{label} {tf}",
            support=float(support) if support is not None else None,
            resistance=float(resistance) if resistance is not None else None,
            out_path=path,
            rsi_period=rsi_period,
            display_bars=n_display,
            show_rsi=show_rsi,
            show_pivots=tf_pivots,
            pivot_period=pivot_period,
        )
    return images


def pivot_history_pad(pivot_period: str, *, timeframe: str) -> int:
    """Extra bars to fetch so the prior pivot session exists before the chart window."""
    period = normalize_pivot_period(pivot_period)
    tf = timeframe.lower()
    if tf in ("1d", "d", "daily"):
        return 0  # pivots not drawn on daily
    if period == "W":
        return 500 if tf in ("15m", "15min") else 120
    if period == "M":
        return 2000 if tf in ("15m", "15min") else 500
    # Daily pivots
    return 120 if tf in ("15m", "15min") else 30
