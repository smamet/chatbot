from __future__ import annotations

import io
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


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.astype(float).diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean().replace(0, 1e-12)
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def render_ohlc_chart(
    df: pd.DataFrame,
    *,
    title: str,
    support: float | None = None,
    resistance: float | None = None,
    out_path: Path | None = None,
    rsi_period: int = 14,
    display_bars: int | None = None,
) -> bytes:
    """Render candlestick + RSI subplot to PNG bytes.

    Pass extra bars before the visible window so RSI can warm up; set
    ``display_bars`` to trim candles/RSI to the last N rows after computing RSI.
    """
    if plt is None:
        raise RuntimeError("matplotlib is required for chart rendering")
    if df.empty:
        raise ValueError("empty OHLC dataframe")

    data = df.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
    period = max(2, int(rsi_period or 14))
    rsi_full = _rsi(data["close"], period=period)

    if display_bars is not None and display_bars > 0 and len(data) > display_bars:
        data = data.iloc[-display_bars:]
        rsi = rsi_full.iloc[-display_bars:]
    else:
        rsi = rsi_full

    fig, (ax, ax_rsi) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    width = 0.6
    for i, (_ts, row) in enumerate(data.iterrows()):
        o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
        color = "#16a34a" if c >= o else "#dc2626"
        ax.plot([i, i], [l, h], color=color, linewidth=1)
        bottom = min(o, c)
        height = abs(c - o) or (h - l) * 0.01 or 0.1
        ax.add_patch(Rectangle((i - width / 2, bottom), width, height, facecolor=color, edgecolor=color))

    if support is not None:
        ax.axhline(support, color="#2563eb", linestyle="--", linewidth=1, label=f"S {support}")
    if resistance is not None:
        ax.axhline(resistance, color="#ea580c", linestyle="--", linewidth=1, label=f"R {resistance}")

    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if support is not None or resistance is not None:
        ax.legend(loc="upper left", fontsize=8)

    # Skip NaN warm-up points instead of drawing a fake flat 50 line.
    rsi_vals = rsi.to_numpy(dtype=float)
    xs = list(range(len(data)))
    ax_rsi.plot(xs, rsi_vals, color="#7c3aed", linewidth=1)
    ax_rsi.axhline(70, color="gray", linestyle=":", linewidth=0.8)
    ax_rsi.axhline(30, color="gray", linestyle=":", linewidth=0.8)
    ax_rsi.set_ylabel(f"RSI({period})")
    ax_rsi.set_ylim(0, 100)
    ax_rsi.grid(True, alpha=0.3)

    step = max(1, len(data) // 6)
    ticks = list(range(0, len(data), step))
    ax_rsi.set_xticks(ticks)
    ax_rsi.set_xticklabels([data.index[i].strftime("%m-%d %H:%M") for i in ticks], rotation=30, ha="right")

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
) -> dict[str, bytes]:
    levels = last_levels or {}
    support = levels.get("support")
    resistance = levels.get("resistance")
    images: dict[str, bytes] = {}
    for tf, df in ohlc.items():
        if df is None or df.empty:
            continue
        path = out_dir / f"chart_{tf}.png" if out_dir else None
        if isinstance(display_bars, dict):
            n_display = display_bars.get(tf)
        else:
            n_display = display_bars
        images[tf] = render_ohlc_chart(
            df,
            title=f"CAC40 {tf}",
            support=float(support) if support is not None else None,
            resistance=float(resistance) if resistance is not None else None,
            out_path=path,
            rsi_period=rsi_period,
            display_bars=n_display,
        )
    return images
