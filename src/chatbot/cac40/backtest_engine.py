from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pandas as pd

from chatbot.cac40.chart_renderer import pivot_history_pad, render_multi_timeframe
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.decision_cache import DecisionCache
from chatbot.cac40.hedge_ledger import HedgeLedger
from chatbot.cac40.llm_decision import (
    GeminiDecisionClient,
    SessionFactory,
    load_prompt,
    parse_llm_json,
    summarize_decision,
)
from chatbot.cac40.llm_trigger import LlmTrigger
from chatbot.cac40.ohlc_store import load_ohlc_csv, resample_ohlc, slice_ohlc_period, window_asof
from chatbot.cac40.risk_gate import RiskGate

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], None]


@dataclass
class BacktestRunState:
    run_id: str
    status: str = "pending"  # pending|running|stopping|done|failed|stopped
    progress: float = 0.0
    current_bar: int = 0
    total_bars: int = 0
    error: str | None = None
    report: dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """Single-run multi-leg hedge backtest driven by 15m bars."""

    def __init__(
        self,
        config: Cac40Config,
        *,
        ohlc_path: Path,
        run_dir: Path,
        api_key: str = "",
        on_progress: ProgressCb | None = None,
        tenant_id: int | None = None,
        session_factory: SessionFactory | None = None,
    ) -> None:
        self.config = config
        self.ohlc_path = ohlc_path
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self.on_progress = on_progress
        self.tenant_id = tenant_id
        self.session_factory = session_factory
        self.state = BacktestRunState(run_id=run_dir.name)
        self._stop = False

    def run(self) -> dict[str, Any]:
        self.state.status = "running"
        self._write_state()
        try:
            report = self._run_inner()
            if self._stop or self.state.status == "stopped":
                self.state.status = "stopped"
            else:
                self.state.status = "done"
                self.state.progress = 1.0
            self.state.report = report
            self._write_state()
            (self.run_dir / "report.json").write_text(
                json.dumps(report, indent=2, default=str), encoding="utf-8"
            )
            return report
        except Exception as exc:
            self.state.status = "failed"
            self.state.error = str(exc)
            self._write_state()
            (self.run_dir / "error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            raise

    def request_stop(self) -> None:
        self._stop = True
        if self.state.status == "running":
            self.state.status = "stopping"
            self._write_state()

    def _run_inner(self) -> dict[str, Any]:
        df_full = load_ohlc_csv(self.ohlc_path, timezone=self.config.data_timezone)
        df_trade = slice_ohlc_period(df_full, self.config.backtest_period)
        # Resample from full history so 1H/1D charts keep pre-period context.
        df1h = resample_ohlc(df_full, "1h")
        df1d = resample_ohlc(df_full, "1D")
        warmup = max(1, int(self.config.warmup_bars or 14))
        lookback_15m = max(1, int(self.config.lookback_15m or 1))
        lookback_1h = max(1, int(self.config.lookback_1h or 1))
        lookback_1d = max(1, int(self.config.lookback_1d or 1))

        ledger = HedgeLedger(config=self.config, symbol=self.config.symbol)
        gate = RiskGate(self.config, ledger)
        cache = DecisionCache(self.run_dir / "decisions.json")
        llm = GeminiDecisionClient(
            api_key=self.api_key,
            model=self.config.gemini_model,
            temperature=float(self.config.llm_temperature),
            tenant_id=self.tenant_id,
            session_factory=self.session_factory,
        )
        trigger = LlmTrigger(
            band_points=float(self.config.llm_level_band_points or 15.0),
            mode=str(self.config.llm_trigger_mode or "levels"),
            every_bars=int(self.config.resolve_llm_every_bars()),
        )
        prompt = load_prompt()
        journal_path = self.run_dir / "journal.jsonl"

        bars = list(df_trade.itertuples())
        self.state.total_bars = len(bars)
        decisions_log: list[dict[str, Any]] = []
        last_decision_summary: dict[str, Any] | None = None

        for i, row in enumerate(bars):
            if self._stop:
                self.state.status = "stopped"
                break
            ts = pd.Timestamp(row.Index)
            bar = {
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
            }
            events = ledger.process_bar(bar, ts=str(ts))
            trigger.note_fills(events)

            # overnight funding on day change
            if i > 0:
                prev_ts = pd.Timestamp(bars[i - 1].Index)
                if ts.date() != prev_ts.date():
                    ledger.apply_overnight_funding()

            history_len = len(df_full.loc[:ts])
            warmed_up = history_len >= warmup
            levels = ledger.last_levels
            trig = trigger.evaluate(
                bar=bar,
                support=levels.support,
                resistance=levels.resistance,
                bar_index=i,
            )
            decision_point = warmed_up and trig.should_call
            if decision_point:
                snap = ledger.get_snapshot()
                # Extra bars so RSI is valid; extra history when drawing session pivots.
                rsi_seed = warmup
                pivots_on = bool(self.config.chart_show_pivots)
                pivot_period = self.config.chart_pivot_period or "D"
                pad_15 = pivot_history_pad(pivot_period, timeframe="15m") if pivots_on else 0
                pad_1h = pivot_history_pad(pivot_period, timeframe="1h") if pivots_on else 0
                w15 = window_asof(df_full, ts, lookback_15m + rsi_seed + pad_15)
                w1h = window_asof(df1h, ts, lookback_1h + rsi_seed + pad_1h)
                w1d = window_asof(df1d, ts, lookback_1d + rsi_seed)
                chart_key = ts.strftime("%Y%m%d_%H%M%S")
                chart_rel = f"charts/{chart_key}"
                chart_dir = self.run_dir / chart_rel
                images = render_multi_timeframe(
                    {"15m": w15, "1H": w1h, "1D": w1d},
                    last_levels=snap.last_levels,
                    out_dir=chart_dir,
                    rsi_period=warmup,
                    display_bars={"15m": lookback_15m, "1H": lookback_1h, "1D": lookback_1d},
                    show_rsi=bool(self.config.chart_show_rsi),
                    show_pivots=pivots_on,
                    pivot_period=pivot_period,
                )
                chart_files = sorted(p.name for p in chart_dir.glob("chart_*.png"))

                decision = None
                llm_error: str | None = None
                mode = (self.config.llm_mode or "live").strip().lower()
                if mode == "charts_only":
                    pass  # charts already rendered; skip Gemini
                elif mode == "replay":
                    cached = cache.get(str(ts))
                    if cached:
                        decision = parse_llm_json(json.dumps(cached["decision"]))
                    else:
                        llm_error = "replay_cache_miss"
                else:
                    decision = llm.decide(
                        images=images,
                        snapshot=snap,
                        phase=ledger.phase,
                        prompt=prompt,
                        order_size=float(self.config.order_size),
                        max_open_positions=int(self.config.max_open_positions),
                        min_exit_profit_points=float(self.config.min_exit_profit_points or 0),
                        last_decision=last_decision_summary,
                        allow_market_orders=bool(self.config.allow_market_orders),
                    )
                    if decision:
                        cache.put(
                            str(ts),
                            decision.to_dict(),
                            meta={"bar": i, "charts_rel": chart_rel, "chart_files": chart_files},
                        )
                    else:
                        llm_error = llm.last_error or "llm_fail_closed"

                gate_result = None
                entry = {
                    "ts": str(ts),
                    "bar": i,
                    "charts_rel": chart_rel,
                    "chart_files": chart_files,
                    "llm_mode": mode,
                    "llm_trigger": list(trig.reasons),
                    "book": {
                        "positions": len(snap.positions),
                        "working_orders": len(snap.working_orders),
                        "phase": snap.phase,
                    },
                    "lookback": {
                        "15m": lookback_15m,
                        "1h": lookback_1h,
                        "1d": lookback_1d,
                        "warmup_bars": warmup,
                        "history_len": history_len,
                    },
                    "decision": decision.to_dict() if decision else None,
                    "executed": [],
                    "rejected": [],
                    "llm_error": llm_error,
                }
                if decision:
                    gate_result = gate.apply(decision)
                    entry["executed"] = gate_result.executed
                    entry["rejected"] = gate_result.rejected
                    last_decision_summary = summarize_decision(decision)
                    trigger.on_success(
                        bar=bar,
                        support=ledger.last_levels.support,
                        resistance=ledger.last_levels.resistance,
                    )
                elif mode == "charts_only":
                    entry["rejected"] = []
                    trigger.on_success(
                        bar=bar,
                        support=ledger.last_levels.support,
                        resistance=ledger.last_levels.resistance,
                    )
                else:
                    entry["rejected"] = [llm_error or "llm_fail_closed"]
                    trigger.on_failure()
                pnl = ledger.pnl_payload()
                entry["pnl"] = {
                    "realized": ledger.realized_session,
                    "net_upl": pnl["net_upl"],
                    "equity": ledger.cash + pnl["net_upl"],
                }
                decisions_log.append(entry)
                self._write_decisions_log(decisions_log)

            with journal_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "ts": str(ts),
                            "bar": i,
                            "events": [
                                {
                                    "type": e.get("type"),
                                    "reason": e.get("reason"),
                                    "fill": e.get("fill"),
                                    "order": e.get("order"),
                                    "leg": e.get("leg"),
                                }
                                for e in events
                            ],
                            "net_upl": ledger.pnl_payload()["net_upl"],
                            "legs": ledger.legs_count(),
                        },
                        default=str,
                    )
                    + "\n"
                )

            self.state.current_bar = i + 1
            self.state.progress = (i + 1) / max(1, len(bars))
            if i % 25 == 0 or i == len(bars) - 1:
                self._write_state()
                if self.on_progress:
                    self.on_progress(asdict(self.state))

        return self._build_report(ledger, decisions_log)

    def _build_report(self, ledger: HedgeLedger, decisions: list[dict[str, Any]]) -> dict[str, Any]:
        equity = ledger.equity_curve
        peak = 0.0
        max_dd = 0.0
        for pt in equity:
            peak = max(peak, pt["equity"])
            dd = peak - pt["equity"]
            max_dd = max(max_dd, dd)
        wins = [t for t in ledger.closed_trades if t.realized_pnl > 0]
        losses = [t for t in ledger.closed_trades if t.realized_pnl <= 0]
        # Gemini API invocations (live mode only; charts_only / replay do not call).
        llm_calls_total = sum(
            1 for d in decisions if str(d.get("llm_mode") or "").strip().lower() == "live"
        )
        return {
            "run_id": self.state.run_id,
            "config": self.config.to_dict(),
            "bars": self.state.current_bar,
            "final_equity": equity[-1]["equity"] if equity else 0.0,
            "realized_pnl": ledger.realized_session,
            "max_drawdown": max_dd,
            "trades": len(ledger.closed_trades),
            "wins": len(wins),
            "losses": len(losses),
            "winrate": (len(wins) / len(ledger.closed_trades)) if ledger.closed_trades else None,
            "closed_trades": [
                {
                    "id": t.id,
                    "side": t.side.value,
                    "entry": t.entry,
                    "exit": t.exit,
                    "pnl": t.realized_pnl,
                    "role": t.role.value,
                    "bars_held": t.bars_held,
                }
                for t in ledger.closed_trades
            ],
            "equity_curve": equity,
            "decisions_count": len(decisions),
            "llm_calls_total": llm_calls_total,
            "decisions": decisions,
            "open_legs_end": [p.to_dict() for p in ledger.positions.values()],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _write_state(self) -> None:
        (self.run_dir / "state.json").write_text(
            json.dumps(asdict(self.state), indent=2, default=str), encoding="utf-8"
        )

    def _write_decisions_log(self, decisions: list[dict[str, Any]]) -> None:
        (self.run_dir / "decisions_log.json").write_text(
            json.dumps(decisions, indent=2, default=str), encoding="utf-8"
        )


def new_run_dir(base: Path, run_id: str | None = None) -> Path:
    rid = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    path = base / rid
    path.mkdir(parents=True, exist_ok=True)
    return path
