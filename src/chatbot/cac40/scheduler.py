from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from chatbot.cac40.chart_renderer import render_multi_timeframe
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.fundmanager_client import FundManagerClient
from chatbot.cac40.ig_connector import IgConnector
from chatbot.cac40.llm_decision import GeminiDecisionClient, load_prompt, summarize_decision
from chatbot.cac40.risk_gate import RiskGate

logger = logging.getLogger(__name__)


class LiveScheduler:
    """15m fail-closed live/demo loop: IG → charts → LLM → RiskGate → orders → FM."""

    def __init__(
        self,
        config: Cac40Config,
        *,
        api_key: str,
        journal_dir: Path,
        dry_run: bool = True,
        sleep_seconds: int = 900,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.sleep_seconds = sleep_seconds
        self.ig = IgConnector(config, dry_run=dry_run)
        self.fm = FundManagerClient(config)
        self.llm = GeminiDecisionClient(api_key=api_key, model=config.gemini_model)
        self._stop = False
        self._last_decision_summary: dict[str, Any] | None = None

    def request_stop(self) -> None:
        self._stop = True

    def run_forever(self) -> None:
        self.ig.login()
        while not self._stop:
            try:
                self.run_once()
            except Exception as exc:
                logger.exception("Live cycle failed")
                self.fm.notify(self.ig.ledger, error=str(exc))
                self._journal({"error": str(exc)})
            for _ in range(self.sleep_seconds):
                if self._stop:
                    break
                time.sleep(1)

    def run_once(self) -> dict[str, Any]:
        gate = RiskGate(self.config, self.ig.ledger)
        self.ig.sync_price()
        rsi_seed = max(2, int(self.config.warmup_bars or 14))
        ohlc_15 = self.ig.get_ohlc("15m", self.config.lookback_15m + rsi_seed)
        ohlc_1h = self.ig.get_ohlc("1h", self.config.lookback_1h + rsi_seed)
        ohlc_1d = self.ig.get_ohlc("1d", self.config.lookback_1d + rsi_seed)

        snap = self.ig.get_snapshot()
        cycle_dir = self.journal_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cycle_dir.mkdir(parents=True, exist_ok=True)

        images = {}
        if not ohlc_15.empty:
            frames = {"15m": ohlc_15}
            if not ohlc_1h.empty:
                frames["1H"] = ohlc_1h
            if not ohlc_1d.empty:
                frames["1D"] = ohlc_1d
            images = render_multi_timeframe(
                frames,
                last_levels=snap.last_levels,
                out_dir=cycle_dir / "charts",
                rsi_period=rsi_seed,
                display_bars={
                    "15m": self.config.lookback_15m,
                    "1H": self.config.lookback_1h,
                    "1D": self.config.lookback_1d,
                },
            )

        decision = None
        gate_result = None
        if images:
            decision = self.llm.decide(
                images=images,
                snapshot=snap,
                phase=self.ig.ledger.phase,
                prompt=load_prompt(),
                order_size=float(self.config.order_size),
                max_open_positions=int(self.config.max_open_positions),
                last_decision=self._last_decision_summary,
                allow_market_orders=bool(self.config.allow_market_orders),
            )
            if decision:
                gate_result = gate.apply(decision)
                self._mirror_orders_to_ig(gate_result)
                self._last_decision_summary = summarize_decision(decision)
            else:
                logger.error("LLM fail-closed: no actions this cycle")
        else:
            logger.warning("No OHLC available; skip LLM")

        self.fm.notify(self.ig.ledger, error=None if decision else "llm_fail_closed")
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "snapshot": self.ig.get_snapshot().to_dict(),
            "decision": decision.to_dict() if decision else None,
            "executed": gate_result.executed if gate_result else [],
            "rejected": gate_result.rejected if gate_result else ["llm_fail_closed"],
            "dry_run": self.dry_run,
        }
        (cycle_dir / "cycle.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        self._journal(payload)
        return payload

    def _mirror_orders_to_ig(self, gate_result: Any) -> None:
        """Orders already in ledger via RiskGate; optionally push to IG API."""
        if self.dry_run:
            return
        # V1: ledger is source; IgConnector.place_order used when wiring actions through connector
        logger.info("Executed: %s | Rejected: %s", gate_result.executed, gate_result.rejected)

    def _journal(self, payload: dict[str, Any]) -> None:
        path = self.journal_dir / "live.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")
