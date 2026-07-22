from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from chatbot.cac40.chart_renderer import pivot_history_pad, render_multi_timeframe
from chatbot.cac40.config import Cac40Config
from chatbot.cac40.fundmanager_client import FundManagerClient
from chatbot.cac40.ig_connector import IgApiError, IgConnector
from chatbot.cac40.live_ohlc_feed import LiveOhlcFeed
from chatbot.cac40.llm_decision import (
    GeminiDecisionClient,
    SessionFactory,
    load_prompt,
    summarize_decision,
)
from chatbot.cac40.llm_trigger import LlmTrigger
from chatbot.cac40.models import OrderPurpose, WorkingOrder
from chatbot.cac40.risk_gate import RiskGate

logger = logging.getLogger(__name__)

LiveOhlcProvider = Callable[[], LiveOhlcFeed]


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
        tenant_id: int | None = None,
        session_factory: SessionFactory | None = None,
        order_connectors: list[tuple[int, IgConnector]] | None = None,
        orders_dir: Path | None = None,
        ohlc_provider: LiveOhlcProvider | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.sleep_seconds = sleep_seconds
        self.tenant_id = tenant_id
        self.session_factory = session_factory
        self.ig = IgConnector(config, dry_run=dry_run)
        self.order_connectors: list[tuple[int, IgConnector]] = order_connectors or [
            (0, self.ig)
        ]
        self.orders_dir = orders_dir or (self.journal_dir / "order_books")
        self.orders_dir.mkdir(parents=True, exist_ok=True)
        self.ohlc_provider = ohlc_provider
        self.fm = FundManagerClient(config)
        self.llm = GeminiDecisionClient(
            api_key=api_key,
            model=config.gemini_model,
            temperature=float(config.llm_temperature),
            tenant_id=tenant_id,
            session_factory=session_factory,
        )
        self.trigger = LlmTrigger(
            band_points=float(config.llm_level_band_points or 15.0),
            mode=str(config.llm_trigger_mode or "levels"),
            every_bars=int(config.resolve_llm_every_bars()),
        )
        self._stop = False
        self._last_decision_summary: dict[str, Any] | None = None
        self._cycle_index = 0
        self.last_mirror_results: list[dict[str, Any]] = []
        self.last_ohlc_feed: LiveOhlcFeed | None = None

    def request_stop(self) -> None:
        self._stop = True

    def run_forever(self) -> None:
        self.ig.login()
        for _cid, conn in self.order_connectors:
            if conn is not self.ig:
                try:
                    conn.login()
                except Exception:
                    logger.exception("Secondary IG login failed (connector continues)")
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

    def ensure_logged_in(self) -> None:
        if not self.ig._cst:
            self.ig.login()
        for _cid, conn in self.order_connectors:
            if conn is not self.ig and not conn._cst:
                try:
                    conn.login()
                except Exception:
                    logger.exception("Secondary IG login failed")

    def run_once(self) -> dict[str, Any]:
        self.ensure_logged_in()
        gate = RiskGate(self.config, self.ig.ledger)
        self.trigger.note_position_ids(set(self.ig.ledger.positions.keys()))

        rsi_seed = max(2, int(self.config.warmup_bars or 14))
        pivots_on = bool(self.config.chart_show_pivots)
        pivot_period = self.config.chart_pivot_period or "D"
        feed_meta: dict[str, Any] = {}
        feed = self._load_ohlc_frames()
        self.last_ohlc_feed = feed
        ohlc_15 = feed["ohlc_15"]
        ohlc_1h = feed["ohlc_1h"]
        ohlc_1d = feed["ohlc_1d"]
        feed_meta = feed.get("meta") or {}

        # Mark price from local/cache close (or tiny IG fetch when no provider).
        if float(feed.get("last_price") or 0) > 0:
            self.ig.ledger.last_price = float(feed["last_price"])
            self.ig.ledger.mark_to_market(float(feed["last_price"]))
        else:
            self.ig.sync_price()

        snap = self.ig.get_snapshot()
        cycle_dir = self.journal_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cycle_dir.mkdir(parents=True, exist_ok=True)

        bar = self._last_closed_bar(ohlc_15)
        levels = self.ig.ledger.last_levels
        trig = (
            self.trigger.evaluate(
                bar=bar,
                support=levels.support,
                resistance=levels.resistance,
                bar_index=self._cycle_index,
            )
            if bar
            else None
        )
        self._cycle_index += 1

        images = {}
        decision = None
        gate_result = None
        llm_trigger_reasons: list[str] = []
        skip_llm_feed = bool(feed_meta.get("skip_llm"))

        should_call = bool(
            bar and trig and trig.should_call and not ohlc_15.empty and not skip_llm_feed
        )
        if skip_llm_feed and trig and trig.should_call:
            logger.warning(
                "LLM skipped — OHLC feed not fresh enough (%s)",
                feed_meta.get("error") or feed_meta.get("warnings"),
            )
        if should_call and trig is not None:
            llm_trigger_reasons = list(trig.reasons)
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
                show_rsi=bool(self.config.chart_show_rsi),
                show_pivots=pivots_on,
                pivot_period=pivot_period,
            )

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
                # Count the attempt for wall-clock Fixed rate (live), even if fail-closed.
                self.trigger.mark_llm_called()
                if decision and bar is not None:
                    gate_result = gate.apply(decision)
                    self._mirror_orders_to_ig(gate_result)
                    self._last_decision_summary = summarize_decision(decision)
                    self.trigger.on_success(
                        bar=bar,
                        support=self.ig.ledger.last_levels.support,
                        resistance=self.ig.ledger.last_levels.resistance,
                    )
                    self.trigger.note_position_ids(set(self.ig.ledger.positions.keys()))
                else:
                    logger.error("LLM fail-closed: no actions this cycle")
                    self.trigger.on_failure()
            else:
                logger.warning("No OHLC available; skip LLM")
                self.trigger.on_failure()
        elif ohlc_15.empty:
            logger.warning("No OHLC available; skip LLM")
        else:
            logger.info(
                "LLM skipped (trigger=%s levels=%s/%s)",
                list(trig.reasons) if trig else [],
                levels.support,
                levels.resistance,
            )

        error = None
        if should_call and not decision:
            error = "llm_fail_closed"
        elif feed_meta.get("error"):
            error = str(feed_meta["error"])
        self.fm.notify(self.ig.ledger, error=error)
        chart_files = sorted(p.name for p in (cycle_dir / "charts").glob("chart_*.png"))
        charts_rel = f"journal/{cycle_dir.name}/charts" if chart_files else ""
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "snapshot": self.ig.get_snapshot().to_dict(),
            "llm_trigger": llm_trigger_reasons,
            "decision": decision.to_dict() if decision else None,
            "executed": gate_result.executed if gate_result else [],
            "rejected": (
                gate_result.rejected
                if gate_result
                else (["llm_fail_closed"] if should_call and not decision else [])
            ),
            "dry_run": self.dry_run,
            "skipped": not should_call,
            "mirror": list(self.last_mirror_results),
            "charts_rel": charts_rel,
            "chart_files": chart_files,
            "cycle_dir": cycle_dir.name,
            "pnl": self.ig.ledger.pnl_payload(),
            "ohlc_feed": {
                "last_bar_ts": feed_meta.get("last_bar_ts"),
                "top_up_added": feed_meta.get("top_up_added"),
                "top_up_ok": feed_meta.get("top_up_ok"),
                "stale": feed_meta.get("stale"),
                "skip_llm": feed_meta.get("skip_llm"),
                "warnings": feed_meta.get("warnings") or [],
                "error": feed_meta.get("error"),
                "allowance": feed_meta.get("allowance"),
                "source": feed_meta.get("source"),
            },
        }
        (cycle_dir / "cycle.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        self._journal(payload)
        return payload

    def _load_ohlc_frames(self) -> dict[str, Any]:
        """
        Prefer local CSV + tiny IG top-up via ``ohlc_provider``.

        Fallback (CLI / no CSV wiring): full IG lookbacks (legacy behaviour).
        """
        if self.ohlc_provider is not None:
            feed = self.ohlc_provider()
            meta = {
                "source": "local_csv",
                "last_bar_ts": feed.last_bar_ts,
                "top_up_added": feed.top_up_added,
                "top_up_ok": feed.top_up_ok,
                "stale": feed.stale,
                "skip_llm": feed.skip_llm,
                "warnings": list(feed.warnings),
                "error": feed.error,
                "allowance": feed.allowance,
            }
            if feed.allowance:
                remaining = feed.allowance.get("remaining") or feed.allowance.get(
                    "remainingAllowance"
                )
                expiry = feed.allowance.get("expiry") or feed.allowance.get(
                    "allowanceExpiry"
                )
                logger.info(
                    "IG historical allowance remaining=%s expiry=%s",
                    remaining,
                    expiry,
                )
            for warn in feed.warnings:
                logger.warning("OHLC feed: %s", warn)
            if feed.error and feed.skip_llm:
                logger.error("OHLC feed: %s", feed.error)
            return {
                "ohlc_15": feed.ohlc_15,
                "ohlc_1h": feed.ohlc_1h,
                "ohlc_1d": feed.ohlc_1d,
                "last_price": feed.last_price,
                "meta": meta,
            }

        pad_15 = (
            pivot_history_pad(self.config.chart_pivot_period or "D", timeframe="15m")
            if self.config.chart_show_pivots
            else 0
        )
        pad_1h = (
            pivot_history_pad(self.config.chart_pivot_period or "D", timeframe="1h")
            if self.config.chart_show_pivots
            else 0
        )
        rsi_seed = max(2, int(self.config.warmup_bars or 14))
        ohlc_15 = self.ig.get_ohlc(
            "15m", self.config.lookback_15m + rsi_seed + pad_15
        )
        ohlc_1h = self.ig.get_ohlc("1h", self.config.lookback_1h + rsi_seed + pad_1h)
        ohlc_1d = self.ig.get_ohlc("1d", self.config.lookback_1d + rsi_seed)
        last_price = (
            float(ohlc_15["close"].iloc[-1]) if not ohlc_15.empty else 0.0
        )
        allowance = getattr(self.ig, "last_price_allowance", None)
        if allowance:
            logger.info(
                "IG historical allowance remaining=%s expiry=%s",
                allowance.get("remaining") or allowance.get("remainingAllowance"),
                allowance.get("expiry") or allowance.get("allowanceExpiry"),
            )
        return {
            "ohlc_15": ohlc_15,
            "ohlc_1h": ohlc_1h,
            "ohlc_1d": ohlc_1d,
            "last_price": last_price,
            "meta": {
                "source": "ig",
                "last_bar_ts": str(ohlc_15.index[-1]) if not ohlc_15.empty else None,
                "top_up_added": None,
                "top_up_ok": True,
                "stale": False,
                "skip_llm": False,
                "warnings": [],
                "error": None,
                "allowance": allowance,
            },
        }

    @staticmethod
    def _last_closed_bar(ohlc_15) -> dict[str, float] | None:
        if ohlc_15 is None or getattr(ohlc_15, "empty", True):
            return None
        row = ohlc_15.iloc[-1]
        return {
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }

    def _order_book_path(self, connector_id: int) -> Path:
        return self.orders_dir / f"orders_{connector_id}.json"

    def _load_order_book(self, connector_id: int) -> dict[str, str]:
        path = self._order_book_path(connector_id)
        if not path.exists():
            return {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}
        return {str(k): str(v) for k, v in raw.items() if k and v}

    def _save_order_book(self, connector_id: int, book: dict[str, str]) -> None:
        path = self._order_book_path(connector_id)
        path.write_text(json.dumps(book, indent=2), encoding="utf-8")

    def _mirror_orders_to_ig(self, gate_result: Any) -> None:
        """Push ledger working orders to each IG account; cancel removed ones."""
        self.last_mirror_results = []
        if self.dry_run:
            logger.info(
                "Dry-run mirror skipped. Executed: %s | Rejected: %s",
                gate_result.executed,
                gate_result.rejected,
            )
            return

        desired = dict(self.ig.ledger.working_orders)
        for connector_id, conn in self.order_connectors:
            book = self._load_order_book(connector_id)
            row: dict[str, Any] = {
                "connector_id": connector_id,
                "placed": [],
                "cancelled": [],
                "errors": [],
            }
            try:
                if not conn.epic_compatible_with_account():
                    row["errors"].append("epic_account_mismatch")
                    self.last_mirror_results.append(row)
                    continue
            except Exception as exc:
                row["errors"].append(f"compat_check:{exc}")
                self.last_mirror_results.append(row)
                continue

            # Cancel orders no longer in the primary ledger.
            for local_id, deal_id in list(book.items()):
                if local_id in desired:
                    continue
                try:
                    conn.cancel_working_order(deal_id)
                    book.pop(local_id, None)
                    row["cancelled"].append({"order_id": local_id, "deal_id": deal_id})
                except Exception as exc:
                    row["errors"].append(f"cancel:{local_id}:{exc}")
                    logger.exception(
                        "IG cancel failed connector=%s order=%s", connector_id, local_id
                    )

            # Place new ledger orders missing from this account's book.
            for local_id, order in desired.items():
                if local_id in book:
                    continue
                try:
                    if conn is self.ig:
                        pushed = conn.push_working_order(order)
                        deal_id = pushed.deal_id
                    else:
                        clone = WorkingOrder(
                            id="",
                            type=order.type,
                            side=order.side,
                            level=order.level,
                            size=order.size,
                            purpose=order.purpose or OrderPurpose.ENTRY,
                            position_id=order.position_id,
                        )
                        pushed = conn.place_order(clone)
                        deal_id = pushed.deal_id
                    if not deal_id:
                        raise IgApiError("IG place returned empty dealId")
                    book[local_id] = deal_id
                    row["placed"].append({"order_id": local_id, "deal_id": deal_id})
                except Exception as exc:
                    row["errors"].append(f"place:{local_id}:{exc}")
                    logger.exception(
                        "IG place failed connector=%s order=%s", connector_id, local_id
                    )

            self._save_order_book(connector_id, book)
            self.last_mirror_results.append(row)

        logger.info(
            "Mirror done. Executed: %s | Rejected: %s | Results: %s",
            gate_result.executed,
            gate_result.rejected,
            self.last_mirror_results,
        )

    def _journal(self, payload: dict[str, Any]) -> None:
        path = self.journal_dir / "live.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=str) + "\n")

    def close(self) -> None:
        try:
            self.ig.close()
        except Exception:
            pass
        for _cid, conn in self.order_connectors:
            if conn is self.ig:
                continue
            try:
                conn.close()
            except Exception:
                pass
