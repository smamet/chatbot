from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from chatbot.trader.chart_renderer import pivot_history_pad, render_multi_timeframe
from chatbot.trader.config import TraderConfig
from chatbot.trader.fundmanager_client import FundManagerClient
from chatbot.trader.ig_connector import IgApiError, IgConnector, compact_ig_error
from chatbot.trader.live_ohlc_feed import LiveOhlcFeed
from chatbot.trader.llm_decision import (
    GeminiDecisionClient,
    SessionFactory,
    load_prompt,
    summarize_decision,
)
from chatbot.trader.llm_trigger import LlmTrigger
from chatbot.trader.market_calendar import session_snapshot
from chatbot.trader.models import (
    LegRole,
    LlmDecision,
    OrderPurpose,
    PositionLeg,
    Side,
    WorkingOrder,
    attached_deal_id,
)
from chatbot.trader.risk_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)

LiveOhlcProvider = Callable[[], LiveOhlcFeed]

TRIGGER_PRE_CLOSE_FLATTEN = "pre_close_flatten"
TRIGGER_MANUAL_FORCE = "manual_force"
TRIGGER_MANUAL_REPLAY = "manual_replay"


class LiveScheduler:
    """15m fail-closed live/demo loop: IG → charts → LLM → RiskGate → orders → FM."""

    def __init__(
        self,
        config: TraderConfig,
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
        self.stream_status_path: Path | None = None
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
        self.last_auto_flatten: dict[str, Any] | None = None
        self._last_processed_bar_ts: str | None = None
        self.last_book_repair: dict[str, Any] | None = None
        self._last_rest_book_sync_at: float | None = None

    def request_stop(self) -> None:
        self._stop = True

    def request_position_reconcile(self) -> None:
        """No-op: every live cycle already rebuilds the open book from IG."""
        return

    @staticmethod
    def _login_shared(conn: IgConnector, *, force: bool = False) -> None:
        from chatbot.trader.ig_session_cache import login_with_shared_cache

        try:
            login_with_shared_cache(conn, force=force)
        except Exception:
            conn.login()

    @staticmethod
    def _is_ig_auth_failure(exc: BaseException) -> bool:
        """True for 401 / invalid client token — worth one forced re-login."""
        info = compact_ig_error(exc)
        status = info.get("http_status")
        code = str(info.get("error_code") or "").lower()
        if status == 401:
            return True
        return any(
            needle in code
            for needle in (
                "client-token-invalid",
                "security.client-token",
                "authentication",
                "invalid-details",
            )
        )

    def _force_relogin_primary(self) -> None:
        """Drop cached CST/XST for the primary connector and login again."""
        from chatbot.trader.ig_session_cache import (
            invalidate_cached_session,
            session_cache_key,
        )

        self.ig._cst = None
        self.ig._security = None
        invalidate_cached_session(
            session_cache_key(
                api_key=self.ig.config.ig_api_key or "",
                username=self.ig.config.ig_username or "",
                account_id=self.ig.config.ig_account_id or "",
                acc_type=self.ig.config.ig_acc_type or "DEMO",
            )
        )
        self._login_shared(self.ig, force=True)

    def run_forever(self) -> None:
        self._login_shared(self.ig)
        for _cid, conn in self.order_connectors:
            if conn is not self.ig:
                try:
                    self._login_shared(conn)
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
            self._login_shared(self.ig)
        for _cid, conn in self.order_connectors:
            if conn is not self.ig and not conn._cst:
                try:
                    self._login_shared(conn)
                except Exception:
                    logger.exception("Secondary IG login failed")

    def run_once(
        self,
        *,
        force_llm: bool = False,
        replay_decision: LlmDecision | None = None,
        replay_of: str | None = None,
        replay_source: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        One live/paper cycle.

        ``force_llm`` (manual Run cycle now): ignore Adaptive/Fixed schedule and
        call Gemini when OHLC is usable and the book is not in unresolved desync.

        ``replay_decision``: skip Gemini and re-apply a stored decision (dev tool).
        ``replay_source``: full source cycle.json (for remapping stale position ids).

        When the FR40 session is closed, returns an idle heartbeat without IG
        login, OHLC, charts, or LLM — full path resumes on the next open poll.
        """
        session = session_snapshot(
            flatten_lead_minutes=int(self.config.flatten_lead_minutes or 30),
            flatten_enabled=bool(self.config.flatten_before_close),
            calendar_id=str(self.config.calendar_id or "") or None,
        )
        if not session.get("dealing_open"):
            return self._idle_market_closed(session)

        self.ensure_logged_in()
        self.trigger.note_position_ids(set(self.ig.ledger.positions.keys()))
        self.last_auto_flatten = None

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

        bar = self._last_closed_bar(ohlc_15)
        bar_ts = ""
        if not ohlc_15.empty:
            bar_ts = str(ohlc_15.index[-1])
        fill_events: list[dict[str, Any]] = []
        new_bar = bool(bar is not None and bar_ts and bar_ts != self._last_processed_bar_ts)
        if new_bar:
            # Paper: OHLC fill simulator. Live: MTM only — IG is SoT for fills.
            fill_events = self.ig.ledger.process_bar(
                bar, ts=bar_ts, apply_fills=bool(self.dry_run)
            )
            self.trigger.note_fills(fill_events)
            self._last_processed_bar_ts = bar_ts
        elif bar is not None and bar_ts == self._last_processed_bar_ts:
            # Same candle already applied (e.g. manual re-run) — MTM only.
            self.ig.ledger.mark_to_market(float(bar["close"]))
            self.ig.ledger.infer_phase()

        flatten_meta = self._build_flatten_meta(session)
        flatten_active = bool(flatten_meta.get("flatten_now"))
        # Live: same replace_open book sync as dashboard Apply — every cycle.
        reconcile = self._sync_ledger_from_ig()
        self.last_book_repair = reconcile.get("repair")
        wo_sync = {
            "ran": bool(reconcile.get("ran")),
            "dropped": list(reconcile.get("dropped_orders") or []),
            "imported": list(reconcile.get("imported") or []),
            "changed": bool(reconcile.get("changed")),
            "warnings": list(reconcile.get("warnings") or []),
        }

        # Prefer IG net for flatten sizing when available.
        net_for_flatten = (
            float(reconcile["ig_net"])
            if reconcile.get("ig_net") is not None
            else float(self.ig.ledger.net_size())
        )
        flatten_meta["net_exposure"] = net_for_flatten
        flatten_meta["local_net"] = float(self.ig.ledger.net_size())
        if reconcile.get("ig_net") is not None:
            flatten_meta["ig_net"] = float(reconcile["ig_net"])

        gate = RiskGate(
            self.config,
            self.ig.ledger,
            flatten_active=flatten_active,
            broker=None if self.dry_run else self.ig,
        )
        snap = self.ig.get_snapshot()
        cycle_dir = self.journal_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        cycle_dir.mkdir(parents=True, exist_ok=True)

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

        needs_flatten = flatten_active and (
            abs(net_for_flatten) > 1e-9 or bool(self.ig.ledger.entry_order_ids())
        )
        skip_llm_feed = bool(feed_meta.get("skip_llm"))
        # Unresolved book desync: do not let Gemini act on a phantom ledger.
        skip_llm_desync = bool(
            not self.dry_run and reconcile.get("desync") and not reconcile.get("repaired")
        )
        force_flatten_llm = bool(
            needs_flatten and not skip_llm_feed and not skip_llm_desync
        )
        force_manual_llm = bool(
            force_llm and not skip_llm_feed and not skip_llm_desync
        )
        replaying = replay_decision is not None

        images = {}
        decision = None
        gate_result = None
        llm_trigger_reasons: list[str] = []

        should_call = bool(
            bar
            and not ohlc_15.empty
            and not skip_llm_feed
            and not skip_llm_desync
            and (
                force_flatten_llm
                or force_manual_llm
                or (trig and trig.should_call)
            )
        )
        # Replay applies a stored decision on the *current* book — feed freshness
        # is not required (Gemini is not called).
        if replaying:
            should_call = not skip_llm_desync
            llm_trigger_reasons = [TRIGGER_MANUAL_REPLAY]
            if replay_of:
                llm_trigger_reasons.append(f"replay_of:{replay_of}")
        elif force_flatten_llm:
            llm_trigger_reasons = [TRIGGER_PRE_CLOSE_FLATTEN]
            if trig and trig.reasons:
                llm_trigger_reasons = list(
                    dict.fromkeys([*trig.reasons, TRIGGER_PRE_CLOSE_FLATTEN])
                )
        elif force_manual_llm:
            llm_trigger_reasons = [TRIGGER_MANUAL_FORCE]
            if trig and trig.reasons:
                llm_trigger_reasons = list(
                    dict.fromkeys([*trig.reasons, TRIGGER_MANUAL_FORCE])
                )
        elif should_call and trig is not None:
            llm_trigger_reasons = list(trig.reasons)

        if skip_llm_desync:
            logger.warning(
                "LLM skipped — IG book desync unresolved (%s)",
                reconcile.get("warnings"),
            )
        elif skip_llm_feed and needs_flatten and not replaying:
            logger.warning(
                "LLM skipped during flatten window — OHLC feed stale; auto-flatten will run (%s)",
                feed_meta.get("error") or feed_meta.get("warnings"),
            )
        elif skip_llm_feed and trig and trig.should_call and not replaying:
            logger.warning(
                "LLM skipped — OHLC feed not fresh enough (%s)",
                feed_meta.get("error") or feed_meta.get("warnings"),
            )

        if should_call and replaying:
            decision = replay_decision
            assert decision is not None
            # After IG book sync, local ids differ from the source cycle — rewrite
            # stale cancels / position ids and clear resting brackets so re-place
            # (and market_close) can land on the current book.
            from chatbot.application.trader_live_service import adapt_decision_for_replay

            decision = adapt_decision_for_replay(
                decision, self.ig.ledger, source=replay_source
            )
            gate_result = gate.apply(decision)
            self._mirror_orders_to_ig(gate_result)
            self._last_decision_summary = summarize_decision(decision)
            self.trigger.note_position_ids(set(self.ig.ledger.positions.keys()))
        elif should_call:
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
                symbol=self.config.symbol,
            )

            if images:
                # Same ledger snapshot as journal / dashboard (cleaned by IG sync above).
                decision = self.llm.decide(
                    images=images,
                    snapshot=snap,
                    phase=self.ig.ledger.phase,
                    prompt=load_prompt(
                        override=self.config.system_prompt or None,
                        profile_id=self.config.market_profile or None,
                    ),
                    order_size=float(self.config.order_size),
                    max_open_positions=int(self.config.max_open_positions),
                    last_decision=self._last_decision_summary,
                    allow_market_orders=bool(self.config.allow_market_orders)
                    or force_flatten_llm,
                    market_clock=flatten_meta,
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
                "LLM skipped (trigger=%s levels=%s/%s flatten=%s)",
                list(trig.reasons) if trig else [],
                levels.support,
                levels.resistance,
                flatten_active,
            )

        # Refresh net after LLM/gate; auto-flatten if still exposed in window.
        if flatten_active:
            if reconcile.get("ig_net") is not None and not gate_result:
                net_after = float(reconcile["ig_net"])
            else:
                # Re-check IG when we may have traded, else local.
                if not self.dry_run and self.ig._cst:
                    try:
                        net_after = float(self.ig.ig_net_size())
                    except Exception:
                        logger.exception("IG net refresh failed after cycle")
                        net_after = float(self.ig.ledger.net_size())
                else:
                    net_after = float(self.ig.ledger.net_size())
            still_needs = abs(net_after) > 1e-9 or bool(self.ig.ledger.entry_order_ids())
            if still_needs:
                self.last_auto_flatten = self._auto_flatten(net_after)
                if gate_result is None:
                    # Ensure entry cancels are mirrored when LLM did not run.
                    self._mirror_orders_to_ig(GateResult())

        error = None
        if should_call and not decision:
            error = "llm_fail_closed"
        elif feed_meta.get("error"):
            error = str(feed_meta["error"])
        if self.last_auto_flatten and self.last_auto_flatten.get("errors"):
            error = error or "auto_flatten_partial_failure"
        self.fm.notify(self.ig.ledger, error=error)
        chart_files = sorted(p.name for p in (cycle_dir / "charts").glob("chart_*.png"))
        charts_rel = f"journal/{cycle_dir.name}/charts" if chart_files else ""
        warnings = list(feed_meta.get("warnings") or [])
        warnings.extend(reconcile.get("warnings") or [])
        warnings.extend(wo_sync.get("warnings") or [])
        if self.last_auto_flatten and self.last_auto_flatten.get("errors"):
            warnings.extend(self.last_auto_flatten["errors"])
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
            "notes": list(gate_result.notes) if gate_result else [],
            "replay_of": replay_of or None,
            "dry_run": self.dry_run,
            "skipped": not should_call,
            "mirror": list(self.last_mirror_results),
            "charts_rel": charts_rel,
            "chart_files": chart_files,
            "cycle_dir": cycle_dir.name,
            "pnl": self.ig.ledger.pnl_payload(),
            "market_clock": flatten_meta,
            "auto_flatten": self.last_auto_flatten,
            "reconcile": reconcile,
            "book_repair": self.last_book_repair,
            "working_order_sync": wo_sync,
            "fill_events": fill_events,
            "ohlc_feed": {
                "last_bar_ts": feed_meta.get("last_bar_ts"),
                "top_up_added": feed_meta.get("top_up_added"),
                "top_up_ok": feed_meta.get("top_up_ok"),
                "stale": feed_meta.get("stale"),
                "skip_llm": feed_meta.get("skip_llm"),
                "warnings": warnings,
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

    def _build_flatten_meta(
        self, session: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        snap = session or session_snapshot(
            flatten_lead_minutes=int(self.config.flatten_lead_minutes or 30),
            flatten_enabled=bool(self.config.flatten_before_close),
            calendar_id=str(self.config.calendar_id or "") or None,
        )
        return {
            "enabled": bool(snap.get("flatten_enabled")),
            "flatten_now": bool(snap.get("flatten_now")),
            "reason": snap.get("flatten_reason") or "",
            "reasons": list(snap.get("flatten_reasons") or []),
            "close_at": snap.get("flatten_close_at"),
            "window_start": snap.get("flatten_window_start"),
            "minutes_to_close": snap.get("minutes_to_close"),
            "next_open_day": snap.get("next_open_day"),
            "next_open": snap.get("next_open"),
            "next_close": snap.get("next_close"),
            "dealing_open": bool(snap.get("dealing_open")),
            "now": snap.get("now"),
            "weekday": snap.get("weekday"),
            "tz": snap.get("tz"),
            "source": snap.get("source"),
            "net_exposure": float(self.ig.ledger.net_size()),
        }

    def _append_market_closed_heartbeat(self, session: dict[str, Any]) -> Path:
        close_id = str(session.get("close_id") or "unknown")
        safe = "".join(c for c in close_id if c.isalnum() or c in ("_", "-"))
        path = self.journal_dir / "market_closed" / f"{safe}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "next_open": session.get("next_open"),
            "skip_reason": "market_closed",
            "close_id": close_id,
            "weekday": session.get("weekday"),
        }
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
        return path

    def _idle_market_closed(self, session: dict[str, Any]) -> dict[str, Any]:
        """No OHLC / sync / charts / LLM — compact heartbeat for UI grouping."""
        self.last_auto_flatten = None
        self.last_ohlc_feed = None
        self.last_mirror_results = []
        self.last_book_repair = None
        heartbeat_path = self._append_market_closed_heartbeat(session)
        flatten_meta = self._build_flatten_meta(session)
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(
            "Market closed — idle (next_open=%s close_id=%s)",
            session.get("next_open"),
            session.get("close_id"),
        )
        return {
            "ts": ts,
            "skipped": True,
            "skip_reason": "market_closed",
            "session": session,
            "market_clock": flatten_meta,
            "mirror": [],
            "executed": [],
            "rejected": [],
            "decision": None,
            "llm_trigger": [],
            "ohlc_feed": {},
            "pnl": self.ig.ledger.pnl_payload(),
            "auto_flatten": None,
            "reconcile": {},
            "book_repair": None,
            "working_order_sync": {},
            "fill_events": [],
            "chart_files": [],
            "charts_rel": "",
            "cycle_dir": "",
            "dry_run": self.dry_run,
            "market_closed_heartbeat": str(heartbeat_path.name),
            "error": None,
        }

    def _live_state_path(self) -> Path | None:
        """``data/trader/{slug}/live/state.json`` when journal lives under live/."""
        parent = self.journal_dir.parent
        if parent.name == "live" or (parent / "state.json").exists():
            return parent / "state.json"
        return None

    def _reload_ledger_from_disk(self) -> bool:
        """Load open book from state.json (stream reconcile may have updated it)."""
        from chatbot.trader.hedge_ledger import HedgeLedger

        path = self._live_state_path()
        if path is None or not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("Failed to reload live state from %s", path)
            return False
        if not isinstance(raw, dict) or not raw:
            return False
        self.ig.ledger = HedgeLedger.from_state_dict(self.config, raw)
        return True

    def _stream_book_sync_skip(self) -> bool:
        """Skip REST book fetch when stream just reconciled (fresh + healthy)."""
        from chatbot.application.trader_stream_service import (
            STREAM_REST_RECONCILE_MINUTES,
            stream_book_reconcile_is_fresh,
        )

        path = self.stream_status_path
        if path is None or not path.exists():
            return False
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(raw, dict):
            return False
        if not stream_book_reconcile_is_fresh(raw):
            return False
        # Periodic safety REST reconcile even when stream is healthy.
        if self._last_rest_book_sync_at is not None:
            age_min = (time.time() - self._last_rest_book_sync_at) / 60.0
            if age_min >= STREAM_REST_RECONCILE_MINUTES:
                return False
        return True

    def _sync_ledger_from_ig(self) -> dict[str, Any]:
        """
        Live: rebuild open book from IG (same as dashboard Apply).

        When Lightstreamer TRADE wake-up just reconciled, skip a redundant
        REST list. Otherwise fetch positions + working orders and replace_open.
        """
        from chatbot.application.trader_live_service import sync_open_book_from_ig

        local_net = float(self.ig.ledger.net_size())
        out: dict[str, Any] = {
            "ran": False,
            "ig_net": None,
            "local_net": local_net,
            "desync": False,
            "repaired": False,
            "warnings": [],
            "closed": [],
            "opened": [],
            "imported": [],
            "repair": None,
            "secondary": [],
        }
        if self.dry_run or not self.ig._cst:
            out["ran"] = True
            out["ig_net"] = local_net
            return out

        if self._stream_book_sync_skip():
            # Stream may have written a newer open book to disk — reload so this
            # cycle does not overwrite nested attached TP/SL with a stale ledger.
            reloaded = self._reload_ledger_from_disk()
            out["ran"] = True
            out["ig_net"] = float(self.ig.ledger.net_size())
            out["local_net"] = out["ig_net"]
            out["skipped_stream_fresh"] = True
            out["reloaded_from_disk"] = reloaded
            out["warnings"].append("book_sync_skipped:stream_reconcile_fresh")
            return out

        try:
            ig_positions = self.ig.list_open_positions()
            ig_orders = self.ig.list_working_orders()
            self._last_rest_book_sync_at = time.time()
        except Exception as exc:
            if self._is_ig_auth_failure(exc):
                auth_info = compact_ig_error(exc)
                logger.warning(
                    "IG book sync auth failure — forcing re-login once (%s)",
                    auth_info.get("error_code")
                    or auth_info.get("http_status")
                    or exc,
                )
                try:
                    self._force_relogin_primary()
                    ig_positions = self.ig.list_open_positions()
                    ig_orders = self.ig.list_working_orders()
                    self._last_rest_book_sync_at = time.time()
                    out["warnings"].append("book_sync:relogin_after_auth_failure")
                except Exception as retry_exc:
                    logger.exception("IG book sync failed after re-login")
                    out["ran"] = True
                    out["desync"] = True
                    out["warnings"].append(f"book_sync:{retry_exc}")
                    return out
            else:
                logger.exception("IG book sync failed")
                out["ran"] = True
                out["desync"] = True
                out["warnings"].append(f"book_sync:{exc}")
                return out

        out["ran"] = True
        sync = sync_open_book_from_ig(
            self.ig.ledger,
            positions=ig_positions,
            working_orders=ig_orders,
            epic=self.config.epic,
            exit_price_for_leg=self._exit_price_for_leg,
        )
        prior_warnings = list(out.get("warnings") or [])
        out.update(
            {
                "ig_net": sync.get("ig_net"),
                "local_net": sync.get("local_net"),
                "desync": False,
                "repaired": True,
                "changed": bool(sync.get("changed")),
                "closed": list(sync.get("closed") or []),
                "opened": list(sync.get("opened") or []),
                "imported": list(sync.get("imported") or []),
                "dropped_orders": list(sync.get("dropped_orders") or []),
                "repair": sync.get("repair"),
                "warnings": prior_warnings + list(sync.get("warnings") or []),
                "quarantined": list(sync.get("quarantined") or []),
            }
        )

        try:
            primary_id = self.order_connectors[0][0]
            book = {
                str(oid): str(did)
                for oid, did in (sync.get("order_book") or {}).items()
            }
            for oid, order in self.ig.ledger.working_orders.items():
                did = (order.deal_id or "").strip()
                if did:
                    book[str(oid)] = did
            self._save_order_book(primary_id, book)
        except Exception:
            logger.exception("Failed to rewrite order book after IG sync")

        if out["closed"] or out["opened"]:
            self.trigger.note_fills(
                [{"type": "sync", "opened": out["opened"], "closed": out["closed"]}]
            )
        self.trigger.note_position_ids(set(self.ig.ledger.positions.keys()))

        for connector_id, conn in self.order_connectors:
            if conn is self.ig:
                continue
            row: dict[str, Any] = {"connector_id": connector_id}
            try:
                if not conn._cst:
                    conn.login()
                sec_net = float(conn.ig_net_size())
                row["ig_net"] = sec_net
                if abs(sec_net - float(out["ig_net"] or 0)) > 1e-6:
                    row["desync"] = True
                    out["warnings"].append(
                        f"secondary_desync:connector={connector_id} "
                        f"net={sec_net:+.4g} primary={float(out['ig_net'] or 0):+.4g}"
                    )
            except Exception as exc:
                row["error"] = str(exc)
                out["warnings"].append(f"secondary_reconcile:{connector_id}:{exc}")
            out["secondary"].append(row)

        return out

    def _exit_price_for_leg(self, leg: PositionLeg) -> float:
        """Best-effort exit when IG position vanishes (linked TP level, else mid)."""
        for order in self.ig.ledger.working_orders.values():
            if (
                order.position_id == leg.id
                and order.purpose in (OrderPurpose.TP, OrderPurpose.CLOSE)
            ):
                try:
                    return float(order.level)
                except (TypeError, ValueError):
                    pass
        if self.ig.ledger.last_price > 0:
            return float(self.ig.ledger.market_close_fill_price(leg))
        return float(leg.entry)

    def _auto_flatten(self, net: float) -> dict[str, Any]:
        """Deterministic directional flatten: market hedge |net| + cancel entries."""
        result: dict[str, Any] = {
            "net_before": float(net),
            "hedged": False,
            "side": None,
            "size": None,
            "cancelled_entries": [],
            "connectors": [],
            "errors": [],
        }
        # Cancel resting entries first (local); mirror will push cancels.
        for oid in list(self.ig.ledger.entry_order_ids()):
            try:
                self.ig.ledger.cancel_order(oid)
                result["cancelled_entries"].append(oid)
            except Exception as exc:
                result["errors"].append(f"cancel:{oid}:{exc}")

        if abs(float(net)) <= 1e-9:
            result["hedged"] = False
            logger.info("Auto-flatten: net already flat; cancelled %s entries", result["cancelled_entries"])
            return result

        side = Side.SELL if net > 0 else Side.BUY
        size = abs(float(net))
        result["side"] = side.value
        result["size"] = size

        # Primary ledger + all mirrored IG accounts.
        for connector_id, conn in self.order_connectors:
            row: dict[str, Any] = {"connector_id": connector_id, "ok": False}
            try:
                if conn is self.ig:
                    pid = conn.open_market_position(side, size, role=LegRole.HEDGE)
                else:
                    # Secondary accounts: open on their own ledger/IG; keep primary as source.
                    pid = conn.open_market_position(side, size, role=LegRole.HEDGE)
                row["ok"] = True
                row["position_id"] = pid
            except Exception as exc:
                row["error"] = str(exc)
                result["errors"].append(f"connector:{connector_id}:{exc}")
                logger.exception("Auto-flatten market open failed connector=%s", connector_id)
            result["connectors"].append(row)

        result["hedged"] = any(r.get("ok") for r in result["connectors"])
        result["net_after"] = float(self.ig.ledger.net_size())
        logger.warning(
            "Auto-flatten %s size=%s net_before=%s net_after=%s errors=%s",
            side.value,
            size,
            net,
            result["net_after"],
            result["errors"],
        )
        return result

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

    @staticmethod
    def _ig_confirm_fields(conn: Any) -> dict[str, Any]:
        raw = getattr(conn, "last_ig_result", None) or {}
        if not isinstance(raw, dict):
            return {}
        out: dict[str, Any] = {}
        for key in ("deal_status", "reason", "deal_reference"):
            val = raw.get(key)
            if val not in (None, ""):
                out[key] = val
        return out

    @staticmethod
    def _mirror_error_line(prefix: str, order_id: str, exc: BaseException) -> str:
        compact = compact_ig_error(exc)
        bits = [prefix, order_id]
        if compact.get("error_code"):
            bits.append(str(compact["error_code"]))
        elif compact.get("http_status") is not None:
            bits.append(f"HTTP{compact['http_status']}")
        bits.append(str(compact.get("error") or exc))
        return ":".join(bits)

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
                "deferred": [],
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

            remote_by_deal: dict[str, dict[str, Any]] = {}
            try:
                for raw in conn.list_working_orders() or []:
                    if not isinstance(raw, dict):
                        continue
                    wo = (
                        raw.get("workingOrderData")
                        if isinstance(raw.get("workingOrderData"), dict)
                        else raw
                    )
                    if not isinstance(wo, dict):
                        continue
                    did = str(wo.get("dealId") or raw.get("dealId") or "").strip()
                    if did:
                        remote_by_deal[did] = wo
            except Exception as exc:
                row["errors"].append(f"list_working_orders:{exc}")
                logger.exception(
                    "list_working_orders failed connector=%s", connector_id
                )
            remote_deal_ids = set(remote_by_deal)

            # Stale book rows (deal gone on IG but still mapped) block re-place — drop them.
            for local_id, deal_id in list(book.items()):
                deal_s = str(deal_id or "")
                if deal_s.startswith("attached:"):
                    continue
                if deal_s and remote_deal_ids and deal_s not in remote_deal_ids:
                    book.pop(local_id, None)
                    logger.info(
                        "Dropped stale order-book mapping order=%s deal=%s",
                        local_id,
                        deal_s,
                    )

            # Cancel orders no longer in the primary ledger.
            for local_id, deal_id in list(book.items()):
                if local_id in desired:
                    continue
                deal_s = str(deal_id or "")
                if deal_s.startswith("attached:"):
                    # IG-attached TP sentinel — drop locally; parent cancel removes it on IG.
                    book.pop(local_id, None)
                    logger.info(
                        "Dropped attached TP sentinel order=%s deal=%s",
                        local_id,
                        deal_s,
                    )
                    continue
                try:
                    conn.cancel_working_order(deal_id)
                    book.pop(local_id, None)
                    row["cancelled"].append(
                        {
                            "order_id": local_id,
                            "deal_id": deal_id,
                            **self._ig_confirm_fields(conn),
                        }
                    )
                except Exception as exc:
                    # If IG already has no such deal, drop the mapping and continue.
                    if deal_s and remote_deal_ids and deal_s not in remote_deal_ids:
                        book.pop(local_id, None)
                        row["cancelled"].append(
                            {
                                "order_id": local_id,
                                "deal_id": deal_id,
                                "via": "already_gone",
                            }
                        )
                        continue
                    row["errors"].append(
                        self._mirror_error_line("cancel", local_id, exc)
                    )
                    logger.exception(
                        "IG cancel failed connector=%s order=%s", connector_id, local_id
                    )

            # Sync booked orders: size drift → cancel+re-place; level drift → PUT.
            row.setdefault("amended", [])
            for local_id, deal_id in list(book.items()):
                if local_id not in desired:
                    continue
                deal_s = str(deal_id or "")
                if not deal_s or deal_s.startswith("attached:"):
                    continue
                order = desired[local_id]
                remote = remote_by_deal.get(deal_s)
                if not remote:
                    continue
                try:
                    remote_size = float(
                        remote["orderSize"]
                        if remote.get("orderSize") is not None
                        else remote.get("size")
                        or 0
                    )
                except (TypeError, ValueError):
                    remote_size = 0.0
                try:
                    remote_level = float(
                        remote["orderLevel"]
                        if remote.get("orderLevel") is not None
                        else remote.get("level")
                        or 0
                    )
                except (TypeError, ValueError):
                    remote_level = 0.0
                local_size = float(order.size)
                snapped = float(conn.snap_level(float(order.level)))
                size_drift = abs(local_size - remote_size) > 1e-6
                level_drift = abs(snapped - remote_level) > 1e-4
                if size_drift:
                    try:
                        conn.cancel_working_order(deal_s)
                        book.pop(local_id, None)
                        if conn is self.ig:
                            order.deal_id = ""
                        row["cancelled"].append(
                            {
                                "order_id": local_id,
                                "deal_id": deal_s,
                                "via": "size_replace",
                                **self._ig_confirm_fields(conn),
                            }
                        )
                    except Exception as exc:
                        row["errors"].append(
                            self._mirror_error_line("size_replace", local_id, exc)
                        )
                        logger.exception(
                            "IG size-replace cancel failed connector=%s order=%s",
                            connector_id,
                            local_id,
                        )
                    continue
                if not level_drift:
                    continue
                try:
                    from chatbot.application.trader_live_service import (
                        _attached_levels_from_wo_payload,
                    )

                    limit_level, stop_level = _attached_levels_from_wo_payload(
                        remote,
                        side=order.side,
                        level=remote_level if remote_level > 0 else float(order.level),
                    )
                    # Prefer local TP child level when present (authoritative).
                    for child in desired.values():
                        if (
                            child.parent_order_id == local_id
                            and child.purpose == OrderPurpose.TP
                        ):
                            limit_level = float(child.level)
                            break
                    conn.amend_working_order_by_deal_id(
                        deal_s,
                        order_type=order.type,
                        level=snapped,
                        limit_level=limit_level,
                        stop_level=stop_level,
                    )
                    if conn is self.ig:
                        order.level = snapped
                    row["amended"].append(
                        {
                            "order_id": local_id,
                            "deal_id": deal_s,
                            "level": snapped,
                            **self._ig_confirm_fields(conn),
                        }
                    )
                except Exception as exc:
                    row["errors"].append(
                        self._mirror_error_line("amend", local_id, exc)
                    )
                    logger.exception(
                        "IG amend failed connector=%s order=%s", connector_id, local_id
                    )

            # Place new ledger orders missing from this account's book.
            for local_id, order in desired.items():
                if local_id in book:
                    continue
                # TP children of a still-working entry: attach via limitLevel /
                # limitDistance on the entry (IG take-profit). Never attach
                # stopLevel — that is a closing stop-loss, not our reverse hedge.
                if (
                    order.purpose == OrderPurpose.TP
                    and order.parent_order_id
                    and order.parent_order_id in desired
                ):
                    parent = desired[order.parent_order_id]
                    parent_deal = str(
                        book.get(order.parent_order_id)
                        or (parent.deal_id or "")
                        or ""
                    ).strip()
                    # Same-cycle: entry not booked yet — attach when entry pushes.
                    if not parent_deal or parent_deal.startswith("attached:"):
                        continue
                    # Later cycle: entry already on IG — PUT limitDistance/Level.
                    try:
                        conn.amend_working_order_by_deal_id(
                            parent_deal,
                            order_type=parent.type,
                            level=float(parent.level),
                            limit_level=float(order.level),
                            stop_level=None,
                        )
                        ig_res = conn.last_ig_result or {}
                        attached = bool(
                            ig_res.get("tp_attached")
                            or ig_res.get("limit_level") is not None
                            or ig_res.get("limit_distance") is not None
                            or str(ig_res.get("deal_status") or "").upper()
                            in ("ACCEPTED", "DRY_RUN")
                        )
                        if not attached:
                            row["deferred"].append(
                                {
                                    "order_id": local_id,
                                    "via": "tp_attach_amend_deferred",
                                    "parent_deal_id": parent_deal,
                                }
                            )
                            continue
                        sentinel = attached_deal_id(parent_deal, OrderPurpose.TP)
                        book[local_id] = sentinel
                        if conn is self.ig:
                            order.deal_id = sentinel
                        via = (
                            "entry_amend_limitDistance"
                            if ig_res.get("limit_distance") is not None
                            else "entry_amend_limitLevel"
                        )
                        row["placed"].append(
                            {
                                "order_id": local_id,
                                "deal_id": sentinel,
                                "via": via,
                                "parent_deal_id": parent_deal,
                                "limit_distance": ig_res.get("limit_distance"),
                                "limit_level": ig_res.get("limit_level"),
                                **self._ig_confirm_fields(conn),
                            }
                        )
                    except Exception as exc:
                        row["errors"].append(
                            self._mirror_error_line("tp_attach_amend", local_id, exc)
                        )
                        logger.exception(
                            "IG TP amend on entry failed connector=%s order=%s",
                            connector_id,
                            local_id,
                        )
                    continue
                if order.purpose == OrderPurpose.CLOSE and order.parent_order_id:
                    if order.parent_order_id in desired:
                        continue
                # Already-attached TP sentinel on ledger — skip HTTP.
                if (order.deal_id or "").startswith("attached:"):
                    book[local_id] = order.deal_id
                    continue
                try:
                    # Take-profit on an open deal: attach limitLevel (closing TP).
                    if order.purpose in (OrderPurpose.TP, OrderPurpose.CLOSE):
                        leg = (
                            self.ig.ledger.positions.get(order.position_id or "")
                            if order.position_id
                            else None
                        )
                        deal_for_attach = (leg.deal_id or "").strip() if leg else ""
                        # Entry filled: parent gone but TP still has parent_order_id.
                        # Bind to the open leg on the same close side when possible.
                        if not deal_for_attach and order.purpose == OrderPurpose.TP:
                            for cand in self.ig.ledger.positions.values():
                                close_side = (
                                    Side.SELL if cand.side == Side.BUY else Side.BUY
                                )
                                if order.side == close_side and (cand.deal_id or "").strip():
                                    leg = cand
                                    deal_for_attach = (cand.deal_id or "").strip()
                                    order.position_id = cand.id
                                    order.parent_order_id = None
                                    break
                        if deal_for_attach:
                            conn.update_position_protection(
                                deal_for_attach,
                                limit_level=float(order.level),
                            )
                            sentinel = attached_deal_id(
                                deal_for_attach, OrderPurpose.TP
                            )
                            book[local_id] = sentinel
                            if conn is self.ig:
                                order.deal_id = sentinel
                            row["placed"].append(
                                {
                                    "order_id": local_id,
                                    "deal_id": sentinel,
                                    "via": "position_attach_tp",
                                    **self._ig_confirm_fields(conn),
                                }
                            )
                            continue
                        row["errors"].append(
                            f"place:{local_id}:pending_primary_deal"
                        )
                        continue

                    # Hedge cover: place immediately as forceOpen reverse STOP/LIMIT
                    # (same cycle as entry). Never IG stopLevel on the primary.
                    limit_level: float | None = None
                    tp_child_id: str | None = None
                    if order.purpose == OrderPurpose.ENTRY:
                        for cid, child in desired.items():
                            if (
                                child.parent_order_id == local_id
                                and child.purpose == OrderPurpose.TP
                            ):
                                limit_level = float(child.level)
                                tp_child_id = cid
                                break
                    if conn is self.ig:
                        pushed = conn.push_working_order(
                            order,
                            limit_level=limit_level,
                            # Never attach stopLevel — hedge is a separate forceOpen STOP.
                            stop_level=None,
                        )
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
                            parent_order_id=order.parent_order_id,
                        )
                        pushed = conn.place_order(
                            clone,
                            limit_level=limit_level,
                            stop_level=None,
                        )
                        deal_id = pushed.deal_id
                    if not deal_id:
                        raise IgApiError("IG place returned empty dealId")
                    book[local_id] = deal_id
                    row["placed"].append(
                        {
                            "order_id": local_id,
                            "deal_id": deal_id,
                            "via": (
                                "force_open_hedge"
                                if order.purpose == OrderPurpose.HEDGE_COVER
                                else "working_order"
                            ),
                            **self._ig_confirm_fields(conn),
                        }
                    )
                    if tp_child_id and tp_child_id not in book:
                        # TP attach: FX sends limitDistance; indices may send limitLevel.
                        # Do not treat missing limitLevel alone as "through market".
                        ig_res = conn.last_ig_result or {}
                        attached = bool(
                            ig_res.get("tp_attached")
                            or ig_res.get("limit_level") is not None
                            or ig_res.get("limit_distance") is not None
                        )
                        if attached:
                            sentinel = attached_deal_id(deal_id, OrderPurpose.TP)
                            book[tp_child_id] = sentinel
                            child = desired.get(tp_child_id)
                            if child is not None and conn is self.ig:
                                child.deal_id = sentinel
                            via = (
                                "entry_limitDistance"
                                if ig_res.get("limit_distance") is not None
                                else "entry_limitLevel"
                            )
                            row["placed"].append(
                                {
                                    "order_id": tp_child_id,
                                    "deal_id": sentinel,
                                    "via": via,
                                    "limit_distance": ig_res.get("limit_distance"),
                                    "limit_level": ig_res.get("limit_level"),
                                }
                            )
                        else:
                            clearance = str(ig_res.get("clearance") or "")
                            reason = (
                                "omit_tp_attach"
                                if "omit_tp_attach" in clearance
                                else "tp_not_in_payload"
                            )
                            row["deferred"].append(
                                {
                                    "order_id": tp_child_id,
                                    "via": "tp_attach_deferred",
                                    "reason": reason,
                                    "clearance": clearance or None,
                                }
                            )
                except Exception as exc:
                    row["errors"].append(
                        self._mirror_error_line("place", local_id, exc)
                    )
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
