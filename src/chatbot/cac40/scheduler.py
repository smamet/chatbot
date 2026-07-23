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
from chatbot.cac40.market_calendar import flatten_check
from chatbot.cac40.models import LegRole, OrderPurpose, PositionLeg, Side, WorkingOrder
from chatbot.cac40.risk_gate import GateResult, RiskGate

logger = logging.getLogger(__name__)

LiveOhlcProvider = Callable[[], LiveOhlcFeed]

TRIGGER_PRE_CLOSE_FLATTEN = "pre_close_flatten"
TRIGGER_MANUAL_FORCE = "manual_force"


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
        self.last_auto_flatten: dict[str, Any] | None = None
        self._force_position_reconcile: bool = True  # arm / first cycle
        self._last_processed_bar_ts: str | None = None
        self._last_positions_bar_ts: str | None = None
        self._mirror_needs_positions: bool = False
        self.last_book_repair: dict[str, Any] | None = None

    def request_stop(self) -> None:
        self._stop = True

    def request_position_reconcile(self) -> None:
        """Next cycle will GET /positions (arm / manual)."""
        self._force_position_reconcile = True

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

    def run_once(self, *, force_llm: bool = False) -> dict[str, Any]:
        """
        One live/paper cycle.

        ``force_llm`` (manual Run cycle now): ignore Adaptive/Fixed schedule and
        call Gemini when OHLC is usable and the book is not in unresolved desync.
        """
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

        wo_sync = self._sync_working_orders_from_ig()
        if wo_sync.get("changed"):
            self._force_position_reconcile = True

        flatten_meta = self._build_flatten_meta()
        flatten_active = bool(flatten_meta.get("flatten_now"))
        bar_positions_due = bool(
            not self.dry_run
            and bar_ts
            and bar_ts != self._last_positions_bar_ts
        )
        need_positions = (
            flatten_active
            or self._force_position_reconcile
            or self._mirror_needs_positions
            or bar_positions_due
            or bool(wo_sync.get("changed"))
        )
        reconcile = self._reconcile_positions_from_ig(force=need_positions)
        if reconcile.get("ran"):
            self._force_position_reconcile = False
            self._mirror_needs_positions = False
            if bar_ts:
                self._last_positions_bar_ts = bar_ts
        self.last_book_repair = reconcile.get("repair")

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
        if force_flatten_llm:
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
        elif skip_llm_feed and needs_flatten:
            logger.warning(
                "LLM skipped during flatten window — OHLC feed stale; auto-flatten will run (%s)",
                feed_meta.get("error") or feed_meta.get("warnings"),
            )
        elif skip_llm_feed and trig and trig.should_call:
            logger.warning(
                "LLM skipped — OHLC feed not fresh enough (%s)",
                feed_meta.get("error") or feed_meta.get("warnings"),
            )

        if should_call:
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
                    allow_market_orders=bool(self.config.allow_market_orders)
                    or force_flatten_llm,
                    market_clock=flatten_meta,
                )
                # Count the attempt for wall-clock Fixed rate (live), even if fail-closed.
                self.trigger.mark_llm_called()
                if decision and bar is not None:
                    gate_result = gate.apply(decision)
                    self._mirror_orders_to_ig(gate_result)
                    if any(
                        (r.get("placed") or r.get("cancelled") or r.get("errors"))
                        for r in self.last_mirror_results
                    ):
                        self._mirror_needs_positions = True
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

    def _build_flatten_meta(self) -> dict[str, Any]:
        if not bool(self.config.flatten_before_close):
            return {
                "flatten_now": False,
                "enabled": False,
                "now": datetime.now(timezone.utc).isoformat(),
            }
        check = flatten_check(
            close_hhmm=str(self.config.market_close_paris or "22:00"),
            lead_minutes=int(self.config.flatten_lead_minutes or 30),
            tz=str(self.config.data_timezone or "Europe/Paris"),
        )
        return {
            "enabled": True,
            "flatten_now": bool(check.get("active")),
            "reason": check.get("reason") or "",
            "reasons": list(check.get("reasons") or []),
            "close_at": check.get("close_at"),
            "window_start": check.get("window_start"),
            "minutes_to_close": check.get("minutes_to_close"),
            "next_open_day": check.get("next_open_day"),
            "now": check.get("now"),
            "weekday": check.get("weekday"),
            "tz": check.get("tz"),
            "net_exposure": float(self.ig.ledger.net_size()),
        }

    def _sync_working_orders_from_ig(self) -> dict[str, Any]:
        """
        Live: drop local working orders whose IG dealId vanished.

        Never open/close legs from WO disappearance alone — that requires
        GET /positions (see ``_reconcile_positions_from_ig``). Attached
        sentinels (``attached:…``) are ignored here.
        """
        out: dict[str, Any] = {
            "ran": False,
            "dropped": [],
            "changed": False,
            "warnings": [],
        }
        if self.dry_run or not self.ig._cst:
            return out
        try:
            remote = self.ig.list_working_orders()
        except Exception as exc:
            logger.exception("list_working_orders failed")
            out["warnings"].append(f"working_order_sync:{exc}")
            return out
        out["ran"] = True
        remote_ids = {
            str(row.get("dealId") or "").strip()
            for row in remote
            if str(row.get("dealId") or "").strip()
        }
        for oid, order in list(self.ig.ledger.working_orders.items()):
            deal_id = (order.deal_id or "").strip()
            if not deal_id or deal_id.startswith("attached:"):
                continue
            if deal_id in remote_ids:
                continue
            # Vanished from IG — drop local WO only; positions reconcile decides fills.
            self.ig.ledger.working_orders.pop(oid, None)
            out["dropped"].append(
                {
                    "order_id": oid,
                    "deal_id": deal_id,
                    "purpose": order.purpose.value if order.purpose else "",
                }
            )
            out["changed"] = True
            out.setdefault("pending_position_check", True)
        return out

    def _reconcile_positions_from_ig(self, *, force: bool = False) -> dict[str, Any]:
        """
        Event-driven GET /positions — IG is SoT for open legs in live.

        Diffs by dealId, closes only when IG position is gone, adopts missing
        legs, and replace-rebuilds the open book on hard desync.
        """
        from chatbot.application.cac40_live_service import adopt_ig_snapshot_into_ledger

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
            "repair": None,
            "secondary": [],
        }
        if not force:
            return out
        if self.dry_run or not self.ig._cst:
            out["ran"] = True
            out["ig_net"] = local_net
            return out

        try:
            ig_positions = self.ig.list_open_positions()
            try:
                ig_orders = self.ig.list_working_orders()
            except Exception as exc:
                logger.exception("list_working_orders during reconcile failed")
                out["warnings"].append(f"working_order_sync:{exc}")
                ig_orders = []
        except Exception as exc:
            logger.exception("GET /positions reconcile failed")
            out["warnings"].append(f"positions_reconcile:{exc}")
            out["ran"] = True
            return out

        out["ran"] = True
        ig_by_deal = {
            str(row.get("deal_id") or "").strip(): row
            for row in ig_positions
            if str(row.get("deal_id") or "").strip()
        }
        ig_net = 0.0
        for row in ig_positions:
            size = float(row.get("size") or 0)
            if row.get("side") == Side.BUY:
                ig_net += size
            else:
                ig_net -= size
        out["ig_net"] = ig_net

        local_by_deal = {
            (p.deal_id or "").strip(): p
            for p in self.ig.ledger.positions.values()
            if (p.deal_id or "").strip()
        }
        unbound = [
            p for p in self.ig.ledger.positions.values() if not (p.deal_id or "").strip()
        ]

        # Close local legs whose IG dealId disappeared.
        for deal_id, leg in list(local_by_deal.items()):
            if deal_id in ig_by_deal:
                # Refresh size from IG.
                try:
                    ig_size = float(ig_by_deal[deal_id].get("size") or leg.size)
                    if ig_size > 0:
                        leg.size = ig_size
                except (TypeError, ValueError):
                    pass
                continue
            exit_px = self._exit_price_for_leg(leg)
            trade = self.ig.ledger.close_position(leg.id, exit_px)
            out["closed"].append(
                {
                    "id": leg.id,
                    "deal_id": deal_id,
                    "exit": exit_px,
                    "trade_id": trade.id if trade else None,
                }
            )

        # Adopt IG positions missing locally (additive bind).
        missing_rows = [
            row
            for did, row in ig_by_deal.items()
            if did not in {
                (p.deal_id or "").strip()
                for p in self.ig.ledger.positions.values()
                if (p.deal_id or "").strip()
            }
        ]
        if missing_rows:
            q = self.ig.ledger.quarantine_phantom_closes(set(ig_by_deal.keys()))
            if q:
                out["warnings"].append(f"phantom_quarantined:{','.join(q)}")
            adopt = adopt_ig_snapshot_into_ledger(
                self.ig.ledger,
                positions=missing_rows,
                working_orders=[],
                epic=self.config.epic,
                mode="additive",
            )
            out["opened"].extend(adopt.get("imported_positions") or [])
        elif unbound:
            # Legs without dealId cannot be matched — force replace below.
            out["desync"] = True
            out["warnings"].append(f"unbound_legs:{[p.id for p in unbound]}")

        local_net = float(self.ig.ledger.net_size())
        out["local_net"] = local_net
        still_unbound = [
            p.id for p in self.ig.ledger.positions.values() if not (p.deal_id or "").strip()
        ]
        if abs(ig_net - local_net) > 1e-6 or still_unbound:
            out["desync"] = True
            out["warnings"].append(
                f"book_desync: local_net={local_net:+.4g} ig_net={ig_net:+.4g} "
                f"unbound={still_unbound or '—'}"
            )
            logger.warning(out["warnings"][-1])
            repair = adopt_ig_snapshot_into_ledger(
                self.ig.ledger,
                positions=ig_positions,
                working_orders=ig_orders,
                epic=self.config.epic,
                mode="replace_open",
            )
            out["repair"] = {
                "imported_positions": repair.get("imported_positions"),
                "imported_orders": repair.get("imported_orders"),
                "quarantined": repair.get("quarantined"),
                "mode": "replace_open",
            }
            out["repaired"] = True
            out["desync"] = False
            out["local_net"] = float(self.ig.ledger.net_size())
            out["opened"] = list(repair.get("imported_positions") or [])
            # Rewrite primary order book from repair mapping.
            try:
                primary_id = self.order_connectors[0][0]
                book = {
                    str(oid): str(did)
                    for oid, did in (repair.get("order_book") or {}).items()
                }
                for oid, order in self.ig.ledger.working_orders.items():
                    did = (order.deal_id or "").strip()
                    if did:
                        book[str(oid)] = did
                self._save_order_book(primary_id, book)
            except Exception:
                logger.exception("Failed to rewrite order book after repair")
            self.trigger.note_fills(
                [{"type": "repair", "opened": out["opened"], "closed": out["closed"]}]
            )
            self.trigger.note_position_ids(set(self.ig.ledger.positions.keys()))

        # Secondary accounts: net check only (warn, no ledger rewrite).
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

        self._link_dormant_brackets_to_legs()

        if out.get("closed") or out.get("opened") or out.get("repaired"):
            self.ig.ledger.infer_phase()
            if self.ig.ledger.last_price:
                self.ig.ledger.mark_to_market(self.ig.ledger.last_price)
        return out

    def _link_dormant_brackets_to_legs(self) -> None:
        """After IG entry fill, attach local TP/hedge children to open legs (live)."""
        working = self.ig.ledger.working_orders
        for order in working.values():
            parent = order.parent_order_id
            if not parent or parent in working:
                continue
            if order.position_id and order.position_id in self.ig.ledger.positions:
                continue
            if order.purpose not in (
                OrderPurpose.TP,
                OrderPurpose.CLOSE,
                OrderPurpose.HEDGE_COVER,
            ):
                continue
            for leg in self.ig.ledger.positions.values():
                if order.purpose == OrderPurpose.HEDGE_COVER:
                    # Stop is usually same-side protective or opposite buy-to-cover.
                    order.position_id = leg.id
                    break
                if leg.side != order.side:
                    order.position_id = leg.id
                    break

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

            remote_deal_ids: set[str] = set()
            try:
                for raw in conn.list_working_orders() or []:
                    if not isinstance(raw, dict):
                        continue
                    wo = raw.get("workingOrderData") if isinstance(raw.get("workingOrderData"), dict) else raw
                    did = str(
                        (wo or {}).get("dealId")
                        or raw.get("dealId")
                        or ""
                    ).strip()
                    if did:
                        remote_deal_ids.add(did)
            except Exception as exc:
                row["errors"].append(f"list_working_orders:{exc}")
                logger.exception(
                    "list_working_orders failed connector=%s", connector_id
                )

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
                    row["cancelled"].append({"order_id": local_id, "deal_id": deal_id})
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
                    row["errors"].append(f"cancel:{local_id}:{exc}")
                    logger.exception(
                        "IG cancel failed connector=%s order=%s", connector_id, local_id
                    )

            # Place new ledger orders missing from this account's book.
            for local_id, order in desired.items():
                if local_id in book:
                    continue
                # TP children of a still-working entry: attach via limitLevel on the
                # entry (IG take-profit). Never attach stopLevel — that is a closing
                # stop-loss, not our reverse-side hedge.
                if (
                    order.purpose == OrderPurpose.TP
                    and order.parent_order_id
                    and order.parent_order_id in desired
                ):
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
                        if deal_for_attach:
                            conn.update_position_protection(
                                deal_for_attach,
                                limit_level=float(order.level),
                            )
                            sentinel = f"attached:{deal_for_attach}"
                            book[local_id] = sentinel
                            if conn is self.ig:
                                order.deal_id = sentinel
                            row["placed"].append(
                                {
                                    "order_id": local_id,
                                    "deal_id": sentinel,
                                    "via": "position_attach_tp",
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
                        }
                    )
                    if tp_child_id and tp_child_id not in book:
                        book[tp_child_id] = f"attached:{deal_id}"
                        row["placed"].append(
                            {
                                "order_id": tp_child_id,
                                "deal_id": f"attached:{deal_id}",
                                "via": "entry_limitLevel",
                            }
                        )
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
