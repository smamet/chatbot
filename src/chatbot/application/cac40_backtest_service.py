from __future__ import annotations

import json
import logging
import re
import shutil
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from chatbot.cac40.backtest_engine import BacktestEngine, new_run_dir
from chatbot.cac40.config import Cac40Config, public_config_snapshot
from chatbot.config.settings import Settings

logger = logging.getLogger(__name__)

MAX_OHLC_GAP_DAYS = 60
BOOTSTRAP_LOOKBACK_DAYS = 60

SessionFactory = Callable[[], Session]

_SAFE_RUN_ID = re.compile(r"^[\w.-]+$")
_SAFE_CHART_KEY = re.compile(r"^[\w.-]+$")
_SAFE_CHART_FILE = re.compile(r"^chart_[\w.-]+\.png$")


def _read_json(path: Path, default: Any = None) -> Any:
    """Load JSON; return default on missing, empty, or corrupt files."""
    if not path.exists():
        return default
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Invalid JSON at %s: %s", path, exc)
        return default


def cac40_root(settings: Settings, tenant_slug: str) -> Path:
    root = settings.data_root / "cac40" / tenant_slug
    root.mkdir(parents=True, exist_ok=True)
    return root


def runs_dir(settings: Settings, tenant_slug: str) -> Path:
    path = cac40_root(settings, tenant_slug) / "backtests"
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_runs(settings: Settings, tenant_slug: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sorted(runs_dir(settings, tenant_slug).iterdir(), reverse=True):
        if not path.is_dir():
            continue
        state_path = path / "state.json"
        report_path = path / "report.json"
        config_path = path / "config.json"
        state = _read_json(state_path, default={}) or {}
        report = _read_json(report_path, default={}) or {}
        raw_cfg: dict[str, Any] = {}
        loaded = _read_json(config_path, default={})
        if isinstance(loaded, dict):
            raw_cfg = loaded
        if not raw_cfg:
            raw_cfg = dict(report.get("config") or {})
        cfg = public_config_snapshot(raw_cfg)
        items.append(
            {
                "run_id": path.name,
                "status": state.get("status", "unknown"),
                "progress": state.get("progress", 0),
                "period": cfg.get("backtest_period") or "—",
                "final_equity": report.get("final_equity"),
                "max_drawdown": report.get("max_drawdown"),
                "trades": report.get("trades"),
                "winrate": report.get("winrate"),
                "llm_calls_total": report.get("llm_calls_total"),
                "llm_trigger_mode": cfg.get("llm_trigger_mode") or "levels",
                "chart_show_rsi": bool(cfg["chart_show_rsi"]) if "chart_show_rsi" in cfg else True,
                "chart_show_pivots": bool(cfg["chart_show_pivots"]) if "chart_show_pivots" in cfg else False,
                "chart_pivot_period": str(cfg.get("chart_pivot_period") or "D"),
                "config": cfg,
            }
        )
    return items


_LOCK = threading.Lock()
_THREADS: dict[str, threading.Thread] = {}
_ENGINES: dict[str, BacktestEngine] = {}


def _run_path(settings: Settings, tenant_slug: str, run_id: str) -> Path | None:
    if not _SAFE_RUN_ID.match(run_id or ""):
        return None
    base = runs_dir(settings, tenant_slug).resolve()
    path = (base / run_id).resolve()
    try:
        path.relative_to(base)
    except ValueError:
        return None
    return path


def get_run(settings: Settings, tenant_slug: str, run_id: str) -> dict[str, Any]:
    path = _run_path(settings, tenant_slug, run_id)
    if path is None or not path.exists():
        raise FileNotFoundError(run_id)
    state = _read_json(path / "state.json", default={}) or {}
    report = _read_json(path / "report.json", default={}) or {}
    decisions = _load_decision_entries(path, tenant_slug=tenant_slug, run_id=run_id)
    return {
        "run_id": run_id,
        "path": str(path),
        "state": state,
        "report": report,
        "decisions": decisions,
    }


def _equity_curve_indexes(run_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Index report equity_curve by 0-based decision bar and ts for PnL backfill."""
    by_bar: dict[int, dict[str, Any]] = {}
    by_ts: dict[str, dict[str, Any]] = {}
    report_path = run_path / "report.json"
    report = _read_json(report_path, default={})
    if not isinstance(report, dict):
        return by_bar, by_ts
    for pt in report.get("equity_curve") or []:
        if not isinstance(pt, dict):
            continue
        pnl = {
            "realized": pt.get("realized"),
            "net_upl": pt.get("net_upl"),
            "equity": pt.get("equity"),
        }
        # equity_curve.bar is 1-based (ledger.bar_index); decisions use 0-based loop i
        if pt.get("bar") is not None:
            try:
                by_bar[int(pt["bar"]) - 1] = pnl
            except (TypeError, ValueError):
                pass
        if pt.get("ts") is not None:
            by_ts[str(pt["ts"])] = pnl
    return by_bar, by_ts


def _resolve_entry_pnl(
    entry: dict[str, Any],
    by_bar: dict[int, dict[str, Any]],
    by_ts: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    existing = entry.get("pnl")
    if isinstance(existing, dict) and (
        existing.get("realized") is not None or existing.get("net_upl") is not None
    ):
        return existing
    ts = entry.get("ts")
    if ts is not None and str(ts) in by_ts:
        return by_ts[str(ts)]
    bar = entry.get("bar")
    if bar is not None:
        try:
            key = int(bar)
        except (TypeError, ValueError):
            key = None
        if key is not None and key in by_bar:
            return by_bar[key]
    return existing if isinstance(existing, dict) else None


def _load_decision_entries(
    run_path: Path, *, tenant_slug: str, run_id: str
) -> list[dict[str, Any]]:
    """Prefer decisions_log.json (charts + gate results); fall back to cache/report."""
    entries: list[dict[str, Any]] = []
    log_path = run_path / "decisions_log.json"
    raw = _read_json(log_path, default=None)
    if isinstance(raw, list):
        entries = list(raw)
    if not entries:
        report = _read_json(run_path / "report.json", default={}) or {}
        if isinstance(report.get("decisions"), list):
            entries = list(report["decisions"])
    if not entries:
        cached = _read_json(run_path / "decisions.json", default=None)
        if isinstance(cached, dict):
            for ts, payload in cached.items():
                meta = (payload or {}).get("meta") or {}
                entries.append(
                    {
                        "ts": ts,
                        "decision": (payload or {}).get("decision"),
                        "charts_rel": meta.get("charts_rel"),
                        "chart_files": meta.get("chart_files") or [],
                        "executed": [],
                        "rejected": [],
                        "bar": meta.get("bar"),
                    }
                )

    by_bar, by_ts = _equity_curve_indexes(run_path)

    out: list[dict[str, Any]] = []
    for entry in entries:
        charts_rel = str(entry.get("charts_rel") or "").strip().replace("\\", "/")
        chart_files = list(entry.get("chart_files") or [])
        if charts_rel and not chart_files:
            chart_dir = run_path / charts_rel
            if chart_dir.is_dir():
                chart_files = sorted(p.name for p in chart_dir.glob("chart_*.png"))
        charts = []
        if charts_rel.startswith("charts/") and chart_files:
            key = charts_rel.split("/", 1)[1]
            for name in chart_files:
                tf = name.removeprefix("chart_").removesuffix(".png")
                charts.append(
                    {
                        "tf": tf,
                        "file": name,
                        "url": (
                            f"/dashboard/bots/{tenant_slug}/cac40/runs/{run_id}"
                            f"/charts/{key}/{name}"
                        ),
                    }
                )
        dec = entry.get("decision") or {}
        analysis = dec.get("analysis") or {}
        out.append(
            {
                **entry,
                "chart_files": chart_files,
                "charts": charts,
                "bias": analysis.get("bias"),
                "support": analysis.get("support"),
                "resistance": analysis.get("resistance"),
                "actions": dec.get("actions") or [],
                "pnl": _resolve_entry_pnl(entry, by_bar, by_ts),
            }
        )
    out.reverse()
    return out


def resolve_chart_file(
    settings: Settings,
    tenant_slug: str,
    run_id: str,
    chart_key: str,
    filename: str,
) -> Path | None:
    run_path = _run_path(settings, tenant_slug, run_id)
    if run_path is None or not run_path.exists():
        return None
    if not _SAFE_CHART_KEY.match(chart_key or "") or not _SAFE_CHART_FILE.match(filename or ""):
        return None
    path = (run_path / "charts" / chart_key / filename).resolve()
    try:
        path.relative_to((run_path / "charts").resolve())
    except ValueError:
        return None
    return path if path.is_file() else None


def delete_run(settings: Settings, tenant_slug: str, run_id: str) -> bool:
    """Stop a live engine if present, then delete the run directory."""
    stop_run(settings, tenant_slug, run_id)
    path = _run_path(settings, tenant_slug, run_id)
    if path is None or not path.exists():
        return False
    with _LOCK:
        _ENGINES.pop(run_id, None)
        _THREADS.pop(run_id, None)
    shutil.rmtree(path, ignore_errors=True)
    return not path.exists()


def start_run(
    settings: Settings,
    tenant_slug: str,
    *,
    config: Cac40Config,
    ohlc_path: Path,
    api_key: str,
    tenant_id: int | None = None,
    session_factory: SessionFactory | None = None,
) -> str:
    run_path = new_run_dir(runs_dir(settings, tenant_slug))
    (run_path / "config.json").write_text(
        json.dumps(config.public_snapshot(), indent=2, default=str),
        encoding="utf-8",
    )
    engine = BacktestEngine(
        config,
        ohlc_path=ohlc_path,
        run_dir=run_path,
        api_key=api_key,
        tenant_id=tenant_id,
        session_factory=session_factory,
    )

    def _target() -> None:
        try:
            engine.run()
        finally:
            with _LOCK:
                _THREADS.pop(run_path.name, None)
                _ENGINES.pop(run_path.name, None)

    thread = threading.Thread(target=_target, name=f"cac40-{run_path.name}", daemon=True)
    with _LOCK:
        _THREADS[run_path.name] = thread
        _ENGINES[run_path.name] = engine
    thread.start()
    return run_path.name


def stop_run(settings: Settings, tenant_slug: str, run_id: str) -> bool:
    """
    Request stop on a live engine, or mark orphaned runs stopped on disk.

    Orphans happen when the API process restarts (reload/crash) while state.json
    still says ``running`` — there is no in-memory engine to signal.
    """
    with _LOCK:
        engine = _ENGINES.get(run_id)
    if engine:
        engine.request_stop()
        return True

    path = _run_path(settings, tenant_slug, run_id)
    if path is None or not path.exists():
        return False
    state_path = path / "state.json"
    state = _read_json(state_path, default={}) or {}
    if not isinstance(state, dict):
        state = {}
    if state.get("status") in ("running", "stopping", "pending"):
        state["status"] = "stopped"
        note = "Stopped (no active worker — process may have restarted)"
        prev = str(state.get("error") or "").strip()
        state["error"] = f"{prev}\n{note}".strip() if prev else note
        state_path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
        return True
    return False


def default_ohlc_path(settings: Settings, tenant_slug: str) -> Path:
    """
    Per-bot OHLC file (1 bot → 1 symbol → 1 CSV).

    Prefers ``ohlc_15m.csv``; falls back to legacy ``cac40_15m.csv`` when present.
    """
    ohlc_dir = cac40_root(settings, tenant_slug) / "ohlc"
    preferred = ohlc_dir / "ohlc_15m.csv"
    legacy = ohlc_dir / "cac40_15m.csv"
    if preferred.exists():
        return preferred
    if legacy.exists():
        return legacy
    return preferred


def ohlc_info(settings: Settings, tenant_slug: str) -> dict[str, Any]:
    path = default_ohlc_path(settings, tenant_slug)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bars": 0,
        "from": None,
        "to": None,
        "last_candle": None,
        "candle_age_hours": None,
        "size_bytes": 0,
    }
    if not path.exists():
        return info
    info["size_bytes"] = path.stat().st_size
    try:
        from chatbot.cac40.ohlc_store import load_ohlc_csv

        df = load_ohlc_csv(path)
        info["bars"] = int(len(df))
        if not df.empty:
            info["from"] = str(df.index[0])
            info["to"] = str(df.index[-1])
            info["last_candle"] = info["to"]
            info["candle_age_hours"] = _candle_age_hours(df.index[-1], datetime.now(UTC))
    except Exception as exc:  # pragma: no cover
        info["error"] = str(exc)
    return info


def save_ohlc_upload(
    settings: Settings,
    tenant_slug: str,
    *,
    filename: str,
    content: bytes,
    source: str = "evenor",
) -> dict[str, Any]:
    """Validate and store uploaded 15m OHLCV CSV as the tenant default dataset."""
    import tempfile

    from chatbot.cac40.ohlc_store import OHLC_SOURCES, load_ohlc_csv

    if not content:
        raise ValueError("Empty file")
    suffix = Path(filename or "upload.csv").suffix.lower() or ".csv"
    if suffix not in {".csv", ".txt"}:
        raise ValueError("Only CSV files are supported")
    src = (source or "evenor").strip().lower()
    if src not in OHLC_SOURCES:
        raise ValueError(f"Unknown source '{source}'. Choose: {', '.join(OHLC_SOURCES)}")

    dest = default_ohlc_path(settings, tenant_slug)
    dest.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        df = load_ohlc_csv(tmp_path, source=src)
        if df.empty:
            raise ValueError("CSV parsed but contains no bars")
        _write_ohlc_df(dest, df)
    finally:
        tmp_path.unlink(missing_ok=True)

    info = ohlc_info(settings, tenant_slug)
    info["upload_source"] = src
    return info


def ohlc_sync_status_path(settings: Settings, tenant_slug: str) -> Path:
    return default_ohlc_path(settings, tenant_slug).parent / "sync_status.json"


def ohlc_worker_status_path(settings: Settings) -> Path:
    return settings.data_root / "cac40" / "worker_status.json"


def read_ohlc_sync_status(settings: Settings, tenant_slug: str) -> dict[str, Any]:
    path = ohlc_sync_status_path(settings, tenant_slug)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_ohlc_sync_status(
    settings: Settings,
    tenant_slug: str,
    payload: dict[str, Any],
) -> None:
    path = ohlc_sync_status_path(settings, tenant_slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def read_ohlc_worker_status(settings: Settings) -> dict[str, Any]:
    path = ohlc_worker_status_path(settings)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def write_ohlc_worker_status(settings: Settings, payload: dict[str, Any]) -> None:
    path = ohlc_worker_status_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def _candle_age_hours(last_ts: Any, now: datetime) -> float | None:
    if last_ts is None:
        return None
    ts = pd.Timestamp(last_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    clock = pd.Timestamp(now)
    if clock.tzinfo is None:
        clock = clock.tz_localize("UTC")
    return round(float((clock - ts.tz_convert("UTC")).total_seconds() / 3600.0), 2)


def sync_ohlc_from_ig(
    settings: Settings,
    tenant_slug: str,
    *,
    ig_config: dict[str, Any],
    max_gap_days: int = MAX_OHLC_GAP_DAYS,
    allow_bootstrap: bool = True,
    trigger: str = "manual",
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Append IG 15m bars since the last CSV timestamp.

    Raises ValueError for gap > max_gap_days or missing bootstrap when not allowed.
    """
    from chatbot.cac40.ig_ohlc import fetch_ig_ohlc_range
    from chatbot.cac40.ohlc_store import append_bars, load_ohlc_csv

    dest = default_ohlc_path(settings, tenant_slug)
    dest.parent.mkdir(parents=True, exist_ok=True)
    clock = now or datetime.now(UTC)
    if clock.tzinfo is None:
        clock = clock.replace(tzinfo=UTC)
    trigger_label = (trigger or "manual").strip().lower() or "manual"

    def _fail(message: str, *, last_candle: str | None = None) -> None:
        prev = read_ohlc_sync_status(settings, tenant_slug)
        write_ohlc_sync_status(
            settings,
            tenant_slug,
            {
                "ok": False,
                "source": "ig",
                "trigger": trigger_label,
                "added": 0,
                "last_ok_at": prev.get("last_ok_at"),
                "last_candle": last_candle or prev.get("last_candle"),
                "last_error": message,
                "last_error_at": clock.isoformat(),
                "last_attempt_at": clock.isoformat(),
            },
        )

    try:
        existing: pd.DataFrame | None = None
        last_ts: pd.Timestamp | None = None
        if dest.exists() and dest.stat().st_size > 0:
            existing = load_ohlc_csv(dest)
            if existing is not None and not existing.empty:
                last_ts = pd.Timestamp(existing.index[-1])

        last_candle_before = str(last_ts) if last_ts is not None else None
        if last_ts is None:
            if not allow_bootstrap:
                raise ValueError(
                    "OHLC CSV missing — upload history or run manual Sync from IG once"
                )
            start = pd.Timestamp(clock) - pd.Timedelta(days=BOOTSTRAP_LOOKBACK_DAYS)
        else:
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("UTC")
            age = pd.Timestamp(clock) - last_ts.tz_convert("UTC")
            if age > pd.Timedelta(days=max_gap_days):
                raise ValueError(
                    f"OHLC gap is {age.days} days (max {max_gap_days}). "
                    "Re-upload a BacktestMarket CSV, then sync again."
                )
            start = last_ts + pd.Timedelta(minutes=15)

        end = pd.Timestamp(clock)
        bars_before = int(len(existing)) if existing is not None else 0
        added = 0
        if start < end:
            fresh = fetch_ig_ohlc_range(ig_config, start=start, end=end)
            if last_ts is not None and not fresh.empty:
                fresh = fresh.loc[fresh.index > last_ts.tz_convert(fresh.index.tz)]
            if not fresh.empty:
                append_bars(dest, fresh)
                added = int(len(fresh))
        info = ohlc_info(settings, tenant_slug)
        last_candle = info.get("to")
        status = {
            "ok": True,
            "source": "ig",
            "trigger": trigger_label,
            "added": added,
            "bars_before": bars_before,
            "bars": info.get("bars"),
            "from_ts": info.get("from"),
            "to_ts": info.get("to"),
            "last_candle": last_candle,
            "last_candle_before": last_candle_before,
            "candle_age_hours": _candle_age_hours(last_candle, clock),
            "fetch_from": str(start),
            "fetch_to": str(end),
            "last_ok_at": clock.isoformat(),
            "last_attempt_at": clock.isoformat(),
            "last_error": None,
        }
        write_ohlc_sync_status(settings, tenant_slug, status)
        info.update(status)
        return info
    except Exception as exc:
        last_candle = None
        try:
            info = ohlc_info(settings, tenant_slug)
            last_candle = info.get("to")
        except Exception:
            last_candle = None
        _fail(str(exc), last_candle=last_candle)
        raise


def run_due_ig_ohlc_syncs(session: Session, settings: Settings) -> list[str]:
    """
    Top up OHLC for tenants with active CAC40 + IG connector + existing CSV.

    Never raises out of the loop; returns log lines.
    """
    from chatbot.adapters.persistence.integration_repository import SqlAlchemyIntegrationRepository
    from chatbot.adapters.persistence.tenant_repository import SqlAlchemyTenantRepository
    from chatbot.application.connector_service import ConnectorService
    from chatbot.adapters.persistence.connector_repository import SqlAlchemyConnectorRepository
    from chatbot.application.tenant_service import TenantService
    from chatbot.domain.models.integration import IntegrationType

    started = datetime.now(UTC)
    repo = SqlAlchemyIntegrationRepository(session)
    tenant_svc = TenantService(SqlAlchemyTenantRepository(session))
    connector_svc = ConnectorService(SqlAlchemyConnectorRepository(session))
    logs: list[str] = []
    ok_count = 0
    fail_count = 0
    skip_count = 0

    for integration in repo.list_active_by_type(IntegrationType.CAC40_BACKTEST):
        tenant = tenant_svc.get_by_id(integration.tenant_id)
        if tenant is None:
            continue
        slug = tenant.slug
        from chatbot.application.cac40_live_service import resolve_primary_ig_config

        ig_config = resolve_primary_ig_config(
            settings, slug, session=session, tenant_id=tenant.id
        )
        if not ig_config:
            skip_count += 1
            logs.append(f"{slug}: skip — no active IG connector")
            continue
        dest = default_ohlc_path(settings, slug)
        if not dest.exists():
            skip_count += 1
            logs.append(f"{slug}: skip — OHLC CSV missing (no cron bootstrap)")
            continue
        try:
            info = sync_ohlc_from_ig(
                settings,
                slug,
                ig_config=ig_config,
                allow_bootstrap=False,
                trigger="worker",
            )
            ok_count += 1
            logs.append(
                f"{slug}: ok — added {info.get('added', 0)} bars "
                f"(total {info.get('bars', 0)}; last candle {info.get('last_candle')})"
            )
        except Exception as exc:
            fail_count += 1
            logger.warning("IG OHLC sync failed for %s: %s", slug, exc)
            logs.append(f"{slug}: failed — {exc}")

    finished = datetime.now(UTC)
    write_ohlc_worker_status(
        settings,
        {
            "ok": fail_count == 0,
            "started_at": started.isoformat(),
            "finished_at": finished.isoformat(),
            "poll_seconds": settings.cac40_ohlc_poll_seconds,
            "tenants_ok": ok_count,
            "tenants_failed": fail_count,
            "tenants_skipped": skip_count,
            "logs": logs[-50:],
        },
    )
    return logs


def _write_ohlc_df(dest: Path, df) -> None:
    out = df.reset_index()
    ts_col = out.columns[0]
    out = out.rename(
        columns={
            ts_col: "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    out.to_csv(dest, index=False)
