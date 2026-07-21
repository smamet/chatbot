from __future__ import annotations

import json
import re
import shutil
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from chatbot.cac40.backtest_engine import BacktestEngine, new_run_dir
from chatbot.cac40.config import Cac40Config, public_config_snapshot
from chatbot.config.settings import Settings

SessionFactory = Callable[[], Session]

_SAFE_RUN_ID = re.compile(r"^[\w.-]+$")
_SAFE_CHART_KEY = re.compile(r"^[\w.-]+$")
_SAFE_CHART_FILE = re.compile(r"^chart_[\w.-]+\.png$")


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
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        raw_cfg: dict[str, Any] = {}
        if config_path.exists():
            loaded = json.loads(config_path.read_text(encoding="utf-8"))
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
    state = json.loads((path / "state.json").read_text()) if (path / "state.json").exists() else {}
    report = json.loads((path / "report.json").read_text()) if (path / "report.json").exists() else {}
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
    if not report_path.exists():
        return by_bar, by_ts
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
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
    if log_path.exists():
        raw = json.loads(log_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            entries = list(raw)
    if not entries and (run_path / "report.json").exists():
        report = json.loads((run_path / "report.json").read_text(encoding="utf-8"))
        if isinstance(report.get("decisions"), list):
            entries = list(report["decisions"])
    if not entries and (run_path / "decisions.json").exists():
        cached = json.loads((run_path / "decisions.json").read_text(encoding="utf-8"))
        for ts, payload in cached.items():
            meta = payload.get("meta") or {}
            entries.append(
                {
                    "ts": ts,
                    "decision": payload.get("decision"),
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
    """Delete a backtest run directory (charts, decisions, report)."""
    with _LOCK:
        if run_id in _ENGINES:
            _ENGINES[run_id].request_stop()
    path = _run_path(settings, tenant_slug, run_id)
    if path is None or not path.exists():
        return False
    shutil.rmtree(path)
    return True


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


def stop_run(run_id: str) -> bool:
    with _LOCK:
        engine = _ENGINES.get(run_id)
    if not engine:
        return False
    engine.request_stop()
    return True


def default_ohlc_path(settings: Settings, tenant_slug: str) -> Path:
    return cac40_root(settings, tenant_slug) / "ohlc" / "cac40_15m.csv"


def ohlc_info(settings: Settings, tenant_slug: str) -> dict[str, Any]:
    path = default_ohlc_path(settings, tenant_slug)
    info: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "bars": 0,
        "from": None,
        "to": None,
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


def fetch_and_store_yahoo_ohlc(
    settings: Settings,
    tenant_slug: str,
    *,
    period: str = "60d",
) -> dict[str, Any]:
    """Download ^FCHI 15m from Yahoo and store as tenant default CSV."""
    from chatbot.cac40.yahoo_ohlc import fetch_yahoo_ohlc, yahoo_source_meta

    dest = default_ohlc_path(settings, tenant_slug)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = fetch_yahoo_ohlc(period=period)
    _write_ohlc_df(dest, df)
    info = ohlc_info(settings, tenant_slug)
    info["source"] = yahoo_source_meta()
    return info


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
