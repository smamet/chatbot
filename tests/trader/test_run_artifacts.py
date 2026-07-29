import json
from pathlib import Path

from chatbot.application.trader_backtest_service import (
    delete_run,
    get_run,
    resolve_chart_file,
    runs_dir,
)
from chatbot.trader.chart_renderer import render_ohlc_chart
from chatbot.config.settings import Settings
import pandas as pd


def test_matplotlib_renders_png(tmp_path: Path) -> None:
    idx = pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100 + i * 0.1 for i in range(20)],
            "high": [101 + i * 0.1 for i in range(20)],
            "low": [99 + i * 0.1 for i in range(20)],
            "close": [100.5 + i * 0.1 for i in range(20)],
        },
        index=idx,
    )
    out = tmp_path / "chart_15m.png"
    png = render_ohlc_chart(df, title="test", support=99.5, resistance=102.0, out_path=out)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert out.exists()


def test_get_run_exposes_chart_urls_and_delete_cleans(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    run_id = "run_test_charts"
    run_path = runs_dir(settings, "demo-bot") / run_id
    chart_key = "20240102_090000"
    chart_dir = run_path / "charts" / chart_key
    chart_dir.mkdir(parents=True)
    (chart_dir / "chart_15m.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (chart_dir / "chart_1H.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (run_path / "state.json").write_text(json.dumps({"status": "done", "progress": 1.0}))
    (run_path / "report.json").write_text(
        json.dumps(
            {
                "equity_curve": [
                    {
                        "bar": 1,
                        "ts": "2024-01-02 09:00:00+01:00",
                        "realized": 12.5,
                        "net_upl": -3.25,
                        "equity": 9.25,
                    }
                ]
            }
        )
    )
    (run_path / "decisions_log.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2024-01-02 09:00:00+01:00",
                    "bar": 0,
                    "charts_rel": f"charts/{chart_key}",
                    "chart_files": ["chart_15m.png", "chart_1H.png"],
                    "decision": {
                        "analysis": {
                            "bias": "long_from_support",
                            "support": 100,
                            "resistance": 110,
                        },
                        "actions": [{"op": "place_limit", "side": "BUY", "level": 100}],
                    },
                    "executed": ["place_limit:x@100"],
                    "rejected": [],
                }
            ]
        )
    )

    run = get_run(settings, "demo-bot", run_id)
    assert len(run["decisions"]) == 1
    charts = run["decisions"][0]["charts"]
    assert len(charts) == 2
    assert charts[0]["url"].endswith(f"/charts/{chart_key}/chart_15m.png")
    assert run["decisions"][0]["pnl"] == {
        "realized": 12.5,
        "net_upl": -3.25,
        "equity": 9.25,
    }

    resolved = resolve_chart_file(settings, "demo-bot", run_id, chart_key, "chart_15m.png")
    assert resolved is not None and resolved.exists()
    assert resolve_chart_file(settings, "demo-bot", run_id, "../x", "chart_15m.png") is None

    assert delete_run(settings, "demo-bot", run_id) is True
    assert not run_path.exists()


def test_get_run_tolerates_empty_decisions_log(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    run_id = "run_empty_log"
    run_path = runs_dir(settings, "demo-bot") / run_id
    run_path.mkdir(parents=True)
    (run_path / "state.json").write_text(json.dumps({"status": "running", "progress": 0.1}))
    (run_path / "report.json").write_text("{}")
    (run_path / "decisions_log.json").write_text("")  # truncated / mid-write

    run = get_run(settings, "demo-bot", run_id)
    assert run["state"]["status"] == "running"
    assert run["decisions"] == []


def test_stop_run_marks_orphan_stopped(tmp_path: Path) -> None:
    from chatbot.application.trader_backtest_service import stop_run

    settings = Settings(data_root=tmp_path)
    run_id = "run_orphan"
    run_path = runs_dir(settings, "demo-bot") / run_id
    run_path.mkdir(parents=True)
    (run_path / "state.json").write_text(json.dumps({"status": "running", "progress": 0.4}))

    assert stop_run(settings, "demo-bot", run_id) is True
    state = json.loads((run_path / "state.json").read_text())
    assert state["status"] == "stopped"
    assert "no active worker" in state["error"]


def test_force_delete_running_orphan(tmp_path: Path) -> None:
    settings = Settings(data_root=tmp_path)
    run_id = "run_force_del"
    run_path = runs_dir(settings, "demo-bot") / run_id
    run_path.mkdir(parents=True)
    (run_path / "state.json").write_text(json.dumps({"status": "running", "progress": 0.2}))
    (run_path / "decisions_log.json").write_text("[]")

    assert delete_run(settings, "demo-bot", run_id) is True
    assert not run_path.exists()
