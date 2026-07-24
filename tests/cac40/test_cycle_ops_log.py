from chatbot.application.cac40_cycle_ops_log import (
    build_cycle_ops_log,
    ops_log_line_count,
)
from chatbot.cac40.ig_connector import compact_ig_error


def test_build_cycle_ops_log_backtest_gate_only():
    sections = build_cycle_ops_log(
        {
            "llm_mode": "live",
            "executed": ["place_limit:o1@8300.0"],
            "rejected": ["place_stop:loss_exit_blocked"],
        }
    )
    by_title = {s["title"]: s["lines"] for s in sections}
    assert "Trigger" in by_title
    assert any("llm_mode=live" in line for line in by_title["Trigger"])
    assert "RiskGate" in by_title
    assert "OK place_limit:o1@8300.0" in by_title["RiskGate"]
    assert "REJECT place_stop:loss_exit_blocked" in by_title["RiskGate"]
    assert "Mirror / IG" not in by_title
    assert ops_log_line_count(sections) >= 3


def test_build_cycle_ops_log_live_mirror_and_reconcile():
    sections = build_cycle_ops_log(
        {
            "dry_run": False,
            "llm_trigger": {"reason": "interval", "every_bars": 1},
            "executed": ["amend_order:o48->8410.0x2"],
            "rejected": [],
            "mirror": [
                {
                    "connector_id": 5,
                    "placed": [],
                    "cancelled": [
                        {
                            "order_id": "o47",
                            "deal_id": "WO1",
                            "deal_status": "ACCEPTED",
                        }
                    ],
                    "amended": [
                        {
                            "order_id": "o48",
                            "deal_id": "WO2",
                            "level": 8410.0,
                            "deal_status": "ACCEPTED",
                        }
                    ],
                    "errors": [
                        "place:o50:IG place_working_order failed: HTTP 400\n"
                        "IG errorCode: validation.invalid.input"
                    ],
                }
            ],
            "reconcile": {
                "ran": True,
                "repaired": True,
                "closed": [{"id": "p1"}],
                "opened": [],
                "imported": ["o9"],
                "warnings": ["secondary_desync:x"],
            },
            "ohlc_feed": {"stale": False, "allowance": {"remainingAllowance": 9, "totalAllowance": 10}},
            "fill_events": [{"type": "sync", "opened": ["p2"]}],
        }
    )
    by_title = {s["title"]: s["lines"] for s in sections}
    assert any("llm_trigger" in line for line in by_title["Trigger"])
    assert any("OK amend_order:o48->8410.0x2" in line for line in by_title["RiskGate"])
    mirror = by_title["Mirror / IG"]
    assert any("CANCEL o47" in line and "status=ACCEPTED" in line for line in mirror)
    assert any("AMEND o48" in line and "level=8410.0" in line for line in mirror)
    assert any("ERROR" in line and "errorCode=validation.invalid.input" in line for line in mirror)
    assert any("reconcile" in line and "closed=1" in line for line in by_title["Book sync"])
    assert any("allowance=9/10" in line for line in by_title["OHLC / flatten"])
    assert any("count=1" in line for line in by_title["Fills"])


def test_compact_ig_error_extracts_code_and_status():
    exc = RuntimeError(
        "IG place_working_order failed: HTTP 403\n"
        "URL: https://demo-api.ig.com/gateway/deal/workingorders/otc\n"
        "IG errorCode: api-key-disabled\n"
        "Hint: API key may be disabled"
    )
    out = compact_ig_error(exc)
    assert out["http_status"] == 403
    assert out["error_code"] == "api-key-disabled"
    assert "HTTP 403" in out["error"] or "place_working_order" in out["error"]
