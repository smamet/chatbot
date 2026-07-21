from chatbot.application.connector_test_service import run_ig_working_order_test


def test_ig_working_order_test_blocks_live() -> None:
    result = run_ig_working_order_test(
        {
            "api_key": "key",
            "username": "user",
            "password": "pass",
            "acc_type": "LIVE",
            "epic": "IX.D.CAC.DAILY.IP",
        }
    )
    assert result.ok is False
    assert result.error == "live_blocked"
    assert "DEMO-only" in result.message


def test_ig_working_order_test_requires_credentials() -> None:
    result = run_ig_working_order_test({"acc_type": "DEMO"})
    assert result.ok is False
    assert result.error == "missing_credentials"
