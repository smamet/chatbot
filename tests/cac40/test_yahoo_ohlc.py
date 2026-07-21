from chatbot.cac40.yahoo_ohlc import fetch_yahoo_ohlc, yahoo_source_meta


def test_yahoo_source_meta():
    meta = yahoo_source_meta()
    assert meta["ticker"] == "^FCHI"
    assert meta["interval"] == "15m"


def test_fetch_yahoo_ohlc_live():
    """Integration-ish: hits Yahoo. Skip soft-fail if network blocked."""
    try:
        df = fetch_yahoo_ohlc(period="5d")
    except Exception as exc:  # pragma: no cover
        import pytest

        pytest.skip(f"Yahoo unavailable: {exc}")
    assert len(df) > 10
    assert {"open", "high", "low", "close"}.issubset(df.columns)
