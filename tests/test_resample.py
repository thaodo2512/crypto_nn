import pandas as pd

from utils_cg import resample_15m_ohlcv, ensure_unique_key


def test_resample_15m_right_closed_right_label():
    ts = pd.date_range("2024-01-01T00:00:00Z", periods=60, freq="1min")
    df = pd.DataFrame(
        {
            "ts": ts,
            "open": range(60),
            "high": [x + 1 for x in range(60)],
            "low": [max(0, x - 1) for x in range(60)],
            "close": range(60),
            "volume_usd": [1.0] * 60,
        }
    )
    out = resample_15m_ohlcv(df)
    expected_ts = pd.to_datetime(
        [
            "2024-01-01T00:15:00Z",
            "2024-01-01T00:30:00Z",
            "2024-01-01T00:45:00Z",
            "2024-01-01T01:00:00Z",
        ],
        utc=True,
    )
    assert list(out["ts"]) == list(expected_ts)
    # Check volume aggregation: 15 mins → sum = 15
    assert (out["volume_usd"].iloc[0]) == 15.0


def test_ensure_unique_key():
    ts = pd.to_datetime(["2024-01-01T00:15:00Z", "2024-01-01T00:15:00Z"], utc=True)
    df = pd.DataFrame({"symbol": ["BTCUSDT", "BTCUSDT"], "ts": ts, "x": [1, 2]})
    try:
        ensure_unique_key(df, ["symbol", "ts"])  # should raise
        raised = False
    except ValueError:
        raised = True
    assert raised, "Expected ValueError for duplicate keys"

