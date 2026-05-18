"""RollingCSVFeed + SyntheticLiveFeed: poll behavior, window growth, exhaustion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.market_data_service import RollingCSVFeed, SyntheticLiveFeed


def _write_csv(tmp_path: Path, n_rows: int = 5) -> Path:
    idx = pd.date_range("2024-01-15 09:30", periods=n_rows, freq="1min", tz="America/New_York")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": [100 + i for i in range(n_rows)],
            "high": [101 + i for i in range(n_rows)],
            "low": [99 + i for i in range(n_rows)],
            "close": [100.5 + i for i in range(n_rows)],
            "volume": [1000.0] * n_rows,
        }
    )
    out = tmp_path / "ohlcv.csv"
    df.to_csv(out, index=False)
    return out


def test_rolling_csv_feed_emits_one_bar_per_poll(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, n_rows=5)
    feed = RollingCSVFeed(
        path=csv, instrument="MES", timeframe="1m",
        tz="America/New_York", window_bars=3,
    )

    seen = []
    while not feed.is_exhausted():
        result = feed.poll_latest()
        assert result.new_candle is not None
        seen.append(result.new_candle)
        assert len(result.rolling_window) <= 3

    assert len(seen) == 5


def test_rolling_csv_feed_returns_none_after_exhaustion(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, n_rows=2)
    feed = RollingCSVFeed(
        path=csv, instrument="MES", timeframe="1m",
        tz="America/New_York", window_bars=10,
    )
    feed.poll_latest()
    feed.poll_latest()
    assert feed.is_exhausted()
    after = feed.poll_latest()
    assert after.new_candle is None
    # Window still has the trailing bars.
    assert len(after.rolling_window) == 2


def test_rolling_window_grows_to_window_bars(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, n_rows=5)
    feed = RollingCSVFeed(
        path=csv, instrument="MES", timeframe="1m",
        tz="America/New_York", window_bars=3,
    )
    sizes = []
    while not feed.is_exhausted():
        sizes.append(len(feed.poll_latest().rolling_window))
    # Should grow 1, 2, 3, 3, 3.
    assert sizes == [1, 2, 3, 3, 3]


def test_synthetic_live_feed_is_deterministic() -> None:
    f1 = SyntheticLiveFeed(max_bars=10, window_bars=5, seed=7)
    f2 = SyntheticLiveFeed(max_bars=10, window_bars=5, seed=7)
    seq1 = [f1.poll_latest().new_candle.close for _ in range(10)]
    seq2 = [f2.poll_latest().new_candle.close for _ in range(10)]
    assert seq1 == seq2


def test_synthetic_live_feed_exhausts_after_max_bars() -> None:
    feed = SyntheticLiveFeed(max_bars=3, window_bars=5)
    for _ in range(3):
        result = feed.poll_latest()
        assert result.new_candle is not None
    assert feed.is_exhausted()
    assert feed.poll_latest().new_candle is None
