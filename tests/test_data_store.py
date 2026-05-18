"""Round-trip CSV → upsert_candles → load_candles."""

from __future__ import annotations

from pathlib import Path

from data.csv_loader import load_ohlcv_csv
from data.data_store import load_candles, upsert_candles
from storage.db import init_db


def _write_csv(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "ohlcv.csv"
    p.write_text(body)
    return p


def test_upsert_and_load_round_trip(tmp_path: Path) -> None:
    init_db()
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15T09:30:00+00:00,100,101,99,100.5,1000\n"
        "2024-01-15T09:31:00+00:00,100.5,101.5,100,101,1100\n",
    )
    df = load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="UTC")
    written = upsert_candles(df, instrument="MES", timeframe="1m")
    assert written == 2

    # Reinsert same rows: should write 0.
    again = upsert_candles(df, instrument="MES", timeframe="1m")
    assert again == 0

    out = load_candles(instrument="MES", timeframe="1m")
    assert len(out) == 2
    assert out["close"].iloc[0] == 100.5


def test_market_data_feed_yields_candles(tmp_path: Path) -> None:
    """CSVReplayFeed yields Pydantic-validated candles in order."""
    from data.market_data_service import CSVReplayFeed

    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15T09:30:00+00:00,100,101,99,100.5,1000\n"
        "2024-01-15T09:31:00+00:00,100.5,101.5,100,101,1100\n",
    )
    feed = CSVReplayFeed(csv, instrument="MES", timeframe="1m", tz="UTC")
    bars = list(feed.iter_bars())
    assert len(bars) == 2
    assert bars[0].close == 100.5
    assert bars[1].ts > bars[0].ts
