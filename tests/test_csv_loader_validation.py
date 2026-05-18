"""``load_ohlcv_csv`` rejects geometrically-impossible candles.

Bad data poisons feature engineering silently. This test pins the OHLC
consistency check so it can't regress: rows with high < low, high <
max(open,close), low > min(open,close), or volume < 0 must be dropped.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.csv_loader import load_ohlcv_csv


def _write_csv(tmp_path: Path, rows: list[dict]) -> Path:
    df = pd.DataFrame(rows)
    p = tmp_path / "ohlcv.csv"
    df.to_csv(p, index=False)
    return p


def test_drops_rows_with_high_below_low(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        [
            {"timestamp": "2024-01-15 09:30:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
            # Geometric impossibility: high < low.
            {"timestamp": "2024-01-15 09:31:00", "open": 100, "high": 99, "low": 101, "close": 100, "volume": 1000},
            {"timestamp": "2024-01-15 09:32:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
        ],
    )
    df = load_ohlcv_csv(p, instrument="MES", timeframe="1m", tz="UTC")
    assert len(df) == 2


def test_drops_rows_with_high_below_open_close(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        [
            {"timestamp": "2024-01-15 09:30:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
            # high < max(open, close) = 100
            {"timestamp": "2024-01-15 09:31:00", "open": 100, "high": 99.5, "low": 98, "close": 100, "volume": 1000},
        ],
    )
    df = load_ohlcv_csv(p, instrument="MES", timeframe="1m", tz="UTC")
    assert len(df) == 1


def test_drops_rows_with_negative_volume(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        [
            {"timestamp": "2024-01-15 09:30:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
            {"timestamp": "2024-01-15 09:31:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": -5},
        ],
    )
    df = load_ohlcv_csv(p, instrument="MES", timeframe="1m", tz="UTC")
    assert len(df) == 1


def test_keeps_well_formed_rows(tmp_path: Path) -> None:
    p = _write_csv(
        tmp_path,
        [
            {"timestamp": "2024-01-15 09:30:00", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000},
            {"timestamp": "2024-01-15 09:31:00", "open": 100, "high": 102, "low": 99, "close": 101, "volume": 500},
        ],
    )
    df = load_ohlcv_csv(p, instrument="MES", timeframe="1m", tz="UTC")
    assert len(df) == 2
