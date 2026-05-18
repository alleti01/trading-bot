"""CSV loader behavior."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from data.csv_loader import load_ohlcv_csv


def _write_csv(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "ohlcv.csv"
    p.write_text(body)
    return p


def test_load_basic(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15T09:30:00+00:00,100,101,99,100.5,1000\n"
        "2024-01-15T09:31:00+00:00,100.5,101.5,100,101,1100\n",
    )
    df = load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="America/New_York")
    assert len(df) == 2
    assert df.index.tz is not None
    assert str(df.index.tz) == "America/New_York"
    assert df["open"].dtype == "float64"
    assert list(df.columns) == ["instrument", "timeframe", "open", "high", "low", "close", "volume"]


def test_naive_timestamps_assumed_utc(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15 14:30:00,100,101,99,100.5,1000\n",
    )
    df = load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="America/New_York")
    # 14:30 UTC → 09:30 NY (in January, EST is UTC-5).
    only = df.index[0]
    assert only.hour == 9 and only.minute == 30


def test_dedupe_keeps_last(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15T09:30:00+00:00,100,101,99,100.5,1000\n"
        "2024-01-15T09:30:00+00:00,200,201,199,200.5,9999\n",  # duplicate ts
    )
    df = load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="UTC")
    assert len(df) == 1
    assert df["open"].iloc[0] == 200.0
    assert df["volume"].iloc[0] == 9999.0


def test_sort_out_of_order(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15T09:32:00+00:00,3,4,2,3.5,1\n"
        "2024-01-15T09:30:00+00:00,1,2,0.5,1.5,1\n"
        "2024-01-15T09:31:00+00:00,2,3,1.5,2.5,1\n",
    )
    df = load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="UTC")
    assert df.index.is_monotonic_increasing
    assert df["open"].tolist() == [1.0, 2.0, 3.0]


def test_drops_nan_rows(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,low,close,volume\n"
        "2024-01-15T09:30:00+00:00,100,101,99,100.5,1000\n"
        "2024-01-15T09:31:00+00:00,,101.5,100,101,1100\n",  # NaN open
    )
    df = load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="UTC")
    assert len(df) == 1


def test_rejects_missing_columns(tmp_path: Path) -> None:
    csv = _write_csv(
        tmp_path,
        "timestamp,open,high,close,volume\n"  # no 'low'
        "2024-01-15T09:30:00+00:00,100,101,100.5,1000\n",
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_ohlcv_csv(csv, instrument="MES", timeframe="1m", tz="UTC")
