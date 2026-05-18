"""OHLCV CSV loader.

Convention: any naive ``timestamp`` column is **assumed UTC** and converted
to the configured tz on load. This is the safest assumption for vendor
exports — getting it wrong produces silent off-by-N-hours bugs in features.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.logging_config import get_logger

REQUIRED_COLUMNS: tuple[str, ...] = ("timestamp", "open", "high", "low", "close", "volume")
NUMERIC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def load_ohlcv_csv(
    path: Path,
    instrument: str,
    timeframe: str,
    tz: str,
) -> pd.DataFrame:
    """Load an OHLCV CSV and return a tz-aware, deduplicated, sorted DataFrame.

    Returned DataFrame is indexed by ``timestamp`` (tz-aware) and has
    columns ``open, high, low, close, volume`` (float64) plus metadata
    columns ``instrument`` and ``timeframe``.
    """
    log = get_logger("data.csv_loader")
    path = Path(path)

    raw = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV {path} missing required columns: {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}"
        )

    df = raw[list(REQUIRED_COLUMNS)].copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert(tz)

    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    n_raw = len(df)

    n_bad_ts = int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    n_nan_rows = int(df[list(NUMERIC_COLUMNS)].isna().any(axis=1).sum())
    df = df.dropna(subset=list(NUMERIC_COLUMNS))

    df = df.sort_values("timestamp", kind="mergesort")
    n_before_dedupe = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    n_dupes = n_before_dedupe - len(df)

    df = df.set_index("timestamp")
    df.index.name = "timestamp"

    df["instrument"] = instrument
    df["timeframe"] = timeframe

    df = df[["instrument", "timeframe", "open", "high", "low", "close", "volume"]]

    log.info(
        "csv.loaded",
        path=str(path),
        instrument=instrument,
        timeframe=timeframe,
        tz=tz,
        rows_in=n_raw,
        rows_out=len(df),
        bad_timestamps_dropped=n_bad_ts,
        nan_rows_dropped=n_nan_rows,
        duplicates_dropped=n_dupes,
        first_ts=str(df.index.min()) if len(df) else None,
        last_ts=str(df.index.max()) if len(df) else None,
    )

    return df
