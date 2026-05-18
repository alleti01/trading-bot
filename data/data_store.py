"""Persisted candle store.

Day 2 only stores OHLCV. Feature snapshots and setups arrive on Day 3+.

The upsert path uses SQLite's ``INSERT ... ON CONFLICT DO NOTHING`` against
the unique index on ``(instrument, timeframe, ts)``. When we move to
Postgres we'll swap the dialect import; the call site doesn't change.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.logging_config import get_logger
from storage.db import session_scope
from storage.tables import Candle


def _to_utc(dt: datetime) -> datetime:
    """SQLite stores tz-aware times as UTC ISO strings, so we normalize first."""
    if dt.tzinfo is None:
        raise ValueError("Refusing to store naive datetime; tz-aware required.")
    return dt.astimezone(timezone.utc)


def upsert_candles(df: pd.DataFrame, instrument: str, timeframe: str) -> int:
    """Insert candles, ignoring rows that conflict on (instrument, timeframe, ts).

    Expects the same shape ``load_ohlcv_csv`` returns: tz-aware DatetimeIndex
    plus float OHLCV columns. Returns the number of new rows actually written.
    """
    log = get_logger("data.data_store")
    if df.empty:
        return 0

    if df.index.tz is None:
        raise ValueError("upsert_candles requires a tz-aware DatetimeIndex.")

    rows = [
        {
            "instrument": instrument,
            "timeframe": timeframe,
            "ts": _to_utc(ts.to_pydatetime()),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
        }
        for ts, row in df[["open", "high", "low", "close", "volume"]].iterrows()
    ]

    written = 0
    with session_scope() as session:
        before = session.execute(
            select(Candle).where(
                Candle.instrument == instrument,
                Candle.timeframe == timeframe,
            )
        ).all()
        n_before = len(before)

        stmt = sqlite_insert(Candle).values(rows).on_conflict_do_nothing(
            index_elements=["instrument", "timeframe", "ts"]
        )
        session.execute(stmt)
        session.flush()

        after = session.execute(
            select(Candle).where(
                Candle.instrument == instrument,
                Candle.timeframe == timeframe,
            )
        ).all()
        written = len(after) - n_before

    log.info(
        "candles.upserted",
        instrument=instrument,
        timeframe=timeframe,
        attempted=len(rows),
        written=written,
        skipped_existing=len(rows) - written,
    )
    return written


def load_candles(
    instrument: str,
    timeframe: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> pd.DataFrame:
    """Load candles back into the same shape ``load_ohlcv_csv`` returns."""
    stmt = select(Candle).where(
        Candle.instrument == instrument,
        Candle.timeframe == timeframe,
    )
    if start is not None:
        stmt = stmt.where(Candle.ts >= _to_utc(start))
    if end is not None:
        stmt = stmt.where(Candle.ts <= _to_utc(end))
    stmt = stmt.order_by(Candle.ts.asc())

    with session_scope() as session:
        rows = session.execute(stmt).scalars().all()

    if not rows:
        return pd.DataFrame(
            columns=["instrument", "timeframe", "open", "high", "low", "close", "volume"]
        ).set_index(pd.DatetimeIndex([], tz="UTC", name="timestamp"))

    df = pd.DataFrame(
        {
            "timestamp": [r.ts for r in rows],
            "instrument": [r.instrument for r in rows],
            "timeframe": [r.timeframe for r in rows],
            "open": [r.open for r in rows],
            "high": [r.high for r in rows],
            "low": [r.low for r in rows],
            "close": [r.close for r in rows],
            "volume": [r.volume for r in rows],
        }
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    return df
