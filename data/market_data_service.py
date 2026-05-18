"""Market-data feeds.

Day 2 shipped batch replay. Day 5 adds **incremental polling** so the
24/7 paper service can ask "is there a new bar?" once per cron tick.

The two interfaces are intentionally side-by-side:

- :class:`MarketDataFeed`        — batch iterator (used by the backtester).
- :class:`IncrementalFeed`       — poll-based, used by ``PaperTradingLoop``.

Day 5 does **not** implement a real broker/exchange WebSocket. The two
incremental feeds available are:

- :class:`RollingCSVFeed`     — replays a CSV in chronological order, one
  bar per ``poll_latest`` call. Useful for paper smoke runs and tests.
- :class:`SyntheticLiveFeed`  — generates deterministic synthetic bars on
  demand. Useful for unit tests that don't want to hit disk.

Both maintain a rolling in-memory window so the loop can recompute
features without reading the entire history every tick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.candle import Candle
from data.csv_loader import load_ohlcv_csv


# ---------------------------------------------------------------------------
# Batch (Day 2)
# ---------------------------------------------------------------------------
class MarketDataFeed(ABC):
    @abstractmethod
    def iter_bars(self) -> Iterator[Candle]:
        """Yield candles in chronological order."""


class CSVReplayFeed(MarketDataFeed):
    """Replays a CSV file as a stream of ``Candle`` objects.

    Useful for backtests, paper-mode dry runs, and fast deterministic tests.
    """

    def __init__(self, path: Path | str, instrument: str, timeframe: str, tz: str) -> None:
        self.path = Path(path)
        self.instrument = instrument
        self.timeframe = timeframe
        self.tz = tz

    def iter_bars(self) -> Iterator[Candle]:
        df = load_ohlcv_csv(self.path, self.instrument, self.timeframe, self.tz)
        for ts, row in df.iterrows():
            yield Candle(
                instrument=self.instrument,
                timeframe=self.timeframe,
                ts=ts.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )


# ---------------------------------------------------------------------------
# Incremental (Day 5)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PollResult:
    """One scan tick. ``new_candle`` is None when nothing has changed."""

    new_candle: Optional[Candle]
    rolling_window: pd.DataFrame  # last N bars (inclusive of new_candle if any)


class IncrementalFeed(ABC):
    """Bar-by-bar live-shaped feed for the paper service.

    The contract is intentionally narrow: ``poll_latest()`` returns a
    :class:`PollResult` whose ``new_candle`` is non-None only on the tick
    when a new bar has actually appeared. The rolling DataFrame has the
    same shape as ``data.csv_loader.load_ohlcv_csv`` returns: tz-aware
    DatetimeIndex, columns ``open, high, low, close, volume``.
    """

    instrument: str
    timeframe: str

    @abstractmethod
    def poll_latest(self) -> PollResult: ...

    @abstractmethod
    def is_exhausted(self) -> bool: ...


# ---------------------------------------------------------------------------
# RollingCSVFeed — drains a CSV one bar per poll()
# ---------------------------------------------------------------------------
class RollingCSVFeed(IncrementalFeed):
    """Walks an OHLCV CSV one bar per ``poll_latest`` call.

    On each call, advances by one row and returns a window of the most
    recent ``window_bars`` bars. When the CSV is exhausted, ``is_exhausted``
    returns True and subsequent polls yield ``new_candle=None`` with the
    final window.

    The CSV is read once at construction and held in memory. For Day 5
    sizes (a few thousand bars) this is fine; a real live feed will plug
    a broker/exchange WebSocket into this same interface later.
    """

    def __init__(
        self,
        path: Path | str,
        *,
        instrument: str,
        timeframe: str,
        tz: str,
        window_bars: int = 500,
        warmup_bars: int = 0,
    ) -> None:
        if window_bars < 1:
            raise ValueError("window_bars must be >= 1")
        if warmup_bars < 0:
            raise ValueError("warmup_bars must be >= 0")

        self.path = Path(path)
        self.instrument = instrument
        self.timeframe = timeframe
        self.tz = tz
        self.window_bars = int(window_bars)

        df = load_ohlcv_csv(self.path, instrument, timeframe, tz)
        if df.empty:
            raise ValueError(f"CSV {self.path} loaded zero rows")

        self._all = df[["open", "high", "low", "close", "volume"]].copy()
        self._cursor = min(int(warmup_bars), len(self._all))

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._all)

    def poll_latest(self) -> PollResult:
        if self.is_exhausted():
            window = self._all.iloc[max(0, len(self._all) - self.window_bars):]
            return PollResult(new_candle=None, rolling_window=window.copy())

        idx = self._cursor
        ts = self._all.index[idx]
        row = self._all.iloc[idx]
        candle = Candle(
            instrument=self.instrument,
            timeframe=self.timeframe,
            ts=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        self._cursor += 1

        end = self._cursor
        start = max(0, end - self.window_bars)
        window = self._all.iloc[start:end].copy()
        return PollResult(new_candle=candle, rolling_window=window)


# ---------------------------------------------------------------------------
# SyntheticLiveFeed — deterministic generator
# ---------------------------------------------------------------------------
class SyntheticLiveFeed(IncrementalFeed):
    """Generates one believable synthetic 1-min bar per ``poll_latest`` call.

    Used by unit tests so they don't have to fixture a CSV. The series is
    deterministic given ``seed``: ``poll_latest`` produces the same sequence
    of bars in the same order every run.

    Bars are 1-minute increments anchored at ``start``. ``max_bars`` caps
    how many bars the feed will ever emit; after that ``is_exhausted`` is
    True.
    """

    def __init__(
        self,
        *,
        instrument: str = "MES",
        timeframe: str = "1m",
        tz: str = "America/New_York",
        seed: int = 42,
        start: datetime | str = "2024-01-15 09:30",
        base_price: float = 4500.0,
        max_bars: int = 500,
        window_bars: int = 200,
    ) -> None:
        if max_bars < 1:
            raise ValueError("max_bars must be >= 1")
        if window_bars < 1:
            raise ValueError("window_bars must be >= 1")

        self.instrument = instrument
        self.timeframe = timeframe
        self.tz = tz
        self.window_bars = int(window_bars)
        self.max_bars = int(max_bars)

        if isinstance(start, str):
            start_ts = pd.Timestamp(start, tz=tz)
        else:
            if start.tzinfo is None:
                start_ts = pd.Timestamp(start, tz=tz)
            else:
                start_ts = pd.Timestamp(start).tz_convert(tz)

        self._start_ts = start_ts
        self._rng = np.random.default_rng(seed)
        self._n_emitted = 0
        self._last_close = float(base_price)

        self._buffer: deque[pd.Series] = deque(maxlen=int(window_bars))

    def is_exhausted(self) -> bool:
        return self._n_emitted >= self.max_bars

    def _build_window(self) -> pd.DataFrame:
        if not self._buffer:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"],
                index=pd.DatetimeIndex([], tz=self.tz, name="timestamp"),
            )
        df = pd.DataFrame(list(self._buffer))
        df.index.name = "timestamp"
        return df[["open", "high", "low", "close", "volume"]]

    def poll_latest(self) -> PollResult:
        if self.is_exhausted():
            return PollResult(new_candle=None, rolling_window=self._build_window())

        ts = self._start_ts + timedelta(minutes=self._n_emitted)
        # Trends up for the first half, then reverses — mirrors the test fixture
        # so unit tests that compare behavior to ``synthetic_ohlcv`` are stable.
        half = self.max_bars // 2
        drift = 0.05 / 1000.0 if self._n_emitted < half else -0.05 / 1000.0
        ret = float(self._rng.normal(0, 0.0008)) + drift

        open_ = self._last_close
        close = open_ * float(np.exp(ret))
        upper_noise = abs(float(self._rng.normal(0, 0.0006)))
        lower_noise = abs(float(self._rng.normal(0, 0.0006)))
        high = max(open_, close) * (1.0 + upper_noise)
        low = min(open_, close) * (1.0 - lower_noise)
        volume = abs(float(self._rng.normal(1000, 200))) + 1.0

        candle = Candle(
            instrument=self.instrument,
            timeframe=self.timeframe,
            ts=ts.to_pydatetime(),
            open=float(open_),
            high=float(high),
            low=float(low),
            close=float(close),
            volume=float(volume),
        )
        bar_row = pd.Series(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            name=ts,
        )
        self._buffer.append(bar_row)
        self._last_close = close
        self._n_emitted += 1

        return PollResult(new_candle=candle, rolling_window=self._build_window())
