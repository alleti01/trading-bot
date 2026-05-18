"""Market-data feeds.

Day 2 ships the replay path only: a ``MarketDataFeed`` ABC and a
``CSVReplayFeed`` that yields ``Candle`` objects from a sorted CSV. Live
feeds (broker WebSocket, exchange WebSocket) plug in on later days behind
the same interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path

from data.candle import Candle
from data.csv_loader import load_ohlcv_csv


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
