"""End-to-end paper loop: setup → risk → executor with stub feed and strategy."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from sqlalchemy import select

from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from config.instruments import get_instrument
from config.settings import reload_settings
from data.candle import Candle
from data.market_data_service import IncrementalFeed, PollResult
from execution.paper_executor import PaperExecutor
from features.feature_builder import FEATURE_COLUMNS
from notifications.notification_service import NotificationService
from paper.loop import PaperTradingLoop
from storage.db import init_db, session_scope
from storage.tables import ClosedTrade, PaperTrade, RiskBlock, Setup as SetupRow
from strategies.base import Setup, Strategy


NY = ZoneInfo("America/New_York")


def _settings(**overrides):
    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "TRADING_WINDOW_START": "09:30",
        "TRADING_WINDOW_END": "15:55",
        "FORCE_FLAT_TIME": "15:55",
        "MAX_TRADES_PER_DAY": "8",
        "MAX_DAILY_LOSS": "10000",
        "MAX_DAILY_PROFIT": "10000",
        "MAX_POSITION_SIZE": "1",
        "RISK_PER_TRADE": "100",
        "MAX_OPEN_POSITIONS": "1",
        "COOLDOWN_AFTER_LOSS_MINUTES": "0",
        "COOLDOWN_AFTER_LARGE_WIN_MINUTES": "0",
        "LARGE_WIN_THRESHOLD": "9999",
        "MAX_HOLD_BARS": "5",
        "SLIPPAGE_TICKS": "0",
        "COMMISSION_PER_CONTRACT": "0",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _ScriptedFeed(IncrementalFeed):
    """Yields a pre-baked sequence of (Candle, rolling_window) tuples."""

    def __init__(self, sequence: list[tuple[Candle, pd.DataFrame]]) -> None:
        if sequence:
            self.instrument = sequence[0][0].instrument
            self.timeframe = sequence[0][0].timeframe
        else:
            self.instrument = "MES"
            self.timeframe = "1m"
        self._sequence = list(sequence)
        self._idx = 0

    def is_exhausted(self) -> bool:
        return self._idx >= len(self._sequence)

    def poll_latest(self) -> PollResult:
        if self.is_exhausted():
            return PollResult(new_candle=None, rolling_window=pd.DataFrame())
        candle, window = self._sequence[self._idx]
        self._idx += 1
        return PollResult(new_candle=candle, rolling_window=window)


class _ScriptedStrategy(Strategy):
    """Returns a fixed list of setups whose timestamps should match feed bars."""

    name = "scripted"

    def __init__(self, setups: list[Setup]) -> None:
        super().__init__()
        self._setups = setups

    @classmethod
    def _default_params(cls):
        from strategies.base import StrategyParams
        return StrategyParams()

    def detect_setups(self, features_df: pd.DataFrame) -> list[Setup]:
        # Only return setups whose timestamp is in the rolling features.
        in_window = set(features_df.index)
        return [s for s in self._setups if s.timestamp in in_window]


def _candle(ts: datetime, *, o: float = 4500, h: float = 4501, lo: float = 4499, c: float = 4500) -> Candle:
    return Candle(
        instrument="MES", timeframe="1m", ts=ts,
        open=o, high=h, low=lo, close=c, volume=1000.0,
    )


def _setup(ts: datetime, *, direction: str = "long", entry: float = 4500.0) -> Setup:
    return Setup(
        instrument="MES",
        timestamp=ts,
        strategy_name="scripted",
        direction=direction,  # type: ignore[arg-type]
        entry_price=entry,
        stop_price=entry - 2.0 if direction == "long" else entry + 2.0,
        target_price=entry + 4.0 if direction == "long" else entry - 4.0,
        atr_at_entry=1.0,
        features={c: 0.0 for c in FEATURE_COLUMNS},
        bar_index=0,
    )


def _window_with(ts_list: list[datetime]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [4500] * len(ts_list),
            "high": [4501] * len(ts_list),
            "low": [4499] * len(ts_list),
            "close": [4500] * len(ts_list),
            "volume": [1000.0] * len(ts_list),
        },
        index=pd.DatetimeIndex(ts_list),
    )
    df.index.name = "timestamp"
    return df


def _make_loop(*, strategy: Strategy, feed: IncrementalFeed, settings):
    spec = get_instrument("MES")
    portfolio = Portfolio(instrument_spec=spec)
    fills = make_fills_model("MES", slippage_ticks=0.0, commission_per_contract=0.0)
    executor = PaperExecutor(portfolio=portfolio, fills_model=fills)
    notifier = NotificationService(discord=None)
    return PaperTradingLoop(
        settings=settings,
        feed=feed,
        strategy=strategy,
        notifier=notifier,
        portfolio=portfolio,
        executor=executor,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_loop_opens_position_on_setup_inside_window() -> None:
    init_db()
    settings = _settings()
    ts = datetime(2024, 1, 15, 10, 0, tzinfo=NY)
    setup = _setup(ts)
    feed = _ScriptedFeed([(_candle(ts), _window_with([ts]))])
    strategy = _ScriptedStrategy([setup])

    loop = _make_loop(strategy=strategy, feed=feed, settings=settings)

    # Stub the feature builder so the loop can use our trivial window
    # (no need to run the full indicator pipeline for this test).
    loop._consider_new_entries = _bypass_features(loop, [setup])  # type: ignore[method-assign]

    res = loop.on_bar_close(ts)
    assert res.new_bar is True
    assert res.setups_seen == 1
    assert res.setups_filled == 1
    assert loop.portfolio.open_position is not None

    with session_scope() as session:
        rows = session.execute(select(PaperTrade)).scalars().all()
    assert len(rows) == 1


def test_loop_blocks_setup_via_risk_engine() -> None:
    init_db()
    # Tiny daily loss cap to *not* trigger; the trick: trip the kill switch
    # to force a definite block.
    settings = _settings()
    from risk.kill_switch import KillSwitch

    KillSwitch().trip("test_block")

    ts = datetime(2024, 1, 15, 10, 0, tzinfo=NY)
    setup = _setup(ts)
    feed = _ScriptedFeed([(_candle(ts), _window_with([ts]))])
    strategy = _ScriptedStrategy([setup])
    loop = _make_loop(strategy=strategy, feed=feed, settings=settings)
    loop._consider_new_entries = _bypass_features(loop, [setup])  # type: ignore[method-assign]

    res = loop.on_bar_close(ts)
    # With kill switch tripped, _consider_new_entries returns early before
    # ever reaching risk evaluation. So setups_seen=0, no fills.
    assert res.setups_filled == 0
    assert loop.portfolio.open_position is None


def test_loop_no_new_bar_returns_early() -> None:
    init_db()
    settings = _settings()
    feed = _ScriptedFeed([])
    strategy = _ScriptedStrategy([])
    loop = _make_loop(strategy=strategy, feed=feed, settings=settings)
    res = loop.on_bar_close(datetime(2024, 1, 15, 10, 0, tzinfo=NY))
    assert res.new_bar is False


def test_loop_filters_setups_to_latest_bar_timestamp() -> None:
    """Old setups (before latest bar) must be ignored."""
    init_db()
    settings = _settings()
    ts_old = datetime(2024, 1, 15, 9, 59, tzinfo=NY)
    ts_new = datetime(2024, 1, 15, 10, 0, tzinfo=NY)

    feed = _ScriptedFeed([(_candle(ts_new), _window_with([ts_old, ts_new]))])
    strategy = _ScriptedStrategy([_setup(ts_old), _setup(ts_new)])
    loop = _make_loop(strategy=strategy, feed=feed, settings=settings)
    # We DO NOT bypass features here — we exercise the real filter logic in the loop.

    # Patch build_features to return a predictable two-row dataframe with
    # both old + new timestamps so the loop's "latest_ts" filter is exercised.
    import paper.loop as loop_mod

    def fake_build_features(df, *, instrument, tz):  # noqa: ARG001
        return df

    loop_mod.build_features = fake_build_features  # type: ignore[attr-defined]
    try:
        res = loop.on_bar_close(ts_new)
    finally:
        # Restore the real feature builder.
        from features.feature_builder import build_features as real_build
        loop_mod.build_features = real_build  # type: ignore[attr-defined]

    # Strategy was scripted to return setups whose ts are in the features
    # window (both old + new), but the loop should only have considered ts_new.
    assert res.setups_seen == 1


# ---------------------------------------------------------------------------
# Helper: bypass the heavy feature builder for the entry-check tests above.
# ---------------------------------------------------------------------------
def _bypass_features(loop: PaperTradingLoop, setups: list[Setup]):
    def _consider(*, rolling_window, latest_ts, bar_series, now, result):
        latest = [s for s in setups if s.timestamp == latest_ts]
        result.setups_seen = len(latest)
        for s in latest:
            loop._handle_setup(s, bar_series, now, result)
    return _consider
