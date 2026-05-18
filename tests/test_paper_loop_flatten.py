"""``PaperTradingLoop.flatten_now`` correctness.

Earlier code closed at the position's *entry price*, which produced a
near-zero gross PnL on every forced flat regardless of where the market
actually moved. We track the last seen bar's close and use it instead.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd

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
from storage.db import init_db
from strategies.base import Setup, Strategy, StrategyParams


NY = ZoneInfo("America/New_York")


def _settings() -> object:
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
        "MAX_HOLD_BARS": "100",
        "SLIPPAGE_TICKS": "0",
        "COMMISSION_PER_CONTRACT": "0",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


class _NullStrategy(Strategy):
    name = "null"

    @classmethod
    def _default_params(cls):
        return StrategyParams()

    def detect_setups(self, features_df: pd.DataFrame):
        return []


class _ScriptedFeed(IncrementalFeed):
    def __init__(self, sequence):
        self.instrument = "MES"
        self.timeframe = "1m"
        self._seq = list(sequence)
        self._idx = 0

    def is_exhausted(self) -> bool:
        return self._idx >= len(self._seq)

    def poll_latest(self) -> PollResult:
        if self.is_exhausted():
            return PollResult(new_candle=None, rolling_window=pd.DataFrame())
        candle, window = self._seq[self._idx]
        self._idx += 1
        return PollResult(new_candle=candle, rolling_window=window)


def _make_loop(settings):
    spec = get_instrument("MES")
    portfolio = Portfolio(instrument_spec=spec)
    fills = make_fills_model("MES", slippage_ticks=0.0, commission_per_contract=0.0)
    executor = PaperExecutor(portfolio=portfolio, fills_model=fills)
    notifier = NotificationService(discord=None)
    return PaperTradingLoop(
        settings=settings,
        feed=_ScriptedFeed([]),
        strategy=_NullStrategy(),
        notifier=notifier,
        portfolio=portfolio,
        executor=executor,
    )


def test_flatten_now_uses_last_observed_bar_close() -> None:
    """A long opened at 4500 with the most recent bar closing at 4510
    must close at 4510 (gross PnL > 0), not at the entry price.
    """
    init_db()
    settings = _settings()
    loop = _make_loop(settings)

    entry_ts = datetime(2024, 1, 15, 10, 0, tzinfo=NY)
    loop.executor.submit(
        order=__import__("execution.base", fromlist=["Order"]).Order(
            instrument="MES",
            direction="long",
            quantity=1,
            entry_price=4500.0,
            stop_price=4495.0,
            target_price=4510.0,
            setup_id="t1",
        )
    )

    # Simulate the loop seeing one bar that closes at 4510.
    loop._last_bar_ts = entry_ts.replace(minute=5)
    loop._last_bar_close = 4510.0

    closed = loop.flatten_now(now=datetime.now(tz=timezone.utc), reason="forced_flat")
    assert closed is True

    last = loop.portfolio.closed_trades[-1]
    assert last.exit_price == 4510.0  # zero slippage
    # PnL = (4510 - 4500) * point_value(=5) * qty(=1) = $50
    assert last.gross_pnl == 50.0


def test_flatten_now_falls_back_to_entry_price_when_no_bar_seen() -> None:
    """Degenerate case: scheduler force-flat fires before any bar arrived."""
    init_db()
    settings = _settings()
    loop = _make_loop(settings)

    loop.executor.submit(
        order=__import__("execution.base", fromlist=["Order"]).Order(
            instrument="MES",
            direction="long",
            quantity=1,
            entry_price=4500.0,
            stop_price=4495.0,
            target_price=4510.0,
            setup_id="t1",
        )
    )
    # _last_bar_close left as None on purpose.

    closed = loop.flatten_now(now=datetime.now(tz=timezone.utc), reason="forced_flat")
    assert closed is True
    last = loop.portfolio.closed_trades[-1]
    assert last.exit_price == 4500.0
    assert last.gross_pnl == 0.0


def test_flatten_now_noop_when_no_open_position() -> None:
    init_db()
    settings = _settings()
    loop = _make_loop(settings)
    closed = loop.flatten_now(now=datetime.now(tz=timezone.utc))
    assert closed is False
