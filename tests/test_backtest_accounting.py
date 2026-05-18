"""Hand-computed PnL after slippage + commission."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import os

import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from config.settings import reload_settings
from features.feature_builder import FEATURE_COLUMNS
from storage.db import init_db
from strategies.base import Setup


NY = ZoneInfo("America/New_York")


def _settings():
    env = {
        "MODE": "BACKTEST",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "TRADING_WINDOW_START": "09:30",
        "TRADING_WINDOW_END": "15:55",
        "FORCE_FLAT_TIME": "15:55",
        "MAX_TRADES_PER_DAY": "10",
        "MAX_DAILY_LOSS": "10000",
        "MAX_DAILY_PROFIT": "10000",
        "MAX_POSITION_SIZE": "1",
        "RISK_PER_TRADE": "100",
        "MAX_OPEN_POSITIONS": "1",
        "COOLDOWN_AFTER_LOSS_MINUTES": "0",
        "COOLDOWN_AFTER_LARGE_WIN_MINUTES": "0",
        "LARGE_WIN_THRESHOLD": "9999",
        "MAX_HOLD_BARS": "20",
        "SLIPPAGE_TICKS": "1",     # 1 tick = $0.25 on MES
        "COMMISSION_PER_CONTRACT": "1.5",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    for k, v in env.items():
        os.environ[k] = v
    return reload_settings()


def _row(hh: str, o, h, lo, c):
    ts = datetime.strptime(f"2024-01-15 {hh}", "%Y-%m-%d %H:%M").replace(tzinfo=NY)
    return ts, {"open": o, "high": h, "low": lo, "close": c, "volume": 1000.0}


def _ohlcv(rows):
    idx = []
    cols = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for r in rows:
        ts, d = r
        idx.append(ts)
        for k, v in d.items():
            cols[k].append(v)
    return pd.DataFrame(cols, index=pd.DatetimeIndex(idx))


def _setup(ts, entry, stop, target, direction="long", idx=0) -> Setup:
    return Setup(
        instrument="MES",
        timestamp=ts,
        strategy_name="test",
        direction=direction,
        entry_price=entry, stop_price=stop, target_price=target,
        atr_at_entry=1.0,
        features={c: 0.0 for c in FEATURE_COLUMNS},
        bar_index=idx,
    )


def test_three_trade_sequence_pnl_matches_handcalc() -> None:
    init_db()
    settings = _settings()

    # Bars (NY local). Each trade signals on its first bar, fills next bar open.
    rows = [
        _row("10:00", 4500, 4500.1, 4499.9, 4500),  # signal #1
        _row("10:01", 4501, 4505,    4500,   4504), # entry @ 4501; tp @ 4504
        _row("10:02", 4504, 4505,    4503,   4504),
        _row("10:03", 4504, 4504.1,  4503.9, 4504), # signal #2
        _row("10:04", 4504, 4505,    4500,   4500.5), # entry @ 4504; sl @ 4502
        _row("10:05", 4500.5, 4501,  4500,   4500.5), # signal #3
        _row("10:06", 4500, 4504,    4499.5, 4503), # entry @ 4500; tp @ 4503
        _row("10:07", 4503, 4503.5,  4502.5, 4503),
    ]
    df = _ohlcv(rows)

    setups = [
        _setup(df.index[0], 4500, 4498, 4504, "long", 0),
        _setup(df.index[3], 4504, 4502, 4508, "long", 3),
        _setup(df.index[5], 4500, 4498, 4503, "long", 5),
    ]
    result = BacktestEngine(settings=settings).run(df, setups)
    trades = result.closed_trades
    assert len(trades) == 3

    # Hand math (1 contract; point_value = $5; 1 tick slippage = $0.25 each side):
    # Trade 1: long entry raw=4501 → 4501.25; tp=4504 → 4503.75. Gross = (4503.75 - 4501.25)*5 = 12.5
    #         Commission = 1.5 + 1.5 = 3.0. Net = 9.5
    # Trade 2: entry raw=4504 → 4504.25; sl=4502 → 4501.75. Gross = (4501.75 - 4504.25)*5 = -12.5
    #         Net = -12.5 - 3 = -15.5
    # Trade 3: entry raw=4500 → 4500.25; tp=4503 → 4502.75. Gross = (4502.75 - 4500.25)*5 = 12.5
    #         Net = 12.5 - 3 = 9.5
    expected = [(12.5, 9.5), (-12.5, -15.5), (12.5, 9.5)]
    for t, (g, n) in zip(trades, expected):
        assert t.gross_pnl == pytest.approx(g, abs=1e-9)
        assert t.net_pnl == pytest.approx(n, abs=1e-9)

    # Aggregate.
    assert result.metrics.gross_pnl == pytest.approx(12.5)
    assert result.metrics.net_pnl == pytest.approx(9.5 - 15.5 + 9.5)
    assert result.metrics.total_commission == pytest.approx(9.0)
    assert result.metrics.n_wins == 2
    assert result.metrics.n_losses == 1
