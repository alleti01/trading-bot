"""Backtest engine — entry-on-next-open, TP/SL, same-bar ambiguity, max hold."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtesting.engine import BacktestEngine
from config.settings import reload_settings
from features.feature_builder import FEATURE_COLUMNS
from strategies.base import Setup
from storage.db import init_db


NY = ZoneInfo("America/New_York")


def _settings(**overrides):
    """Build settings with friendly defaults; tests can override keys."""
    import os

    defaults = {
        "MODE": "BACKTEST",
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


def _ohlcv(rows: list[tuple[str, float, float, float, float]]) -> pd.DataFrame:
    """rows: (HH:MM, open, high, low, close)."""
    idx = []
    data = {"open": [], "high": [], "low": [], "close": [], "volume": []}
    for hhmm, o, h, lo, c in rows:
        ts = datetime.strptime(f"2024-01-15 {hhmm}", "%Y-%m-%d %H:%M").replace(tzinfo=NY)
        idx.append(ts)
        data["open"].append(o)
        data["high"].append(h)
        data["low"].append(lo)
        data["close"].append(c)
        data["volume"].append(1000.0)
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


def _setup_at(ts, *, direction="long", entry, stop, target, idx=0) -> Setup:
    return Setup(
        instrument="MES",
        timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
        strategy_name="test",
        direction=direction,
        entry_price=entry, stop_price=stop, target_price=target,
        atr_at_entry=1.0,
        features={c: 0.0 for c in FEATURE_COLUMNS},
        bar_index=idx,
    )


# ---------------------------------------------------------------------------
# Entry-on-next-open
# ---------------------------------------------------------------------------
def test_signal_on_close_fills_on_next_bar_open() -> None:
    init_db()
    settings = _settings()
    df = _ohlcv([
        ("10:00", 4500, 4500.5, 4499.5, 4500.0),  # signal bar
        ("10:01", 4501, 4505, 4500, 4504),         # entry bar; tp inside (target=4504)
        ("10:02", 4504, 4506, 4503, 4505),
    ])
    setup = _setup_at(df.index[0], entry=4500, stop=4498, target=4504)

    engine = BacktestEngine(settings=settings)
    result = engine.run(df, [setup])

    assert result.metrics.n_trades == 1
    trade = result.closed_trades[0]
    # Entry filled at next bar's open = 4501 (zero slippage).
    assert trade.entry_price == pytest.approx(4501.0)
    # TP target=4504 → exit_price 4504 (zero slippage).
    assert trade.exit_price == pytest.approx(4504.0)
    assert trade.exit_reason == "tp"


# ---------------------------------------------------------------------------
# Same-bar ambiguity → SL first
# ---------------------------------------------------------------------------
def test_same_bar_tp_and_sl_resolves_sl_first() -> None:
    init_db()
    settings = _settings()
    df = _ohlcv([
        ("10:00", 4500, 4500.5, 4499.5, 4500.0),
        ("10:01", 4501, 4506, 4495, 4501),  # range covers BOTH tp and sl
    ])
    setup = _setup_at(df.index[0], entry=4500, stop=4498, target=4504)

    engine = BacktestEngine(settings=settings)
    result = engine.run(df, [setup])
    assert result.metrics.n_trades == 1
    trade = result.closed_trades[0]
    assert trade.exit_reason == "sl"
    assert trade.exit_price == pytest.approx(4498.0)


# ---------------------------------------------------------------------------
# Max hold time-out
# ---------------------------------------------------------------------------
def test_max_hold_forces_exit_at_close() -> None:
    init_db()
    settings = _settings(MAX_HOLD_BARS="3")
    # Bars after entry that never reach tp or sl. Entry on bar 10:01 open.
    df = _ohlcv([
        ("10:00", 4500, 4500.1, 4499.9, 4500),
        ("10:01", 4500, 4500.5, 4499.5, 4500.2),  # entry here
        ("10:02", 4500.2, 4500.6, 4499.8, 4500.3),
        ("10:03", 4500.3, 4500.7, 4499.9, 4500.4),
        ("10:04", 4500.4, 4500.8, 4500.0, 4500.5),  # bar 3 after entry → time-out
        ("10:05", 4500.5, 4500.9, 4500.1, 4500.6),
    ])
    setup = _setup_at(df.index[0], entry=4500, stop=4498, target=4504)
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, [setup])
    assert result.metrics.n_trades == 1
    trade = result.closed_trades[0]
    assert trade.exit_reason == "time"


# ---------------------------------------------------------------------------
# Forced flat
# ---------------------------------------------------------------------------
def test_forced_flat_closes_open_position() -> None:
    init_db()
    settings = _settings(FORCE_FLAT_TIME="15:55", MAX_HOLD_BARS="100")
    df = _ohlcv([
        ("15:50", 4500, 4500.1, 4499.9, 4500),
        ("15:51", 4500, 4500.5, 4499.5, 4500.2),  # entry here
        ("15:55", 4500.2, 4501, 4499, 4500.5),    # at flat time → forced exit
        ("15:56", 4500.5, 4502, 4500, 4501),
    ])
    setup = _setup_at(df.index[0], entry=4500, stop=4498, target=4504)
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, [setup])
    assert result.metrics.n_trades == 1
    assert result.closed_trades[0].exit_reason == "forced_flat"


def test_no_new_entries_at_or_after_force_flat() -> None:
    init_db()
    # FORCE_FLAT_TIME = 14:00 — setup detected at 14:00 close → would-be entry
    # at 14:01 open is past flat → risk engine blocks.
    settings = _settings(FORCE_FLAT_TIME="14:00")
    df = _ohlcv([
        ("13:59", 4500, 4501, 4499, 4500),
        ("14:00", 4500, 4501, 4499, 4500),  # signal bar
        ("14:01", 4500, 4503, 4499, 4502),  # would have hit tp, but blocked
    ])
    setup = _setup_at(df.index[1], entry=4500, stop=4498, target=4502)
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, [setup])
    assert result.metrics.n_trades == 0
    assert result.n_setups_risk_blocked == 1


# ---------------------------------------------------------------------------
# Max trades per day
# ---------------------------------------------------------------------------
def test_max_trades_per_day_blocks_third_setup() -> None:
    init_db()
    settings = _settings(MAX_TRADES_PER_DAY="2")
    df = _ohlcv([
        ("10:00", 4500, 4501, 4499, 4500),
        ("10:01", 4500, 4504, 4499, 4504),  # trade 1 (entry on 10:01)
        ("10:02", 4504, 4505, 4503, 4504),
        ("10:03", 4504, 4504.5, 4503.5, 4504),
        ("10:04", 4504, 4508, 4503, 4508),  # trade 2
        ("10:05", 4508, 4509, 4507, 4508),
        ("10:06", 4508, 4509, 4507, 4508),
        ("10:07", 4508, 4512, 4507, 4512),  # trade 3 — should be blocked
    ])
    setups = [
        _setup_at(df.index[0], entry=4500, stop=4498, target=4504, idx=0),
        _setup_at(df.index[3], entry=4504, stop=4502, target=4508, idx=3),
        _setup_at(df.index[6], entry=4508, stop=4506, target=4512, idx=6),
    ]
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, setups)
    assert result.metrics.n_trades == 2
    assert result.n_setups_risk_blocked == 1
    assert any(rb.rule == "max_trades_per_day" for rb in result.risk_blocks)


# ---------------------------------------------------------------------------
# Daily loss / profit caps (set very tight to trip on first trade)
# ---------------------------------------------------------------------------
def test_max_daily_loss_blocks_after_one_loss() -> None:
    init_db()
    # MES: 1pt = $5; risk_per_trade $100 → qty = floor(100/5) = 20, capped at 1
    # max_position_size=1 → losing 2pt = -$10 net. Set MAX_DAILY_LOSS = $5 → next
    # entry must be blocked.
    settings = _settings(MAX_DAILY_LOSS="5", MAX_POSITION_SIZE="1")
    df = _ohlcv([
        ("10:00", 4500, 4501, 4499, 4500),
        ("10:01", 4500, 4500.5, 4498, 4498.5),  # entry; sl=4498 hit
        ("10:02", 4498.5, 4499, 4498, 4498.5),
        ("10:03", 4498.5, 4499, 4498, 4498.5),
        ("10:04", 4498.5, 4499, 4498, 4498.5),
        ("10:05", 4498.5, 4502, 4498, 4501),    # second setup — should be blocked
    ])
    setups = [
        _setup_at(df.index[0], entry=4500, stop=4498, target=4504, idx=0),
        _setup_at(df.index[4], entry=4498.5, stop=4496.5, target=4502.5, idx=4),
    ]
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, setups)
    assert result.metrics.n_trades == 1
    assert any(rb.rule == "max_daily_loss" for rb in result.risk_blocks)


def test_max_daily_profit_blocks_after_one_win() -> None:
    init_db()
    # MAX_DAILY_PROFIT=$5 — first win trips it.
    settings = _settings(MAX_DAILY_PROFIT="5", MAX_POSITION_SIZE="1")
    df = _ohlcv([
        ("10:00", 4500, 4501, 4499, 4500),
        ("10:01", 4500, 4504, 4499, 4504),
        ("10:02", 4504, 4505, 4503, 4504),
        ("10:03", 4504, 4505, 4503, 4504),
        ("10:04", 4504, 4505, 4503, 4504),
        ("10:05", 4504, 4508, 4503, 4508),  # second setup — blocked
    ])
    setups = [
        _setup_at(df.index[0], entry=4500, stop=4498, target=4504, idx=0),
        _setup_at(df.index[4], entry=4504, stop=4502, target=4508, idx=4),
    ]
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, setups)
    assert result.metrics.n_trades == 1
    assert any(rb.rule == "max_daily_profit" for rb in result.risk_blocks)


# ---------------------------------------------------------------------------
# No hedging
# ---------------------------------------------------------------------------
def test_no_hedging_blocks_opposite_direction_in_engine() -> None:
    init_db()
    settings = _settings()
    df = _ohlcv([
        ("10:00", 4500, 4501, 4499, 4500),
        ("10:01", 4500, 4501, 4500, 4500.5),  # long entry here
        ("10:02", 4500.5, 4501, 4500, 4500.5),
        ("10:03", 4500.5, 4501, 4500, 4500.5),  # short setup here while long open
        ("10:04", 4500.5, 4501, 4500, 4500.5),
        ("10:05", 4500.5, 4504, 4500, 4504),    # tp for long
    ])
    long_setup = _setup_at(df.index[0], direction="long", entry=4500, stop=4498, target=4504, idx=0)
    short_setup = _setup_at(df.index[2], direction="short", entry=4500.5, stop=4502.5, target=4498.5, idx=2)
    engine = BacktestEngine(settings=settings)
    result = engine.run(df, [long_setup, short_setup])
    assert result.metrics.n_trades == 1
    # The single-position book blocks any second entry while one is open
    # (which inherently prevents hedging), via the risk engine or the
    # engine's own position-open guard.
    assert any(
        rb.rule in {"max_open_positions", "no_hedging", "position_already_open"}
        for rb in result.risk_blocks
    )
