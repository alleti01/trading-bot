"""Shared exit logic — TP, SL, same-bar tie-break, max hold, forced flat."""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtesting.portfolio import OpenPosition
from backtesting.trade_management import check_exit


NY = ZoneInfo("America/New_York")


def _bar(o: float, h: float, lo: float, c: float, *, ts: datetime) -> pd.Series:
    return pd.Series(
        {"open": o, "high": h, "low": lo, "close": c, "volume": 1000.0}, name=ts
    )


def _open(direction: str = "long", *, entry: float = 100.0, stop: float = 98.0,
          target: float = 104.0, entry_idx: int = 0) -> OpenPosition:
    return OpenPosition(
        setup_id="s1",
        instrument="MES",
        direction=direction,
        quantity=1.0,
        entry_ts=datetime(2024, 1, 15, 9, 31, tzinfo=NY),
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        entry_commission=0.0,
        entry_slippage=0.0,
        entry_bar_index=entry_idx,
    )


def test_tp_only_returns_target() -> None:
    pos = _open()
    bar_ts = datetime(2024, 1, 15, 9, 32, tzinfo=NY)
    bar = _bar(100, 105, 100, 104, ts=bar_ts)  # high reaches 105 > target 104
    decision = check_exit(
        position=pos, bar=bar, bar_index=1, bar_ts=bar_ts,
        max_hold_bars=20, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    )
    assert decision is not None
    assert decision.reason == "tp"
    assert decision.raw_price == 104.0


def test_sl_only_returns_stop() -> None:
    pos = _open()
    bar_ts = datetime(2024, 1, 15, 9, 32, tzinfo=NY)
    bar = _bar(100, 100.5, 97, 99, ts=bar_ts)  # low 97 <= stop 98
    decision = check_exit(
        position=pos, bar=bar, bar_index=1, bar_ts=bar_ts,
        max_hold_bars=20, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    )
    assert decision is not None
    assert decision.reason == "sl"
    assert decision.raw_price == 98.0


def test_same_bar_tp_and_sl_resolves_to_sl() -> None:
    pos = _open()
    bar_ts = datetime(2024, 1, 15, 9, 32, tzinfo=NY)
    bar = _bar(100, 105, 97, 100, ts=bar_ts)  # both hit
    decision = check_exit(
        position=pos, bar=bar, bar_index=1, bar_ts=bar_ts,
        max_hold_bars=20, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    )
    assert decision is not None
    assert decision.reason == "sl"


def test_max_hold_returns_close() -> None:
    pos = _open(entry_idx=0)
    bar_ts = datetime(2024, 1, 15, 9, 35, tzinfo=NY)
    bar = _bar(100, 100.1, 99.9, 100.05, ts=bar_ts)
    decision = check_exit(
        position=pos, bar=bar, bar_index=5, bar_ts=bar_ts,
        max_hold_bars=5, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    )
    assert decision is not None
    assert decision.reason == "time"
    assert decision.raw_price == 100.05


def test_no_exit_when_inside_range_and_under_max_hold() -> None:
    pos = _open()
    bar_ts = datetime(2024, 1, 15, 9, 32, tzinfo=NY)
    bar = _bar(100, 100.5, 99.5, 100.1, ts=bar_ts)
    assert check_exit(
        position=pos, bar=bar, bar_index=1, bar_ts=bar_ts,
        max_hold_bars=20, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    ) is None


def test_forced_flat_for_futures_at_flat_time() -> None:
    pos = _open()
    bar_ts = datetime(2024, 1, 15, 15, 55, tzinfo=NY)
    bar = _bar(100, 100.3, 99.8, 100.1, ts=bar_ts)
    decision = check_exit(
        position=pos, bar=bar, bar_index=300, bar_ts=bar_ts,
        max_hold_bars=10000, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    )
    assert decision is not None
    assert decision.reason == "forced_flat"
    assert decision.raw_price == 100.0  # bar's open


def test_forced_flat_does_not_apply_to_crypto() -> None:
    pos = _open()
    bar_ts = datetime(2024, 1, 15, 15, 55, tzinfo=NY)
    bar = _bar(100, 100.3, 99.8, 100.1, ts=bar_ts)
    # Past flat time, crypto market — should not force-flat.
    decision = check_exit(
        position=pos, bar=bar, bar_index=2, bar_ts=bar_ts,
        max_hold_bars=20, force_flat_time=time(15, 55),
        market_type="crypto", tz=NY,
    )
    assert decision is None


def test_short_position_tp_sl_logic_mirrors_long() -> None:
    pos = _open(direction="short", entry=100.0, stop=102.0, target=96.0)
    bar_ts = datetime(2024, 1, 15, 9, 32, tzinfo=NY)
    # Low reaches 95 → target hit (short tp = low <= target).
    bar = _bar(100, 100.2, 95, 96, ts=bar_ts)
    decision = check_exit(
        position=pos, bar=bar, bar_index=1, bar_ts=bar_ts,
        max_hold_bars=20, force_flat_time=time(15, 55),
        market_type="futures", tz=NY,
    )
    assert decision is not None
    assert decision.reason == "tp"
    assert decision.raw_price == 96.0


def test_unknown_direction_raises() -> None:
    pos = OpenPosition(
        setup_id="s1", instrument="MES", direction="LONG",  # wrong case
        quantity=1.0, entry_ts=datetime(2024, 1, 15, 9, 31, tzinfo=NY),
        entry_price=100.0, stop_price=98.0, target_price=104.0,
        entry_commission=0.0, entry_slippage=0.0, entry_bar_index=0,
    )
    bar_ts = datetime(2024, 1, 15, 9, 32, tzinfo=NY)
    with pytest.raises(ValueError):
        check_exit(
            position=pos, bar=_bar(100, 101, 99, 100, ts=bar_ts),
            bar_index=1, bar_ts=bar_ts, max_hold_bars=20,
            force_flat_time=time(15, 55), market_type="futures", tz=NY,
        )
