"""Time-of-day filter for the VWAP/EMA strategy."""

from __future__ import annotations

import pandas as pd

from strategies.vwap_ema_pullback import VWAPEMAPullback, VWAPEMAPullbackParams


def _index(times: list[str]):
    return pd.DatetimeIndex(
        [pd.Timestamp(f"2026-06-17 {t}", tz="America/New_York") for t in times]
    )


def test_no_filter_keeps_all_bars() -> None:
    idx = _index(["09:31", "10:00", "12:00", "15:50"])
    p = VWAPEMAPullbackParams()
    mask = VWAPEMAPullback._time_of_day_mask(idx, p)
    assert mask == [True, True, True, True]


def test_skip_open_minutes_drops_early_bars() -> None:
    idx = _index(["09:35", "09:45", "10:05", "12:00"])
    # Skip first 30 min after 09:30 → window starts 10:00.
    p = VWAPEMAPullbackParams(skip_open_minutes=30)
    mask = VWAPEMAPullback._time_of_day_mask(idx, p)
    assert mask == [False, False, True, True]


def test_skip_close_minutes_drops_late_bars() -> None:
    idx = _index(["12:00", "15:30", "15:50", "15:59"])
    # Skip last 30 min before 16:00 → window ends 15:30.
    p = VWAPEMAPullbackParams(skip_close_minutes=30)
    mask = VWAPEMAPullback._time_of_day_mask(idx, p)
    assert mask == [True, True, False, False]


def test_both_windows() -> None:
    idx = _index(["09:40", "10:30", "15:00", "15:55"])
    p = VWAPEMAPullbackParams(skip_open_minutes=15, skip_close_minutes=15)
    mask = VWAPEMAPullback._time_of_day_mask(idx, p)
    # window = 09:45..15:45
    assert mask == [False, True, True, False]
