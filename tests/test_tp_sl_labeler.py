"""Triple-barrier labeler — hand-crafted future-bar sequences."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from features.feature_builder import FEATURE_COLUMNS
from labeling.tp_sl_labeler import label_setup, label_setups
from strategies.base import Setup


def _setup(direction: str, entry: float, stop: float, target: float, *, atr: float = 1.0) -> Setup:
    return Setup(
        instrument="MES",
        timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        strategy_name="test",
        direction=direction,  # type: ignore[arg-type]
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        atr_at_entry=atr,
        features={col: 0.0 for col in FEATURE_COLUMNS},
        bar_index=0,
    )


def _bars(*ohlcs) -> pd.DataFrame:
    """Quick helper — each arg is (open, high, low, close); volume is filled in."""
    rows = []
    for o, h, lo, c in ohlcs:
        rows.append({"open": o, "high": h, "low": lo, "close": c, "volume": 100.0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Long
# ---------------------------------------------------------------------------
def test_long_clean_tp_first() -> None:
    setup = _setup("long", entry=100, stop=99, target=102)
    # bar 1: drifts up but doesn't hit tp; bar 2: hits tp.
    future = _bars(
        (100.0, 101.0, 99.5, 100.8),
        (100.8, 102.5, 100.5, 102.0),
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 1
    assert r.exit_reason == "tp"
    assert r.exit_bar == 2
    assert r.exit_price == 102.0
    assert r.bars_held == 2


def test_long_clean_sl_first() -> None:
    setup = _setup("long", entry=100, stop=99, target=102)
    future = _bars(
        (100.0, 100.5, 99.7, 100.0),
        (100.0, 100.6, 98.5, 99.0),  # low <= 99 → sl hit
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 0
    assert r.exit_reason == "sl"
    assert r.exit_bar == 2
    assert r.exit_price == 99.0


def test_long_same_bar_tp_and_sl_resolves_sl_first() -> None:
    """Bar's range straddles BOTH levels — conservative rule says SL first."""
    setup = _setup("long", entry=100, stop=99, target=102)
    future = _bars(
        (100.0, 102.5, 98.5, 100.0),  # high >= tp AND low <= sl in same bar
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 0
    assert r.exit_reason == "sl"
    assert r.exit_bar == 1


def test_long_immediate_fill_first_bar() -> None:
    setup = _setup("long", entry=100, stop=99, target=102)
    future = _bars(
        (100.0, 102.5, 100.0, 102.0),  # tp hit instantly, no sl risk
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 1
    assert r.exit_reason == "tp"
    assert r.exit_bar == 1


# ---------------------------------------------------------------------------
# Short (mirrored)
# ---------------------------------------------------------------------------
def test_short_clean_tp_first() -> None:
    setup = _setup("short", entry=100, stop=101, target=98)
    future = _bars(
        (100.0, 100.5, 99.5, 99.7),
        (99.7, 100.0, 97.5, 98.0),  # low <= 98 → tp hit
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 1
    assert r.exit_reason == "tp"
    assert r.exit_bar == 2


def test_short_clean_sl_first() -> None:
    setup = _setup("short", entry=100, stop=101, target=98)
    future = _bars(
        (100.0, 100.5, 99.7, 100.0),
        (100.0, 101.5, 99.8, 101.0),  # high >= 101 → sl hit
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 0
    assert r.exit_reason == "sl"
    assert r.exit_bar == 2


def test_short_same_bar_ambiguity_resolves_sl_first() -> None:
    setup = _setup("short", entry=100, stop=101, target=98)
    future = _bars(
        (100.0, 101.5, 97.5, 100.0),
    )
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 0
    assert r.exit_reason == "sl"


# ---------------------------------------------------------------------------
# Time-out / edge cases
# ---------------------------------------------------------------------------
def test_time_out_returns_zero_with_close_exit() -> None:
    setup = _setup("long", entry=100, stop=99, target=102)
    # Drifts but never reaches tp or sl.
    future = _bars(
        (100.0, 100.4, 99.6, 100.1),
        (100.1, 100.5, 99.7, 100.2),
        (100.2, 100.6, 99.8, 100.3),
    )
    r = label_setup(setup, future, max_hold_bars=3)
    assert r.label == 0
    assert r.exit_reason == "time"
    assert r.bars_held == 3
    assert r.exit_price == 100.3


def test_empty_future_bars_returns_time_out() -> None:
    setup = _setup("long", entry=100, stop=99, target=102)
    future = _bars()
    r = label_setup(setup, future, max_hold_bars=10)
    assert r.label == 0
    assert r.exit_reason == "time"
    assert r.bars_held == 0
    assert r.exit_price == 100.0


def test_max_hold_bars_truncates_future_bars() -> None:
    setup = _setup("long", entry=100, stop=99, target=102)
    # 5 bars of drift, then a tp on bar 6 — but max_hold=3 truncates so we time out.
    future = _bars(
        (100.0, 100.4, 99.6, 100.1),
        (100.1, 100.5, 99.7, 100.2),
        (100.2, 100.6, 99.8, 100.3),
        (100.3, 100.7, 99.9, 100.4),
        (100.4, 100.8, 100.0, 100.5),
        (100.5, 102.5, 100.4, 102.0),  # tp here, but we should never see it
    )
    r = label_setup(setup, future, max_hold_bars=3)
    assert r.exit_reason == "time"
    assert r.bars_held == 3


# ---------------------------------------------------------------------------
# Batch helper
# ---------------------------------------------------------------------------
def test_label_setups_batch() -> None:
    """End-to-end: locate setups in ohlcv by timestamp, label each."""
    idx = pd.date_range("2024-01-15 09:30", periods=10, freq="1min", tz="UTC")
    ohlcv = pd.DataFrame(
        {
            "open":   [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
            "high":   [100, 101, 102.5, 100, 100, 100, 100, 100, 100, 100],
            "low":    [100,  99,  98.5,  99,  99,  99,  99,  99,  99,  99],
            "close":  [100, 100, 102.0, 100, 100, 100, 100, 100, 100, 100],
            "volume": [100] * 10,
        },
        index=idx,
    )

    s = Setup(
        instrument="MES",
        timestamp=idx[0].to_pydatetime(),  # bar 0 → future starts at bar 1
        strategy_name="test",
        direction="long",
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr_at_entry=1.0,
        features={col: 0.0 for col in FEATURE_COLUMNS},
        bar_index=0,
    )
    results = label_setups([s], ohlcv, max_hold_bars=5)
    assert len(results) == 1
    # Bar 1 (idx[1]): high=101 < tp=102, low=99 <= sl=99 → sl hits at bar 1.
    assert results[0].label == 0
    assert results[0].exit_reason == "sl"
    assert results[0].exit_bar == 1
