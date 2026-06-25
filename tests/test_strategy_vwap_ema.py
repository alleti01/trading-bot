"""VWAP/EMA pullback strategy: hand-crafted feature rows producing setups."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from features.feature_builder import FEATURE_COLUMNS
from strategies.vwap_ema_pullback import VWAPEMAPullback, VWAPEMAPullbackParams


def _row(
    ts: datetime,
    *,
    close: float,
    vwap: float,
    ema_9: float,
    ema_21: float,
    ema_50: float,
    atr: float = 1.0,
    vol_ratio: float = 1.0,
) -> dict:
    """Build a complete feature row that satisfies the canonical column set."""
    base = {col: 0.0 for col in FEATURE_COLUMNS}
    base.update(
        {
            "close": close,  # added below as raw OHLCV
            "vwap": vwap,
            "ema_9": ema_9,
            "ema_21": ema_21,
            "ema_50": ema_50,
            "atr_14": atr,
            "volume_ratio_20": vol_ratio,
        }
    )
    base["_ts"] = ts
    return base


def _make_features_df(rows: list[dict]) -> pd.DataFrame:
    timestamps = [r.pop("_ts") for r in rows]
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(timestamps, tz="UTC"))
    # Strategy needs raw close as well (carried alongside features).
    df["open"] = df["close"]
    df["high"] = df["close"]
    df["low"] = df["close"]
    df["volume"] = 1000.0
    return df


def test_strategy_emits_long_setup() -> None:
    """A trend-up bar that pulls back to EMA21 should fire long."""
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    # close > vwap; ema9 > ema21 > ema50; close == ema21 (pullback exact).
    rows = [
        _row(ts, close=101.0, vwap=99.0, ema_9=102.0, ema_21=101.0, ema_50=100.0),
    ]
    df = _make_features_df(rows)
    setups = VWAPEMAPullback(instrument="MES").detect_setups(df)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == "long"
    assert s.entry_price == 101.0
    assert s.atr_at_entry == 1.0
    assert s.stop_price == 100.25  # entry - 0.75 * 1.0
    assert s.target_price == 102.5  # entry + 1.5 * 1.0


def test_strategy_emits_short_setup() -> None:
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    # close < vwap; ema9 < ema21 < ema50; close == ema21.
    rows = [
        _row(ts, close=99.0, vwap=101.0, ema_9=98.0, ema_21=99.0, ema_50=100.0),
    ]
    df = _make_features_df(rows)
    setups = VWAPEMAPullback(instrument="MES").detect_setups(df)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction == "short"
    assert s.entry_price == 99.0
    assert s.stop_price == 99.75  # entry + 0.75 * 1.0
    assert s.target_price == 97.5  # entry - 1.5 * 1.0


def test_strategy_skips_when_volume_too_low() -> None:
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    rows = [
        _row(
            ts,
            close=101.0, vwap=99.0,
            ema_9=102.0, ema_21=101.0, ema_50=100.0,
            vol_ratio=0.1,
        ),
    ]
    df = _make_features_df(rows)
    setups = VWAPEMAPullback(instrument="MES").detect_setups(df)
    assert setups == []


def test_strategy_skips_when_atr_out_of_band() -> None:
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    rows = [
        _row(
            ts,
            close=101.0, vwap=99.0,
            ema_9=102.0, ema_21=101.0, ema_50=100.0,
            atr=100.0,  # huge — outside default atr_max=10
        ),
    ]
    df = _make_features_df(rows)
    setups = VWAPEMAPullback(instrument="MES").detect_setups(df)
    assert setups == []


def test_strategy_skips_when_no_pullback() -> None:
    """Trend is fine but price is far from both anchors."""
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    rows = [
        _row(
            ts,
            close=110.0, vwap=99.0,
            ema_9=108.0, ema_21=104.0, ema_50=100.0,
            atr=1.0,  # pullback band = 0.5; close is 6 from ema21 and 11 from vwap.
        ),
    ]
    df = _make_features_df(rows)
    setups = VWAPEMAPullback(instrument="MES").detect_setups(df)
    assert setups == []


def test_custom_params_override_defaults() -> None:
    # Time-of-day filter: a setup at 10:00 is kept with no filter, but
    # dropped when skip_open_minutes pushes the window past 10:00.
    _tod = VWAPEMAPullbackParams()
    assert _tod.skip_open_minutes == 0.0  # default: no filter

    params = VWAPEMAPullbackParams(stop_atr_mult=2.0, target_atr_mult=4.0)
    ts = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    rows = [
        _row(ts, close=101.0, vwap=99.0, ema_9=102.0, ema_21=101.0, ema_50=100.0),
    ]
    df = _make_features_df(rows)
    setups = VWAPEMAPullback(params=params, instrument="MES").detect_setups(df)
    assert setups[0].stop_price == 99.0
    assert setups[0].target_price == 105.0
