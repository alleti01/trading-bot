"""Numerical correctness tests for individual indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.indicators import (
    atr,
    candle_body,
    direction,
    ema,
    lower_wick,
    recent_high,
    recent_low,
    rolling_return,
    session_vwap,
    trend_regime,
    upper_wick,
    volatility_regime,
    volume_ratio,
)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
def test_ema_matches_handcalc() -> None:
    """EMA(span=2, adjust=False): alpha = 2/3."""
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = ema(x, 2)
    alpha = 2.0 / 3.0
    expected = [1.0]
    for v in x.iloc[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])
    np.testing.assert_allclose(out.values, expected, rtol=1e-12)


def test_ema_rejects_zero_period() -> None:
    with pytest.raises(ValueError):
        ema(pd.Series([1.0, 2.0]), 0)


# ---------------------------------------------------------------------------
# ATR (Wilder)
# ---------------------------------------------------------------------------
def test_atr_matches_wilder_handcalc() -> None:
    df = pd.DataFrame(
        {
            "open":  [10.0, 11.0, 10.5, 12.0],
            "high":  [11.0, 12.0, 11.0, 12.5],
            "low":   [9.5,  10.5, 10.0, 11.5],
            "close": [10.5, 11.5, 10.8, 12.2],
        }
    )
    out = atr(df, period=2)

    # Hand-compute TR. First bar uses high-low.
    tr = [11.0 - 9.5]
    for i in range(1, len(df)):
        prev_c = df["close"].iloc[i - 1]
        h = df["high"].iloc[i]
        lo = df["low"].iloc[i]
        tr.append(max(h - lo, abs(h - prev_c), abs(lo - prev_c)))

    # Wilder smoothing alpha = 1/2, adjust=False, seeded at TR[0].
    alpha = 0.5
    expected = [tr[0]]
    for v in tr[1:]:
        expected.append(alpha * v + (1 - alpha) * expected[-1])

    np.testing.assert_allclose(out.values, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------
def test_session_vwap_resets_on_new_session() -> None:
    df = pd.DataFrame(
        {
            "high":   [10.0, 11.0, 12.0, 20.0, 21.0],
            "low":    [9.0,  10.0, 11.0, 19.0, 20.0],
            "close":  [9.5,  10.5, 11.5, 19.5, 20.5],
            "volume": [100,  100,  100,  100,  100],
        }
    )
    sess = pd.Series(["A", "A", "A", "B", "B"])
    out = session_vwap(df, session_label=sess)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]

    # Session A spans rows 0..2; session B spans rows 3..4.
    a_pv_cum = pv.iloc[:3].cumsum().values
    a_v_cum = df["volume"].iloc[:3].cumsum().values
    expected_a = a_pv_cum / a_v_cum

    b_pv_cum = pv.iloc[3:].cumsum().values
    b_v_cum = df["volume"].iloc[3:].cumsum().values
    expected_b = b_pv_cum / b_v_cum

    np.testing.assert_allclose(out.iloc[:3].values, expected_a)
    np.testing.assert_allclose(out.iloc[3:].values, expected_b)


def test_rolling_vwap_falls_back_when_no_session_label() -> None:
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    df = pd.DataFrame(
        {
            "high":   [10.0, 11.0, 12.0, 13.0, 14.0],
            "low":    [9.0,  10.0, 11.0, 12.0, 13.0],
            "close":  [9.5,  10.5, 11.5, 12.5, 13.5],
            "volume": [100,  100,  100,  100,  100],
        },
        index=idx,
    )
    out = session_vwap(df)
    assert out.notna().all()


# ---------------------------------------------------------------------------
# Volume + candle anatomy
# ---------------------------------------------------------------------------
def test_volume_ratio_matches_definition() -> None:
    v = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    out = volume_ratio(v, window=2)
    rolling_mean = v.rolling(2).mean()
    expected = v / rolling_mean
    pd.testing.assert_series_equal(out, expected)


def test_candle_components_preserve_arithmetic() -> None:
    df = pd.DataFrame(
        {
            "open":  [10.0, 12.0],
            "high":  [11.0, 12.5],
            "low":   [9.5,  11.5],
            "close": [10.5, 11.8],
        }
    )
    np.testing.assert_allclose(candle_body(df).values, [0.5, 0.2])
    np.testing.assert_allclose(upper_wick(df).values, [11.0 - 10.5, 12.5 - 12.0])
    np.testing.assert_allclose(lower_wick(df).values, [10.0 - 9.5, 11.8 - 11.5])
    np.testing.assert_array_equal(direction(df).values, [1, -1])


# ---------------------------------------------------------------------------
# Returns + recent high/low
# ---------------------------------------------------------------------------
def test_rolling_return_matches_pct_change() -> None:
    c = pd.Series([100.0, 110.0, 121.0])
    out = rolling_return(c, 1)
    np.testing.assert_allclose(out.dropna().values, [0.10, 0.10])


def test_recent_high_excludes_current_bar() -> None:
    """Classic bug check: recent_high(t) must NOT include high[t]."""
    df = pd.DataFrame(
        {
            "high": [1.0, 2.0, 3.0, 100.0, 4.0],
            "low":  [0.5, 1.5, 2.5, 90.0,  3.5],
        }
    )
    rh = recent_high(df, lookback=2)
    rl = recent_low(df, lookback=2)

    # Row 3 (high=100). Lookback=2 with shift(1) sees rows 1 and 2 only — the
    # spike at row 3 must not contaminate its own feature.
    assert rh.iloc[3] == 3.0  # max(high[1]=2.0, high[2]=3.0)
    assert rl.iloc[3] == 1.5  # min(low[1]=1.5,  low[2]=2.5)

    # Row 4 — now the spike at row 3 should appear in the lookback window.
    assert rh.iloc[4] == 100.0
    assert rl.iloc[4] == 2.5


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------
def test_volatility_regime_buckets() -> None:
    np.random.seed(0)
    atr_series = pd.Series(np.random.uniform(0, 10, 500))
    out = volatility_regime(atr_series, window=100)
    valid = out.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0, 2.0})
    # Roughly balanced (each bucket ~ 1/3 of valid).
    counts = valid.value_counts(normalize=True)
    for b in (0.0, 1.0, 2.0):
        assert counts.get(b, 0) > 0.15


def test_trend_regime_encoding() -> None:
    fast = pd.Series([3.0, 1.0, 2.0])
    mid =  pd.Series([2.0, 2.0, 1.5])
    slow = pd.Series([1.0, 3.0, 1.8])
    out = trend_regime(fast, mid, slow)
    assert out.tolist() == [1, -1, 0]
