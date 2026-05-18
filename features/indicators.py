"""Technical indicators — pure functions, no lookahead.

Every function in this module is a deterministic transformation of an
input DataFrame/Series. The contract: a value at index ``t`` may use only
data at indices ``<= t``. Centered windows, ``shift(-k)``, and
forward-looking interpolation are all forbidden.

The leakage test (``validation.leakage_checks.assert_no_lookahead``)
exercises this contract end-to-end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages / volatility
# ---------------------------------------------------------------------------
def ema(close: pd.Series, period: int) -> pd.Series:
    """Exponential moving average. ``adjust=False`` so it depends only on past."""
    if period <= 0:
        raise ValueError("ema period must be positive")
    return close.ewm(span=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Average True Range.

    TR_t = max(high_t - low_t, |high_t - close_{t-1}|, |low_t - close_{t-1}|)
    ATR_t = Wilder smoothing of TR with alpha = 1/period
    """
    if period <= 0:
        raise ValueError("atr period must be positive")
    high = df["high"]
    low = df["low"]
    prev_close = df["close"].shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # First bar has no prev close → fall back to high - low for that bar.
    tr.iloc[0] = high.iloc[0] - low.iloc[0] if len(df) else tr.iloc[0]

    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


# ---------------------------------------------------------------------------
# VWAP
# ---------------------------------------------------------------------------
def session_vwap(
    df: pd.DataFrame,
    session_label: pd.Series | None = None,
) -> pd.Series:
    """Volume-weighted average price.

    - If ``session_label`` is provided, VWAP resets every time the label
      changes (futures convention: NY RTH session reset).
    - Otherwise we fall back to a 24h rolling VWAP (crypto convention).
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]

    if session_label is not None:
        groups = (session_label != session_label.shift()).cumsum()
        cum_pv = pv.groupby(groups).cumsum()
        cum_v = df["volume"].groupby(groups).cumsum()
        return cum_pv / cum_v

    if isinstance(df.index, pd.DatetimeIndex):
        win = "24h"
        return pv.rolling(win).sum() / df["volume"].rolling(win).sum()

    return pv.cumsum() / df["volume"].cumsum()


# ---------------------------------------------------------------------------
# Volume / candle anatomy
# ---------------------------------------------------------------------------
def volume_ratio(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by its trailing rolling mean (no lookahead)."""
    if window <= 0:
        raise ValueError("volume_ratio window must be positive")
    return volume / volume.rolling(window).mean()


def candle_body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def direction(df: pd.DataFrame) -> pd.Series:
    diff = df["close"] - df["open"]
    return pd.Series(np.sign(diff).astype(int), index=df.index)


# ---------------------------------------------------------------------------
# Returns / range distance
# ---------------------------------------------------------------------------
def rolling_return(close: pd.Series, periods: int) -> pd.Series:
    if periods <= 0:
        raise ValueError("rolling_return periods must be positive")
    return close.pct_change(periods)


def recent_high(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Highest high over the previous ``lookback`` bars, *excluding* the current bar.

    Using ``shift(1)`` before ``rolling().max()`` is critical — without it, the
    current bar contaminates its own feature.
    """
    if lookback <= 0:
        raise ValueError("recent_high lookback must be positive")
    return df["high"].shift(1).rolling(lookback).max()


def recent_low(df: pd.DataFrame, lookback: int) -> pd.Series:
    if lookback <= 0:
        raise ValueError("recent_low lookback must be positive")
    return df["low"].shift(1).rolling(lookback).min()


# ---------------------------------------------------------------------------
# Regimes
# ---------------------------------------------------------------------------
def volatility_regime(atr_series: pd.Series, window: int = 100) -> pd.Series:
    """Low / mid / high volatility bucket (0/1/2).

    Quantiles are computed over the previous ``window`` bars (excluding
    current) so the bucket label uses past distribution only.
    """
    if window <= 0:
        raise ValueError("volatility_regime window must be positive")
    past = atr_series.shift(1)
    q33 = past.rolling(window).quantile(0.33)
    q66 = past.rolling(window).quantile(0.66)
    out = pd.Series(np.nan, index=atr_series.index, dtype=float)
    out = out.mask(atr_series <= q33, 0.0)
    out = out.mask((atr_series > q33) & (atr_series <= q66), 1.0)
    out = out.mask(atr_series > q66, 2.0)
    return out


def trend_regime(
    ema_fast: pd.Series, ema_mid: pd.Series, ema_slow: pd.Series
) -> pd.Series:
    """1 = stacked up, -1 = stacked down, 0 = mixed."""
    up = (ema_fast > ema_mid) & (ema_mid > ema_slow)
    down = (ema_fast < ema_mid) & (ema_mid < ema_slow)
    out = pd.Series(0, index=ema_fast.index, dtype=int)
    out = out.mask(up, 1)
    out = out.mask(down, -1)
    return out
