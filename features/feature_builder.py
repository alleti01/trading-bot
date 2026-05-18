"""Compose the canonical feature set from raw OHLCV.

This module is the single source of truth for feature names + ordering.
Strategies, the trainer, the predictor, and the leakage test all consume
``FEATURE_COLUMNS`` so renaming/extending features is one-touch.
"""

from __future__ import annotations

import pandas as pd

from app.logging_config import get_logger
from config.instruments import get_instrument
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


FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_15",
    "ema_9",
    "ema_21",
    "ema_50",
    "vwap",
    "atr_14",
    "volume_ratio_20",
    "body",
    "upper_wick",
    "lower_wick",
    "direction",
    "time_of_day",
    "day_of_week",
    "dist_from_vwap",
    "dist_from_ema21",
    "dist_from_or_high",
    "dist_from_or_low",
    "recent_high_50",
    "recent_low_50",
    "volatility_regime",
    "trend_regime",
    "session_label",
)


def _session_label_for(df: pd.DataFrame, *, market_type: str, tz: str) -> pd.Series:
    """Encode session as integers per bar.

    - futures: 1 if inside RTH (09:30–16:00 NY), else 0.
    - crypto:  always 1 (24/7).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("feature_builder requires a DatetimeIndex")

    if market_type == "crypto":
        return pd.Series(1, index=df.index, dtype=int)

    local = df.index.tz_convert(tz)
    rth_mask = (
        (local.time >= pd.Timestamp("09:30").time())
        & (local.time < pd.Timestamp("16:00").time())
        & (local.dayofweek < 5)
    )
    return pd.Series(rth_mask.astype(int), index=df.index)


def _opening_range(
    df: pd.DataFrame,
    session_label: pd.Series,
    bars_in_or: int,
) -> tuple[pd.Series, pd.Series]:
    """Per-session opening-range high/low, forward-filled inside the session.

    Bars before the OR completes get NaN. The OR window is defined in *bars*
    so this function is timeframe-agnostic; the caller picks ``bars_in_or``.
    """
    or_high = pd.Series(index=df.index, dtype=float)
    or_low = pd.Series(index=df.index, dtype=float)

    in_session = session_label.astype(bool)
    if not in_session.any():
        return or_high, or_low

    # Each contiguous run of "in session" bars is one session.
    boundary = (in_session != in_session.shift()).cumsum()
    sessions = boundary.where(in_session)

    for _, group_idx in df.groupby(sessions, dropna=True).groups.items():
        if len(group_idx) <= bars_in_or:
            continue
        sess = df.loc[group_idx]
        first = sess.iloc[:bars_in_or]
        rest = sess.iloc[bars_in_or:]
        h = first["high"].max()
        lo = first["low"].min()
        or_high.loc[rest.index] = h
        or_low.loc[rest.index] = lo

    return or_high, or_low


def build_features(
    df: pd.DataFrame,
    *,
    instrument: str,
    tz: str,
    bars_in_or: int = 30,
) -> pd.DataFrame:
    """Compute the canonical feature set. Returns a DataFrame with no NaNs."""
    log = get_logger("features.feature_builder")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("build_features requires a DatetimeIndex")

    spec = get_instrument(instrument)

    session_label = _session_label_for(df, market_type=spec.market_type, tz=tz)

    ema_9 = ema(df["close"], 9)
    ema_21 = ema(df["close"], 21)
    ema_50 = ema(df["close"], 50)
    atr_14 = atr(df, 14)

    if spec.market_type == "futures":
        vwap = session_vwap(df, session_label=session_label)
    else:
        vwap = session_vwap(df, session_label=None)

    or_h, or_l = _opening_range(df, session_label, bars_in_or=bars_in_or)

    local = df.index.tz_convert(tz)
    minutes = pd.Series(local.hour * 60 + local.minute, index=df.index, dtype=float)
    time_of_day = minutes / 1440.0

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = rolling_return(df["close"], 1)
    out["ret_5"] = rolling_return(df["close"], 5)
    out["ret_15"] = rolling_return(df["close"], 15)
    out["ema_9"] = ema_9
    out["ema_21"] = ema_21
    out["ema_50"] = ema_50
    out["vwap"] = vwap
    out["atr_14"] = atr_14
    out["volume_ratio_20"] = volume_ratio(df["volume"], 20)
    out["body"] = candle_body(df)
    out["upper_wick"] = upper_wick(df)
    out["lower_wick"] = lower_wick(df)
    out["direction"] = direction(df).astype(float)
    out["time_of_day"] = time_of_day
    out["day_of_week"] = pd.Series(local.dayofweek.astype(float), index=df.index)
    out["dist_from_vwap"] = (df["close"] - vwap) / atr_14
    out["dist_from_ema21"] = (df["close"] - ema_21) / atr_14
    out["dist_from_or_high"] = (df["close"] - or_h) / atr_14
    out["dist_from_or_low"] = (df["close"] - or_l) / atr_14
    out["recent_high_50"] = recent_high(df, 50)
    out["recent_low_50"] = recent_low(df, 50)
    out["volatility_regime"] = volatility_regime(atr_14, 100)
    out["trend_regime"] = trend_regime(ema_9, ema_21, ema_50).astype(float)
    out["session_label"] = session_label.astype(float)

    # Carry the raw OHLCV alongside features — strategies need both.
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df[col]

    out = out[list(FEATURE_COLUMNS) + ["open", "high", "low", "close", "volume"]]

    n_in = len(out)
    out = out.dropna(subset=list(FEATURE_COLUMNS))
    n_dropped = n_in - len(out)

    log.info(
        "features.built",
        instrument=instrument,
        rows_in=n_in,
        rows_out=len(out),
        warmup_dropped=n_dropped,
        feature_count=len(FEATURE_COLUMNS),
    )

    return out
