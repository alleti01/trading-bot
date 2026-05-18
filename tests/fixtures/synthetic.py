"""Synthetic OHLCV generators used by the smoke run and several tests.

Kept under ``tests/fixtures`` so it doesn't leak into the production import
graph but stays available as an importable helper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_ohlcv(
    n_bars: int = 500,
    seed: int = 42,
    start: str = "2024-01-15 09:30",
    tz: str = "America/New_York",
    base_price: float = 4500.0,
) -> pd.DataFrame:
    """Generate a believable 1-minute OHLCV DataFrame.

    Trends up for the first half, then reverses — so a trend-pullback
    strategy has a chance of producing both long and short setups in the
    same series.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, periods=n_bars, freq="1min", tz=tz)

    half = n_bars // 2
    drift_up = np.full(half, 0.05 / 1000.0)
    drift_dn = np.full(n_bars - half, -0.05 / 1000.0)
    drift = np.concatenate([drift_up, drift_dn])

    returns = rng.normal(0, 0.0008, n_bars) + drift
    close = base_price * np.exp(np.cumsum(returns))

    open_ = np.r_[close[0], close[:-1]]
    upper_noise = np.abs(rng.normal(0, 0.0006, n_bars))
    lower_noise = np.abs(rng.normal(0, 0.0006, n_bars))
    high = np.maximum(open_, close) * (1.0 + upper_noise)
    low = np.minimum(open_, close) * (1.0 - lower_noise)
    volume = np.abs(rng.normal(1000, 200, n_bars)) + 1.0

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
    df.index.name = "timestamp"
    return df
