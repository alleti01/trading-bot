"""Strictly chronological train / val / test split.

No shuffling, ever. Random splits are the single biggest source of
silently-overfit trading models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_split(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer position arrays ``(train_idx, val_idx, test_idx)``.

    ``test_frac`` is implicitly ``1 - train_frac - val_frac``. The DataFrame's
    index must be a monotonically increasing DatetimeIndex.
    """
    if train_frac <= 0:
        raise ValueError("train_frac must be > 0")
    if val_frac < 0:
        raise ValueError("val_frac must be >= 0")
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1 (need a non-empty test set)")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("chronological_split requires a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "DatetimeIndex must be monotonically increasing — refusing to split a "
            "non-chronological frame to prevent silent leakage."
        )

    n = len(df)
    if n < 3:
        raise ValueError(f"Need at least 3 rows to form three splits; got {n}")

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    if n_train < 1 or n_val < 0 or n - n_train - n_val < 1:
        raise ValueError(
            f"Split fractions produce empty partitions for n={n}: "
            f"train={n_train}, val={n_val}, test={n - n_train - n_val}"
        )

    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n)
    return train_idx, val_idx, test_idx
