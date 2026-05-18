"""Walk-forward expanding-window validation.

We use an **expanding window** (train always grows fold-over-fold) rather
than a sliding window. The trade-off:

- Expanding gives the model more history each fold, which usually helps
  with stable distributions.
- Sliding adapts faster to regime change but throws away earlier signal.

For a paper-first MVP, expanding is the safer default. We can always swap
to sliding later by implementing a second function — the call site signature
won't change because it iterates over ``(fold_id, train_idx, test_idx)``.

Layout
------
The N test windows are placed so that the **last** test window ends
exactly at ``len(df)``. Test windows are disjoint, each of size
``test_bars``. For each fold:

    test  = [test_start + embargo : test_start + test_bars)
    train = [0 : test_start - purge)

``purge`` removes the last ``purge_bars`` of training right before the
test window — this prevents label-leakage when a setup's max-hold spans
across the boundary. ``embargo`` removes the first ``embargo_bars`` of
test for the same reason.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class WalkForwardSplit(NamedTuple):
    fold_id: int
    train_idx: np.ndarray
    test_idx: np.ndarray


def walk_forward_splits(
    df: pd.DataFrame,
    *,
    n_folds: int,
    train_min_bars: int,
    test_bars: int,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> list[WalkForwardSplit]:
    """Build the list of walk-forward splits eagerly.

    We return a list (not a generator) so parameter validation surfaces at
    call time. With a generator, errors only fire on first iteration which
    masks misconfigurations until much later in the call stack.
    """
    if n_folds <= 0:
        raise ValueError("n_folds must be positive")
    if train_min_bars <= 0:
        raise ValueError("train_min_bars must be positive")
    if test_bars <= 0:
        raise ValueError("test_bars must be positive")
    if purge_bars < 0 or embargo_bars < 0:
        raise ValueError("purge_bars and embargo_bars must be non-negative")
    if embargo_bars >= test_bars:
        raise ValueError("embargo_bars must be < test_bars")

    n = len(df)
    required = train_min_bars + purge_bars + n_folds * test_bars
    if n < required:
        raise ValueError(
            f"Not enough rows for {n_folds} folds: need at least {required} "
            f"(train_min={train_min_bars}, test_bars={test_bars}, purge={purge_bars}); "
            f"got {n}."
        )

    splits: list[WalkForwardSplit] = []
    for i in range(n_folds):
        test_start = n - (n_folds - i) * test_bars
        test_end = test_start + test_bars
        train_end = test_start - purge_bars

        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start + embargo_bars, test_end)

        splits.append(WalkForwardSplit(fold_id=i, train_idx=train_idx, test_idx=test_idx))

    return splits
