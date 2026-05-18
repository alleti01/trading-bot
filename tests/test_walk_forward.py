"""Walk-forward expanding-window splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.walk_forward import walk_forward_splits


def _df(n: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"x": np.arange(n)}, index=idx)


@pytest.mark.parametrize(
    "n, n_folds, train_min, test_bars, purge, embargo",
    [
        (1000, 5, 200, 100, 0, 0),
        (1000, 5, 200, 100, 5, 5),
        (500,  3, 100, 50,  10, 5),
        (300,  2, 80,  60,  3, 0),
    ],
)
def test_walk_forward_invariants(n, n_folds, train_min, test_bars, purge, embargo) -> None:
    df = _df(n)
    splits = list(
        walk_forward_splits(
            df,
            n_folds=n_folds,
            train_min_bars=train_min,
            test_bars=test_bars,
            purge_bars=purge,
            embargo_bars=embargo,
        )
    )
    assert len(splits) == n_folds

    prev_train_size = -1
    test_windows: list[tuple[int, int]] = []
    for s in splits:
        # Train must end strictly before test starts, with at least purge+embargo gap.
        train_end_exclusive = s.train_idx[-1] + 1 if len(s.train_idx) else 0
        test_start = s.test_idx[0]
        gap = test_start - train_end_exclusive
        assert gap >= purge + embargo, f"fold {s.fold_id}: gap {gap} < purge+embargo {purge + embargo}"

        # Train at least train_min after purge.
        assert len(s.train_idx) >= train_min

        # Train grows fold-over-fold (expanding window).
        assert len(s.train_idx) > prev_train_size
        prev_train_size = len(s.train_idx)

        # Test inside the DataFrame.
        assert s.test_idx[0] >= 0 and s.test_idx[-1] < n
        test_windows.append((int(s.test_idx[0]), int(s.test_idx[-1])))

    # Test windows pairwise disjoint.
    for i in range(len(test_windows)):
        for j in range(i + 1, len(test_windows)):
            a_lo, a_hi = test_windows[i]
            b_lo, b_hi = test_windows[j]
            assert a_hi < b_lo or b_hi < a_lo, f"test folds overlap: {test_windows[i]} {test_windows[j]}"

    # Last fold's test ends exactly at n - 1.
    assert splits[-1].test_idx[-1] == n - 1


def test_walk_forward_rejects_too_little_data() -> None:
    df = _df(100)
    with pytest.raises(ValueError, match="Not enough rows"):
        walk_forward_splits(df, n_folds=5, train_min_bars=200, test_bars=50)


def test_walk_forward_rejects_embargo_geq_test() -> None:
    df = _df(1000)
    with pytest.raises(ValueError, match="embargo_bars must be < test_bars"):
        list(walk_forward_splits(df, n_folds=3, train_min_bars=100, test_bars=50, embargo_bars=50))
