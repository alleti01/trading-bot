"""Chronological split — disjoint, contiguous, ordered."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from validation.time_split import chronological_split


def _df(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"x": np.arange(n)}, index=idx)


def test_split_is_disjoint_and_contiguous() -> None:
    df = _df(100)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    # No overlaps.
    assert set(train).isdisjoint(val)
    assert set(val).isdisjoint(test)
    assert set(train).isdisjoint(test)
    # Contiguous.
    assert train[-1] + 1 == val[0]
    assert val[-1] + 1 == test[0]
    # Covers everything.
    assert sorted(np.concatenate([train, val, test]).tolist()) == list(range(100))


def test_split_is_chronological() -> None:
    df = _df(50)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert df.index[train].max() < df.index[val].min()
    assert df.index[val].max() < df.index[test].min()


def test_split_sizes_match_fractions_within_one() -> None:
    df = _df(1000)
    train, val, test = chronological_split(df, train_frac=0.7, val_frac=0.15)
    assert abs(len(train) - 700) <= 1
    assert abs(len(val) - 150) <= 1
    assert abs(len(test) - 150) <= 1


def test_rejects_non_monotonic_index() -> None:
    df = _df(20)
    df = df.iloc[[5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]  # shuffle the first 10
    with pytest.raises(ValueError, match="monotonically increasing"):
        chronological_split(df)


def test_rejects_non_datetime_index() -> None:
    df = pd.DataFrame({"x": np.arange(10)})
    with pytest.raises(ValueError, match="DatetimeIndex"):
        chronological_split(df)


def test_rejects_invalid_fractions() -> None:
    df = _df(100)
    with pytest.raises(ValueError):
        chronological_split(df, train_frac=0.0)
    with pytest.raises(ValueError):
        chronological_split(df, train_frac=0.6, val_frac=-0.1)
    with pytest.raises(ValueError, match="< 1"):
        chronological_split(df, train_frac=0.7, val_frac=0.4)


def test_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError):
        chronological_split(_df(2))
