"""End-to-end no-lookahead test for the real feature builder.

We also include a *negative* test that builds an intentionally-broken
feature builder (uses ``shift(-1)``) and proves the leakage detector
catches it. If the negative test ever stops failing, the leakage
detector itself is broken.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.feature_builder import FEATURE_COLUMNS, build_features
from validation.leakage_checks import assert_no_lookahead

from tests.fixtures.synthetic import synthetic_ohlcv


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return synthetic_ohlcv(n_bars=500, seed=7, tz="America/New_York")


@pytest.mark.parametrize("t_index", [120, 200, 300, 380])
def test_real_builder_passes_no_lookahead(df: pd.DataFrame, t_index: int) -> None:
    """Indices kept inside RTH so dist_from_or_* is defined.

    The synthetic series starts at 09:30 NY; bars after index 389 fall after
    the 16:00 RTH close and ``dist_from_or_*`` becomes NaN (the OR is
    session-scoped). Those rows are dropped by ``build_features`` and
    therefore aren't valid leakage-test targets.
    """
    assert_no_lookahead(
        build_features,
        df,
        t_index,
        build_kwargs={"instrument": "MES", "tz": "America/New_York"},
    )


def _broken_build_features(df: pd.DataFrame, **_kwargs) -> pd.DataFrame:
    """Builder that contaminates current bar with the *next* bar's close.

    This is the classic ``shift(-1)`` lookahead. Used to prove the leakage
    detector actually detects something.
    """
    out = pd.DataFrame(index=df.index)
    out["future_leak"] = df["close"].shift(-1)
    return out.dropna()


def test_broken_builder_with_shift_neg1_is_caught(df: pd.DataFrame) -> None:
    with pytest.raises(AssertionError, match="Lookahead detected"):
        assert_no_lookahead(_broken_build_features, df, t_index=200)


def test_feature_set_is_canonical(df: pd.DataFrame) -> None:
    features = build_features(df, instrument="MES", tz="America/New_York")
    feature_only = [c for c in features.columns if c not in {"open", "high", "low", "close", "volume"}]
    assert tuple(feature_only) == FEATURE_COLUMNS, (
        "Column order in build_features output must match FEATURE_COLUMNS exactly."
    )


def test_no_nan_rows_returned(df: pd.DataFrame) -> None:
    features = build_features(df, instrument="MES", tz="America/New_York")
    assert features[list(FEATURE_COLUMNS)].notna().all().all(), "Feature output must be NaN-free."
