"""Static + dynamic checks against future-looking features.

The dynamic check (``assert_no_lookahead``) is the headline test: it
recomputes the entire feature DataFrame after corrupting every bar
*after* index ``t`` and asserts the row at ``t`` is identical. Any
``shift(-k)``, centered window, or accidental future-include will trip
this immediately.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd


BuildFn = Callable[..., pd.DataFrame]


def _corrupt_after(
    df: pd.DataFrame,
    t_index: int,
    seed: int = 1234,
) -> pd.DataFrame:
    """Replace every OHLCV value after ``t_index`` with valid-but-random data."""
    out = df.copy()
    n = len(out)
    if t_index >= n - 1:
        return out

    rng = np.random.default_rng(seed)
    n_after = n - t_index - 1

    # Generate OHLC that satisfies high >= max(open,close) and low <= min(open,close).
    o = rng.uniform(50.0, 150.0, size=n_after)
    c = rng.uniform(50.0, 150.0, size=n_after)
    high_extra = np.abs(rng.standard_normal(n_after))
    low_extra = np.abs(rng.standard_normal(n_after))
    h = np.maximum(o, c) + high_extra
    lo = np.minimum(o, c) - low_extra
    v = np.abs(rng.standard_normal(n_after)) * 1000.0 + 1.0

    cols = ["open", "high", "low", "close", "volume"]
    arrays = [o, h, lo, c, v]
    for col, arr in zip(cols, arrays):
        col_idx = out.columns.get_loc(col)
        out.iloc[t_index + 1 :, col_idx] = arr

    return out


def assert_no_lookahead(
    build_fn: BuildFn,
    df: pd.DataFrame,
    t_index: int,
    *,
    build_kwargs: Optional[dict] = None,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    """Assert that ``build_fn(df).loc[t]`` does not depend on bars after ``t``.

    Strategy:
        1. Build features on the full DataFrame.
        2. Build features on a copy where every bar after ``t_index`` is
           replaced with random (but OHLC-valid) noise.
        3. Compare the row at the original timestamp ``df.index[t_index]``.

    Raises ``AssertionError`` listing the offending feature columns.
    """
    if build_kwargs is None:
        build_kwargs = {}

    if t_index < 0 or t_index >= len(df):
        raise ValueError(f"t_index {t_index} out of range [0, {len(df)})")

    target_ts = df.index[t_index]

    full = build_fn(df.copy(), **build_kwargs)
    if target_ts not in full.index:
        raise ValueError(
            f"t_index {t_index} (ts {target_ts}) is in the warm-up window; "
            "pick a later index for a meaningful leakage check."
        )

    corrupted = _corrupt_after(df, t_index)
    after = build_fn(corrupted, **build_kwargs)
    if target_ts not in after.index:
        raise AssertionError(
            f"After corruption, row at {target_ts} disappeared from feature output "
            "— the warm-up region of build_fn depends on future bars."
        )

    expected = full.loc[target_ts]
    actual = after.loc[target_ts]

    bad: list[str] = []
    for col in expected.index:
        e = expected[col]
        a = actual[col]
        if pd.isna(e) and pd.isna(a):
            continue
        if pd.isna(e) or pd.isna(a):
            bad.append(f"{col}: NaN-mismatch (expected={e}, actual={a})")
            continue
        if not np.isclose(e, a, rtol=rtol, atol=atol):
            bad.append(f"{col}: expected={e}, actual={a}, diff={a - e}")

    if bad:
        raise AssertionError(
            f"Lookahead detected at {target_ts} in {len(bad)} feature(s):\n  "
            + "\n  ".join(bad)
        )
