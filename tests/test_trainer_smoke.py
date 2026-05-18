"""Trainer wiring sanity check.

NOT a profitability test — only verifies that the training pipeline runs
end-to-end on a deterministic synthetic feature matrix and produces a
calibrated model that beats random.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.feature_builder import FEATURE_COLUMNS
from models.trainer import has_lightgbm, train


def _synthetic_labeled_features(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    cols = list(FEATURE_COLUMNS)
    X = pd.DataFrame(rng.standard_normal((n, len(cols))), columns=cols)
    # Inject a real signal: label is positive iff a linear combo of the first 3
    # features exceeds 0. This is what the model is supposed to learn.
    score = 1.5 * X.iloc[:, 0] + 0.8 * X.iloc[:, 1] - 0.5 * X.iloc[:, 2]
    y = (score + 0.3 * rng.standard_normal(n) > 0).astype(int)
    X.index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    y.index = X.index
    y.name = "label"
    return X, y


def test_logreg_trains_and_beats_random() -> None:
    X, y = _synthetic_labeled_features(n=400)
    n_train = 280
    n_val = 60
    X_train, y_train = X.iloc[:n_train], y.iloc[:n_train]
    X_val, y_val = X.iloc[n_train : n_train + n_val], y.iloc[n_train : n_train + n_val]

    result = train(X_train, y_train, X_val, y_val, model_kind="logreg")
    assert result.aggregate_metrics["roc_auc"] > 0.5
    assert len(result.calibration_table) == 10
    assert result.feature_names == list(FEATURE_COLUMNS)
    assert result.n_train == n_train
    assert result.n_val == n_val


def test_logreg_produces_calibrated_probabilities() -> None:
    X, y = _synthetic_labeled_features(n=300)
    X_train, y_train = X.iloc[:200], y.iloc[:200]
    X_val, y_val = X.iloc[200:260], y.iloc[200:260]

    result = train(X_train, y_train, X_val, y_val, model_kind="logreg")
    # All predict_proba outputs must be in [0, 1] (calibration didn't break anything).
    proba = result.estimator.predict_proba(X_val)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_lightgbm_optional_smoke() -> None:
    if not has_lightgbm():
        # Trainer must not fail merely because LightGBM is missing.
        return
    X, y = _synthetic_labeled_features(n=400)
    result = train(
        X.iloc[:280], y.iloc[:280],
        X.iloc[280:340], y.iloc[280:340],
        model_kind="lightgbm",
    )
    assert result.aggregate_metrics["roc_auc"] > 0.5
