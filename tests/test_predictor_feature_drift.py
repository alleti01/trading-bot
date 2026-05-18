"""Predictor's feature-drift refusal — the headline safety property."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from features.feature_builder import FEATURE_COLUMNS
from models.model_registry import LoadedModel
from models.predictor import FeatureDriftError, Predictor
from strategies.base import Setup


class _ConstantEstimator:
    """Stand-in calibrator that always returns 0.6 for class 1."""

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 0.4), np.full(n, 0.6)])


def _make_setup() -> Setup:
    return Setup(
        instrument="MES",
        timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        strategy_name="test",
        direction="long",
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr_at_entry=1.0,
        features={col: 0.0 for col in FEATURE_COLUMNS},
        bar_index=0,
    )


def test_drift_when_metadata_drops_a_feature() -> None:
    """Loaded model knows fewer features than the Setup carries → refuse."""
    loaded = LoadedModel(
        estimator=_ConstantEstimator(),
        metadata={
            "name": "drift_test",
            "version": "v0",
            "feature_columns": list(FEATURE_COLUMNS[:-1]),  # corrupted: missing one
            "feature_names_used": list(FEATURE_COLUMNS[:-1]),
        },
    )
    pred = Predictor(loaded, default_threshold=0.5)
    with pytest.raises(FeatureDriftError, match="Feature drift detected"):
        pred.predict_setup(_make_setup())


def test_drift_when_metadata_has_extra_feature() -> None:
    """Loaded model expects features the Setup doesn't carry → also refuse."""
    loaded = LoadedModel(
        estimator=_ConstantEstimator(),
        metadata={
            "name": "drift_test",
            "version": "v0",
            "feature_columns": list(FEATURE_COLUMNS) + ["something_new"],
            "feature_names_used": list(FEATURE_COLUMNS) + ["something_new"],
        },
    )
    pred = Predictor(loaded, default_threshold=0.5)
    with pytest.raises(FeatureDriftError):
        pred.predict_setup(_make_setup())


def test_no_drift_when_features_match_canonical() -> None:
    """When metadata.feature_columns == FEATURE_COLUMNS, scoring proceeds."""
    loaded = LoadedModel(
        estimator=_ConstantEstimator(),
        metadata={
            "name": "ok",
            "version": "v0",
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_names_used": list(FEATURE_COLUMNS),
        },
    )
    pred = Predictor(loaded, default_threshold=0.5)
    out = pred.predict_setup(_make_setup())
    assert out.probability == 0.6
    assert out.approved is True
    assert out.threshold == 0.5
    assert out.model_name == "ok"
    assert out.model_version == "v0"
