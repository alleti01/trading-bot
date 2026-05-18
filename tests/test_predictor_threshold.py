"""Predictor.approved respects the threshold — both default and override."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from features.feature_builder import FEATURE_COLUMNS
from models.model_registry import LoadedModel
from models.predictor import Predictor
from strategies.base import Setup


class _ConstantEstimator:
    def __init__(self, prob: float) -> None:
        self.prob = prob

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.prob), np.full(n, self.prob)])


def _setup() -> Setup:
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


def _loaded(prob: float) -> LoadedModel:
    return LoadedModel(
        estimator=_ConstantEstimator(prob),
        metadata={
            "name": "thr_test",
            "version": "v0",
            "feature_columns": list(FEATURE_COLUMNS),
            "feature_names_used": list(FEATURE_COLUMNS),
        },
    )


def test_high_threshold_blocks_normal_prediction() -> None:
    pred = Predictor(_loaded(0.6), default_threshold=0.5)
    out = pred.predict_setup(_setup(), threshold=0.99)
    assert out.approved is False
    assert out.threshold == 0.99


def test_low_threshold_approves_normal_prediction() -> None:
    pred = Predictor(_loaded(0.6), default_threshold=0.5)
    out = pred.predict_setup(_setup(), threshold=0.01)
    assert out.approved is True
    assert out.threshold == 0.01


def test_default_threshold_used_when_none_passed() -> None:
    pred = Predictor(_loaded(0.6), default_threshold=0.7)
    out = pred.predict_setup(_setup())
    assert out.threshold == 0.7
    assert out.approved is False  # 0.6 < 0.7


def test_threshold_at_exact_probability_approves() -> None:
    """approved = probability >= threshold (note the >=)."""
    pred = Predictor(_loaded(0.6), default_threshold=0.5)
    out = pred.predict_setup(_setup(), threshold=0.6)
    assert out.approved is True
