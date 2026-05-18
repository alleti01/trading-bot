"""Inference + the feature-drift refusal.

The predictor is **read-only**. It must not call execution, risk, or any
DB-writing code. The only safety property worth highlighting:

    Before scoring, the predictor compares ``setup.features.keys()``
    against ``metadata.feature_columns`` (the FEATURE_COLUMNS snapshot
    captured at training time). If they differ, ``FeatureDriftError`` is
    raised. This catches the "we added a feature since this model was
    trained" class of bug — the model would otherwise silently score with
    an arbitrary feature ordering and produce confidently-wrong probs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from config.settings import get_settings
from models.model_registry import LoadedModel
from strategies.base import Setup


class FeatureDriftError(RuntimeError):
    """Raised when a Setup's feature set differs from the model's training features."""


@dataclass(frozen=True)
class Prediction:
    probability: float
    approved: bool
    threshold: float
    model_name: str
    model_version: str


class Predictor:
    """Score a ``Setup`` with a previously-saved model.

    Construct with a ``LoadedModel``; call ``predict_setup(setup)``. If
    ``threshold`` is not given, the env-configured ``CONFIDENCE_THRESHOLD``
    is used.
    """

    def __init__(
        self,
        loaded_model: LoadedModel,
        *,
        default_threshold: float | None = None,
    ) -> None:
        self.estimator: Any = loaded_model.estimator
        self.metadata: dict[str, Any] = loaded_model.metadata
        if default_threshold is None:
            default_threshold = get_settings().CONFIDENCE_THRESHOLD
        self.default_threshold: float = float(default_threshold)

    def predict_setup(
        self,
        setup: Setup,
        *,
        threshold: float | None = None,
    ) -> Prediction:
        # 1) Feature-drift check.
        trained_set = set(self.metadata["feature_columns"])
        setup_set = set(setup.features.keys())
        if trained_set != setup_set:
            missing = trained_set - setup_set
            extra = setup_set - trained_set
            raise FeatureDriftError(
                "Feature drift detected: model "
                f"{self.metadata.get('name')!r} v{self.metadata.get('version')!r} "
                "was trained on a different feature set than this Setup provides. "
                f"missing_in_setup={sorted(missing)} extra_in_setup={sorted(extra)}."
            )

        # 2) Score using the model's recorded feature ordering (NOT FEATURE_COLUMNS,
        #    which may have been reordered or filtered between train and now).
        #    We pass a DataFrame (not a raw ndarray) so sklearn pipelines that
        #    were fit with feature names don't emit "X has no feature names" warnings.
        feature_names_used = self.metadata.get("feature_names_used") or list(
            self.metadata["feature_columns"]
        )
        x = pd.DataFrame(
            [[float(setup.features[col]) for col in feature_names_used]],
            columns=feature_names_used,
        )
        proba = float(self.estimator.predict_proba(x)[0, 1])

        thr = float(threshold) if threshold is not None else self.default_threshold
        return Prediction(
            probability=proba,
            approved=proba >= thr,
            threshold=thr,
            model_name=str(self.metadata.get("name", "")),
            model_version=str(self.metadata.get("version", "")),
        )
