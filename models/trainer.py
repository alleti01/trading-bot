"""Model training: LR baseline + optional LightGBM, with calibration.

The trainer's job is to take a *labeled* feature matrix (one row per
setup, columns are a subset of ``FEATURE_COLUMNS``, plus a 0/1 label)
and return a ``TrainResult`` carrying:

- the calibrated estimator (output is ``P(label==1)``)
- per-fold walk-forward metrics
- aggregate validation metrics
- a 10-bin calibration table
- per-slice metrics (month, vol regime, trend regime, time-of-day quintile)

Day 3 deliberately does NOT auto-tune the decision threshold. The
``CONFIDENCE_THRESHOLD`` from settings is the gate; threshold selection
by post-cost expectancy belongs in Day 4 (backtester).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

try:  # sklearn 1.6+
    from sklearn.frozen import FrozenEstimator  # type: ignore[attr-defined]
    _HAVE_FROZEN_ESTIMATOR = True
except ImportError:  # pragma: no cover - fallback for sklearn < 1.6
    _HAVE_FROZEN_ESTIMATOR = False
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.logging_config import get_logger
from validation.walk_forward import WalkForwardSplit

ModelKind = Literal["logreg", "lightgbm"]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FoldMetrics:
    fold_id: int
    train_size: int
    test_size: int
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision_at_60: float
    recall_at_60: float


@dataclass
class TrainResult:
    estimator: Any  # calibrated estimator; ``predict_proba`` returns calibrated probs
    feature_names: list[str]
    fold_metrics: list[FoldMetrics]
    aggregate_metrics: dict[str, float]
    calibration_table: list[dict[str, Any]]
    slice_metrics: dict[str, dict[str, dict[str, float]]]
    train_start: datetime
    train_end: datetime
    val_start: datetime
    val_end: datetime
    n_train: int
    n_val: int
    model_kind: str
    params: dict[str, Any]


# ---------------------------------------------------------------------------
# Optional LightGBM
# ---------------------------------------------------------------------------
def has_lightgbm() -> bool:
    try:
        importlib.import_module("lightgbm")
        return True
    except ImportError:
        return False


def _default_params(model_kind: ModelKind) -> dict[str, Any]:
    if model_kind == "logreg":
        return {"class_weight": "balanced", "max_iter": 1000}
    if model_kind == "lightgbm":
        return {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "verbose": -1,
        }
    return {}


def _make_estimator(model_kind: ModelKind, params: dict[str, Any]) -> Any:
    if model_kind == "logreg":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(**params)),
            ]
        )
    if model_kind == "lightgbm":
        if not has_lightgbm():
            raise RuntimeError("lightgbm is not installed")
        import lightgbm as lgb

        return lgb.LGBMClassifier(**params)
    raise ValueError(f"Unknown model_kind: {model_kind}")


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def _evaluate(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.6) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_at_60": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_at_60": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    # roc_auc / pr_auc require both classes present.
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")
    return metrics


def _calibration_table(
    y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10
) -> list[dict[str, Any]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    table: list[dict[str, Any]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (y_proba >= lo) & (y_proba < hi)
        else:
            mask = (y_proba >= lo) & (y_proba <= hi)
        n = int(mask.sum())
        table.append(
            {
                "bin_lo": float(lo),
                "bin_hi": float(hi),
                "n_samples": n,
                "predicted_mean": float(y_proba[mask].mean()) if n > 0 else None,
                "realized_rate": float(y_true[mask].mean()) if n > 0 else None,
            }
        )
    return table


def _slice_metrics(
    features_df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    min_samples: int = 5,
) -> dict[str, dict[str, dict[str, float]]]:
    """Slice metrics by month / vol-regime / trend-regime / time-of-day quintile."""
    df = features_df.copy()
    df["_y_true"] = y_true
    df["_y_proba"] = y_proba

    out: dict[str, dict[str, dict[str, float]]] = {}

    # by month — drop tz first; to_period() warns on tz-aware indexes.
    naive_index = df.index.tz_convert("UTC").tz_localize(None) if df.index.tz else df.index
    df["_month"] = naive_index.to_period("M").astype(str)
    by_month: dict[str, dict[str, float]] = {}
    for m, sub in df.groupby("_month"):
        if len(sub) >= min_samples:
            by_month[str(m)] = {**_evaluate(sub["_y_true"].values, sub["_y_proba"].values), "n": float(len(sub))}
    out["by_month"] = by_month

    # by volatility_regime / trend_regime — only if columns exist on the slice frame
    for col, key in [("volatility_regime", "by_volatility_regime"), ("trend_regime", "by_trend_regime")]:
        if col in df.columns:
            buckets: dict[str, dict[str, float]] = {}
            for v, sub in df.groupby(col):
                if len(sub) >= min_samples:
                    buckets[str(v)] = {**_evaluate(sub["_y_true"].values, sub["_y_proba"].values), "n": float(len(sub))}
            out[key] = buckets

    # by time-of-day quintile
    if "time_of_day" in df.columns:
        try:
            df["_tod_q"] = pd.qcut(df["time_of_day"], 5, labels=False, duplicates="drop")
        except ValueError:
            df["_tod_q"] = 0
        by_tod: dict[str, dict[str, float]] = {}
        for q, sub in df.groupby("_tod_q"):
            if len(sub) >= min_samples:
                by_tod[f"q{int(q)}"] = {**_evaluate(sub["_y_true"].values, sub["_y_proba"].values), "n": float(len(sub))}
        out["by_time_of_day"] = by_tod

    return out


# ---------------------------------------------------------------------------
# Walk-forward CV
# ---------------------------------------------------------------------------
def walk_forward_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    splits: list[WalkForwardSplit],
    model_kind: ModelKind,
    params: dict[str, Any] | None = None,
) -> list[FoldMetrics]:
    log = get_logger("models.trainer")
    if params is None:
        params = _default_params(model_kind)

    metrics: list[FoldMetrics] = []
    for split in splits:
        X_tr = X.iloc[split.train_idx]
        y_tr = y.iloc[split.train_idx]
        X_te = X.iloc[split.test_idx]
        y_te = y.iloc[split.test_idx]

        if len(np.unique(y_tr)) < 2:
            log.warning("walk_forward.skip_fold", fold_id=split.fold_id, reason="single class in train")
            continue

        est = _make_estimator(model_kind, params)
        est.fit(X_tr, y_tr)
        proba = est.predict_proba(X_te)[:, 1]
        m = _evaluate(y_te.values, proba)
        metrics.append(
            FoldMetrics(
                fold_id=split.fold_id,
                train_size=int(len(X_tr)),
                test_size=int(len(X_te)),
                roc_auc=m["roc_auc"],
                pr_auc=m["pr_auc"],
                accuracy=m["accuracy"],
                precision_at_60=m["precision_at_60"],
                recall_at_60=m["recall_at_60"],
            )
        )

    return metrics


# ---------------------------------------------------------------------------
# Main trainer
# ---------------------------------------------------------------------------
def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    model_kind: ModelKind,
    feature_names: list[str] | None = None,
    walk_forward_cv: list[WalkForwardSplit] | None = None,
    features_full_df: pd.DataFrame | None = None,
    params: dict[str, Any] | None = None,
) -> TrainResult:
    log = get_logger("models.trainer")

    if model_kind == "lightgbm" and not has_lightgbm():
        raise RuntimeError(
            "LightGBM not installed. Install lightgbm or use model_kind='logreg'."
        )

    params = dict(params or _default_params(model_kind))
    feature_names = list(feature_names or list(X_train.columns))

    X_tr = X_train[feature_names]
    X_v = X_val[feature_names]

    # 1. Walk-forward CV (optional)
    fold_metrics: list[FoldMetrics] = []
    if walk_forward_cv is not None:
        # Stack train+val for the CV exercise. (Each fold sees its own time-aware split.)
        X_combined = pd.concat([X_tr, X_v])
        y_combined = pd.concat([y_train, y_val])
        fold_metrics = walk_forward_evaluate(
            X_combined,
            y_combined,
            splits=list(walk_forward_cv),
            model_kind=model_kind,
            params=params,
        )

    # 2. Final fit on train, calibrate on val
    base = _make_estimator(model_kind, params)
    base.fit(X_tr, y_train)

    # Pick calibration method by val size (isotonic needs lots of samples to behave).
    cal_method = "isotonic" if len(X_v) >= 1000 else "sigmoid"
    log.info("calibration.method", method=cal_method, n_val=int(len(X_v)))

    # sklearn 1.6+ replaced ``cv="prefit"`` with the FrozenEstimator wrapper.
    # Wrapping a fitted estimator in FrozenEstimator tells CalibratedClassifierCV
    # not to refit it; the cv parameter then only applies to the calibration split
    # over X_v itself.
    if _HAVE_FROZEN_ESTIMATOR:
        cv_folds = 2 if len(X_v) >= 4 else 2  # min cv must be 2; sigmoid handles tiny folds
        calibrator = CalibratedClassifierCV(
            FrozenEstimator(base), method=cal_method, cv=cv_folds
        )
        calibrator.fit(X_v, y_val)
    else:  # pragma: no cover - older sklearn
        calibrator = CalibratedClassifierCV(base, method=cal_method, cv="prefit")
        calibrator.fit(X_v, y_val)

    # 3. Aggregate metrics on val
    val_proba = calibrator.predict_proba(X_v)[:, 1]
    aggregate = _evaluate(y_val.values, val_proba)

    # 4. Calibration table on val
    cal_table = _calibration_table(y_val.values, val_proba, n_bins=10)

    # 5. Slice metrics on val
    slice_metrics: dict[str, dict[str, dict[str, float]]] = {}
    if features_full_df is not None:
        try:
            val_slice = features_full_df.loc[X_val.index]
        except KeyError:
            val_slice = features_full_df.reindex(X_val.index)
        slice_metrics = _slice_metrics(val_slice, y_val.values, val_proba)

    def _ts(idx, default_pos: int) -> datetime:
        if len(idx) == 0:
            return datetime.now(timezone.utc)
        ts = idx[default_pos]
        return ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts

    return TrainResult(
        estimator=calibrator,
        feature_names=feature_names,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate,
        calibration_table=cal_table,
        slice_metrics=slice_metrics,
        train_start=_ts(X_train.index, 0),
        train_end=_ts(X_train.index, -1),
        val_start=_ts(X_val.index, 0),
        val_end=_ts(X_val.index, -1),
        n_train=int(len(X_train)),
        n_val=int(len(X_val)),
        model_kind=model_kind,
        params=params,
    )
