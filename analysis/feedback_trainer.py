"""Train a candidate model from the post-trade feedback dataset.

Inputs:
    A list of :class:`FeedbackDatasetRow` carrying the *frozen* feature
    snapshot at entry, the realized win/loss label, and the realized PnL.

Outputs:
    A :class:`CandidateTrainResult` containing both the calibrated
    estimator (passable straight to :func:`models.model_registry.save_model`)
    and a dict of *realized trade metrics* (expectancy, profit factor,
    drawdown proxy, false-positive rate, calibration MAE) computed on the
    chronological holdout test split.

Hard rules enforced here:

- Splits are **always chronological** by ``entry_ts``. Random shuffling
  is rejected explicitly.
- Mistake tags are stored in metadata for inspection but are **not**
  used as labels unless ``use_mistake_tags_as_label=True`` (off by
  default — they are an *input signal*, not ground truth).
- Below ``min_rows`` the trainer raises :class:`InsufficientFeedbackError`.
  The CLI wrapper turns that into a non-zero exit code; nothing is
  written to the model registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from typing import Literal

from analysis.types import FeedbackDatasetRow
from app.logging_config import get_logger
from features.feature_builder import FEATURE_COLUMNS
from models.trainer import TrainResult, has_lightgbm, train
from validation.time_split import chronological_split

CandidateModelKind = Literal["logreg", "lightgbm"]


# ---------------------------------------------------------------------------
# Errors / result types
# ---------------------------------------------------------------------------
class InsufficientFeedbackError(ValueError):
    """Raised when the feedback dataset has too few rows to train safely."""


@dataclass
class CandidateTrainResult:
    """Aggregate output of the feedback retrain pipeline."""

    train_result: TrainResult  # primary artifact (logreg) — what gets saved
    boosted_train_result: Optional[TrainResult]  # optional LightGBM, metrics-only
    realized_metrics: dict[str, float]
    test_metrics: dict[str, float]
    val_metrics: dict[str, float]
    calibration_table: list[dict[str, Any]]
    feature_columns: list[str]
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    train_range: tuple[datetime, datetime]
    val_range: tuple[datetime, datetime]
    test_range: tuple[datetime, datetime]
    mistake_tag_counts: dict[str, int] = field(default_factory=dict)
    label_strategy: str = "pnl_positive"  # or "mistake_tag_inverse"
    excluded_no_features: int = 0


# ---------------------------------------------------------------------------
# Trade-level realized metrics (post-cost)
# ---------------------------------------------------------------------------
def _expectancy(pnls: np.ndarray) -> float:
    """Mean realized PnL per trade (signed)."""
    if pnls.size == 0:
        return 0.0
    return float(np.mean(pnls))


def _profit_factor(pnls: np.ndarray) -> float:
    """Sum of positive PnL / |sum of negative PnL|.

    Returns ``+inf`` if there are gains and zero losses, ``0.0`` if
    only losses, and ``0.0`` for an empty array. Standard convention
    in trading evaluation.
    """
    if pnls.size == 0:
        return 0.0
    gains = float(pnls[pnls > 0].sum())
    losses = float(-pnls[pnls < 0].sum())
    if losses == 0.0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def _max_drawdown_pct(pnls: np.ndarray) -> float:
    """Drawdown proxy on a cumulative-PnL curve.

    ``pct`` is computed against the peak cumulative PnL — not against
    starting capital, which the feedback dataset does not record. This
    is a relative dispersion measure: the larger the value, the deeper
    the worst peak-to-trough excursion of the equity curve seen so far.
    Returns 0.0 when the curve only goes up.
    """
    if pnls.size == 0:
        return 0.0
    equity = np.cumsum(pnls)
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks  # <= 0
    worst = float(-drawdowns.min())
    peak_at_worst = float(peaks[np.argmin(drawdowns)])
    if peak_at_worst <= 0:
        # No positive peak yet — fall back to absolute drawdown vs initial 0.
        return worst
    return worst / peak_at_worst


def _false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """FP / (FP + TN). Returns 0.0 when there are no negatives in y_true."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    negatives = (y_true == 0)
    n_neg = int(negatives.sum())
    if n_neg == 0:
        return 0.0
    fp = int(((y_pred == 1) & negatives).sum())
    return fp / n_neg


def _calibration_mae(y_true: np.ndarray, y_proba: np.ndarray, n_bins: int = 10) -> float:
    """Mean |predicted_mean - realized_rate| across non-empty bins.

    A perfectly calibrated model has 0.0; values above ~0.10 indicate
    the predictor's confidence is unreliable.
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    diffs: list[float] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_proba >= lo) & (y_proba < hi if i < n_bins - 1 else y_proba <= hi)
        n = int(mask.sum())
        if n == 0:
            continue
        diffs.append(abs(float(y_proba[mask].mean()) - float(y_true[mask].mean())))
    if not diffs:
        return 0.0
    return float(np.mean(diffs))


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------
def build_feedback_dataframe(
    rows: list[FeedbackDatasetRow],
    *,
    use_mistake_tags_as_label: bool = False,
) -> pd.DataFrame:
    """Build a chronological DataFrame with one row per trade.

    Index is ``entry_ts`` (timezone-aware, sorted ascending). Columns:
    ``FEATURE_COLUMNS`` + ``_label`` + ``_pnl`` + ``_has_mistake_tag``.

    Rows missing any required feature are dropped — a paper trade
    logged before feature-snapshot persistence was wired cannot be
    used for retraining. The number of dropped rows is communicated
    to the caller via ``df.attrs['excluded_no_features']``.
    """
    records: list[dict[str, Any]] = []
    excluded = 0
    for row in rows:
        if not row.features:
            excluded += 1
            continue
        # Reject rows missing any canonical feature — partial vectors
        # would force imputation, which is silent leakage we refuse
        # to do at retrain time.
        if not all(col in row.features for col in FEATURE_COLUMNS):
            excluded += 1
            continue
        rec: dict[str, Any] = {col: float(row.features[col]) for col in FEATURE_COLUMNS}
        rec["_pnl"] = float(row.realized_pnl)
        rec["_has_mistake_tag"] = bool(row.mistake_tags)
        if use_mistake_tags_as_label:
            # A trade is a "good" example iff it has no mistake tag.
            rec["_label"] = int(not row.mistake_tags)
        else:
            rec["_label"] = int(row.label)  # 1 if pnl > 0
        rec["_entry_ts"] = row.entry_ts
        records.append(rec)

    df = pd.DataFrame.from_records(records)
    if df.empty:
        df.attrs["excluded_no_features"] = excluded
        return df

    df = df.set_index("_entry_ts")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.DatetimeIndex(df.index)
    df = df.sort_index()
    df.attrs["excluded_no_features"] = excluded
    return df


def _evaluate_classification(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Trade-style classification metrics at a fixed decision threshold."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    y_pred = (y_proba >= threshold).astype(int)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": _false_positive_rate(y_true, y_pred),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    metrics["calibration_mae"] = _calibration_mae(y_true, y_proba)
    return metrics


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def train_candidate_from_feedback(
    rows: list[FeedbackDatasetRow],
    *,
    min_rows: int = 100,
    confidence_threshold: float = 0.60,
    use_mistake_tags_as_label: bool = False,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    model_kind: CandidateModelKind = "logreg",
) -> CandidateTrainResult:
    """Run the candidate retrain pipeline end-to-end.

    Steps (in order):

    1. Reject rows with no feature snapshot (cannot retrain without features).
    2. Enforce ``min_rows`` (raises :class:`InsufficientFeedbackError`).
    3. Sort by ``entry_ts`` and chronologically split (train / val / test).
    4. Train the model selected by ``model_kind`` (default ``logreg``);
       calibrate on val. This is the artifact that gets saved.
    5. Train the *other* kind (when available) for metrics-only
       comparison — never saved as the primary artifact.
    6. Compute realized trade metrics (expectancy / profit factor /
       drawdown proxy / FPR / calibration) on the **test** split using the
       saved model's gating threshold.

    Notes on ``model_kind``:

    - ``"logreg"`` (default) is the safe baseline; it always works.
    - ``"lightgbm"`` requires the optional ``lightgbm`` package. If it
      is not installed we raise :class:`InsufficientFeedbackError`
      rather than silently downgrade — the caller asked for a specific
      model.
    """
    log = get_logger("analysis.feedback_trainer")

    if model_kind not in ("logreg", "lightgbm"):
        raise ValueError(f"Unknown model_kind: {model_kind!r}")
    if model_kind == "lightgbm" and not has_lightgbm():
        raise InsufficientFeedbackError(
            "model_kind='lightgbm' requested but the lightgbm package is "
            "not installed. Install it (pip install lightgbm) or use "
            "model_kind='logreg'."
        )

    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        raise ValueError(
            "train_frac + val_frac must leave a non-empty test slice "
            f"(got train_frac={train_frac}, val_frac={val_frac})"
        )

    n_input = len(rows)
    df = build_feedback_dataframe(
        rows, use_mistake_tags_as_label=use_mistake_tags_as_label
    )
    excluded = int(df.attrs.get("excluded_no_features", 0))

    if len(df) < min_rows:
        raise InsufficientFeedbackError(
            f"Need >= {min_rows} feedback rows with full feature snapshots, "
            f"got {len(df)} (input rows={n_input}, excluded for missing "
            f"features={excluded}). Run paper mode for longer or check "
            f"that FeatureSnapshot persistence is wired in the paper loop."
        )

    if df["_label"].nunique() < 2:
        raise InsufficientFeedbackError(
            "All feedback rows have the same label — cannot train a "
            "binary classifier. Wait for both wins and losses to "
            "accumulate before retraining."
        )

    X = df[list(FEATURE_COLUMNS)]
    y = df["_label"].astype(int)
    pnl = df["_pnl"].astype(float)

    # Strictly chronological 3-way split. ``chronological_split``
    # validates monotonic index for us.
    train_idx, val_idx, test_idx = chronological_split(
        df, train_frac=train_frac, val_frac=val_frac
    )
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    pnl_test = pnl.iloc[test_idx]

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise InsufficientFeedbackError(
            "Chronological split produced a fold with a single class. "
            "Collect more diverse trades (both wins and losses across "
            "the recent window) before retraining."
        )

    log.info(
        "feedback_trainer.split",
        n_total=int(len(df)),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        train_start=str(X_train.index[0]),
        train_end=str(X_train.index[-1]),
        val_start=str(X_val.index[0]),
        val_end=str(X_val.index[-1]),
        test_start=str(X_test.index[0]),
        test_end=str(X_test.index[-1]),
    )

    # 1. Primary (saved) model — kind selected by the operator.
    primary_result = train(
        X_train,
        y_train,
        X_val,
        y_val,
        model_kind=model_kind,
        feature_names=list(FEATURE_COLUMNS),
    )

    # 2. Secondary (metrics-only) model: train the *other* kind when
    #    available so the operator can compare. Never saved.
    boosted_result: Optional[TrainResult] = None
    secondary_kind = "lightgbm" if model_kind == "logreg" else "logreg"
    secondary_available = secondary_kind != "lightgbm" or has_lightgbm()
    if secondary_available:
        try:
            boosted_result = train(
                X_train,
                y_train,
                X_val,
                y_val,
                model_kind=secondary_kind,
                feature_names=list(FEATURE_COLUMNS),
            )
        except Exception as e:  # noqa: BLE001 - secondary is best-effort
            log.warning(
                "feedback_trainer.secondary_failed",
                kind=secondary_kind,
                error=str(e),
            )

    # 3. Holdout test metrics using the calibrated primary.
    test_proba = primary_result.estimator.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= confidence_threshold).astype(int)
    test_metrics = _evaluate_classification(
        y_test.values, test_proba, threshold=confidence_threshold
    )

    # 4. Realized trade metrics on the test slice — only on trades the
    #    candidate model would have *approved* (predicted positive).
    approved_mask = test_pred == 1
    approved_pnl = pnl_test.values[approved_mask]
    realized_metrics: dict[str, float] = {
        "n_approved": float(int(approved_mask.sum())),
        "n_test": float(int(len(test_idx))),
        "expectancy_per_trade": _expectancy(approved_pnl),
        "profit_factor": _profit_factor(approved_pnl),
        "max_drawdown_pct": _max_drawdown_pct(approved_pnl),
        "false_positive_rate": test_metrics["false_positive_rate"],
        "calibration_mae": test_metrics["calibration_mae"],
        "win_rate_approved": (
            float((approved_pnl > 0).mean()) if approved_pnl.size > 0 else 0.0
        ),
        "all_trades_expectancy": _expectancy(pnl_test.values),
        "all_trades_profit_factor": _profit_factor(pnl_test.values),
    }

    # 5. Mistake tag counts go into metadata (advisory only).
    tag_counts: dict[str, int] = {}
    for row in rows:
        for tag in row.mistake_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    def _tup(start: datetime, end: datetime) -> tuple[datetime, datetime]:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return start, end

    log.info(
        "feedback_trainer.complete",
        n_train=int(len(train_idx)),
        n_test=int(len(test_idx)),
        test_accuracy=round(test_metrics["accuracy"], 4),
        test_precision=round(test_metrics["precision"], 4),
        test_recall=round(test_metrics["recall"], 4),
        test_fpr=round(test_metrics["false_positive_rate"], 4),
        expectancy=round(realized_metrics["expectancy_per_trade"], 4),
        profit_factor=round(realized_metrics["profit_factor"], 4),
        max_drawdown_pct=round(realized_metrics["max_drawdown_pct"], 4),
        calibration_mae=round(realized_metrics["calibration_mae"], 4),
        boosted_trained=boosted_result is not None,
    )

    return CandidateTrainResult(
        train_result=primary_result,
        boosted_train_result=boosted_result,
        realized_metrics=realized_metrics,
        test_metrics=test_metrics,
        val_metrics=primary_result.aggregate_metrics,
        calibration_table=primary_result.calibration_table,
        feature_columns=list(FEATURE_COLUMNS),
        n_total=int(len(df)),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        train_range=_tup(X_train.index[0].to_pydatetime(), X_train.index[-1].to_pydatetime()),
        val_range=_tup(X_val.index[0].to_pydatetime(), X_val.index[-1].to_pydatetime()),
        test_range=_tup(X_test.index[0].to_pydatetime(), X_test.index[-1].to_pydatetime()),
        mistake_tag_counts=tag_counts,
        label_strategy=("mistake_tag_inverse" if use_mistake_tags_as_label else "pnl_positive"),
        excluded_no_features=excluded,
    )
