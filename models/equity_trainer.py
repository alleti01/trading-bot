"""Multi-symbol equity model trainer.

Combines per-symbol OHLCV (``data/historical/<SYM>/1m.csv``) across the
universe into one labeled dataset, then trains + registers a single
model the workflow can use to gate entries. Pooling symbols gives the
model far more setups than any single name's 30-day window, which is
what clears the minimum-setup bar and produces a stable classifier.

Reuses the same building blocks as ``MODE=TRAIN``: feature builder,
strategy registry, TP/SL labeler, chronological split, walk-forward CV,
and ``model_registry.save_model``. It only ever writes to the local
model registry — never trades, never touches LIVE.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from app.logging_config import get_logger
from config.equity_allowlist import is_allowed_equity
from config.instruments import register_equity
from config.settings import Settings
from data.csv_loader import load_ohlcv_csv
from features.feature_builder import FEATURE_COLUMNS, build_features
from labeling.tp_sl_labeler import label_setups
from models.model_registry import save_model
from models.trainer import has_lightgbm, train
from strategies.registry import instantiate as instantiate_strategy
from validation.time_split import chronological_split
from validation.walk_forward import walk_forward_splits

_log = get_logger("models.equity_trainer")

MIN_SETUPS = 100
MIN_OHLCV_ROWS = 200


class EquityTrainError(RuntimeError):
    """Raised when the universe training cannot proceed (clear message)."""


@dataclass
class EquityTrainResult:
    model_name: str
    version: str
    n_total_setups: int
    n_pos: int
    symbols_used: list[str]
    test_metrics: dict


def _labeled_rows_for_symbol(
    settings: Settings,
    symbol: str,
    *,
    strategy_name: str,
    max_hold_bars: int,
) -> list[dict]:
    """Build (features + label + ts + symbol) rows for one symbol, or []."""
    sym = symbol.upper()
    path = Path(settings.HISTORICAL_DATA_DIR) / sym / "1m.csv"
    if not path.exists():
        _log.info("equity_train.no_data", symbol=sym)
        return []
    # Equities all share one contract shape; mint a spec if needed so the
    # feature builder can look it up.
    register_equity(sym)
    try:
        df = load_ohlcv_csv(path, sym, "1m", settings.TIMEZONE)
    except ValueError as e:
        _log.warning("equity_train.csv_invalid", symbol=sym, error=str(e))
        return []
    if df.empty or len(df) < MIN_OHLCV_ROWS:
        _log.info("equity_train.too_small", symbol=sym, rows=int(len(df)))
        return []

    features = build_features(df, instrument=sym, tz=settings.TIMEZONE)
    if any(c not in features.columns for c in FEATURE_COLUMNS):
        return []

    strategy = instantiate_strategy(strategy_name, instrument=sym)
    setups = strategy.detect_setups(features)
    if not setups:
        return []
    labels = label_setups(setups, df, max_hold_bars=max_hold_bars)

    rows: list[dict] = []
    for setup, lab in zip(setups, labels):
        row = dict(setup.features)
        row["_label"] = int(lab.label)
        row["_ts"] = setup.timestamp
        row["_symbol"] = sym
        rows.append(row)
    _log.info("equity_train.symbol_rows", symbol=sym, n=len(rows))
    return rows


def train_universe_model(
    settings: Settings,
    *,
    symbols: list[str],
    model_name: str,
    model_kind: str = "logreg",
    strategy_name: str = "vwap_ema_pullback",
    max_hold_bars: Optional[int] = None,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    allowlist_only: bool = True,
) -> EquityTrainResult:
    """Train + register one model from pooled multi-symbol equity setups."""
    if model_kind == "lightgbm" and not has_lightgbm():
        raise EquityTrainError(
            "LightGBM not installed; use model_kind='logreg' or pip install lightgbm."
        )
    max_hold_bars = int(
        max_hold_bars
        if max_hold_bars is not None
        else getattr(settings, "MAX_HOLD_BARS", 20) or 20
    )

    syms = [s.upper() for s in symbols]
    if allowlist_only:
        kept = [s for s in syms if is_allowed_equity(s)]
        dropped = [s for s in syms if s not in kept]
        if dropped:
            _log.info("equity_train.dropped_off_allowlist", dropped=dropped)
        syms = kept
    if not syms:
        raise EquityTrainError("No eligible symbols to train on.")

    all_rows: list[dict] = []
    used: list[str] = []
    for sym in syms:
        rows = _labeled_rows_for_symbol(
            settings, sym, strategy_name=strategy_name, max_hold_bars=max_hold_bars
        )
        if rows:
            all_rows.extend(rows)
            used.append(sym)

    if len(all_rows) < MIN_SETUPS:
        raise EquityTrainError(
            f"Only {len(all_rows)} pooled setups across {len(used)} symbols "
            f"(need >= {MIN_SETUPS}). Download more history or more symbols."
        )

    setup_df = pd.DataFrame(all_rows).set_index("_ts")
    setup_df.index = pd.DatetimeIndex(setup_df.index)
    setup_df = setup_df.sort_index()

    # Pooling symbols means many setups share a minute timestamp. The
    # split needs a monotonic DatetimeIndex and the trainer aligns by a
    # *unique* DatetimeIndex (it reads index.tz). Break ties by nudging
    # duplicates microseconds apart — preserves order, keeps it a unique,
    # strictly-increasing tz-aware DatetimeIndex.
    ts = setup_df.index.to_series().reset_index(drop=True)
    dup_offset = ts.groupby(ts).cumcount()
    setup_df.index = pd.DatetimeIndex(
        (ts + pd.to_timedelta(dup_offset, unit="us")).values
    )
    setup_df = setup_df.sort_index()

    train_idx, val_idx, test_idx = chronological_split(
        setup_df, train_frac=train_frac, val_frac=val_frac
    )
    X = setup_df[list(FEATURE_COLUMNS)]
    y = setup_df["_label"].astype(int)
    n_pos = int(y.sum())
    if y.nunique() < 2:
        raise EquityTrainError(
            f"Single-class labels (positives={n_pos}/{len(y)}); cannot train."
        )
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        raise EquityTrainError(
            "Single class in train/val fold after split; widen data."
        )

    n_combined = len(train_idx) + len(val_idx)
    test_bars = max(20, n_combined // 10)
    train_min = max(50, n_combined - 5 * test_bars)
    purge_bars = max(2, max_hold_bars)
    embargo_bars = max(2, min(max_hold_bars // 4, test_bars - 1))
    wf_splits: list = []
    try:
        wf_splits = list(
            walk_forward_splits(
                X.iloc[:n_combined],
                n_folds=3,
                train_min_bars=train_min,
                test_bars=test_bars,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
            )
        )
    except ValueError as e:
        _log.warning("equity_train.walk_forward_skipped", error=str(e))

    result = train(
        X_train,
        y_train,
        X_val,
        y_val,
        model_kind=model_kind,
        walk_forward_cv=wf_splits,
        features_full_df=X,
        feature_names=list(FEATURE_COLUMNS),
    )

    test_proba = result.estimator.predict_proba(X_test)[:, 1]
    thr = float(settings.CONFIDENCE_THRESHOLD)
    test_pred = (test_proba >= thr).astype(int)
    test_metrics = {
        "n_test": int(len(test_idx)),
        "accuracy": round(float((test_pred == y_test.values).mean()), 4),
        "positive_rate": round(float(y_test.mean()), 4),
        "approve_rate": round(float(test_pred.mean()), 4),
    }

    extra_metadata = {
        "source": "train_universe",
        "asset_class": "equity",
        "symbols_used": used,
        "n_symbols": len(used),
        "timeframe": "1m",
        "strategy": strategy_name,
        "model_kind": model_kind,
        "max_hold_bars": max_hold_bars,
        "confidence_threshold": thr,
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_total_setups": int(len(setup_df)),
        "n_pos": n_pos,
        "positive_rate": round(n_pos / len(setup_df), 4),
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "test_metrics": test_metrics,
    }
    version = save_model(result, name=model_name, extra_metadata=extra_metadata)
    _log.info(
        "equity_train.saved",
        model_name=model_name,
        version=version,
        symbols=used,
        n_setups=len(setup_df),
        test_metrics=test_metrics,
    )
    return EquityTrainResult(
        model_name=model_name,
        version=version,
        n_total_setups=int(len(setup_df)),
        n_pos=n_pos,
        symbols_used=used,
        test_metrics=test_metrics,
    )
