"""Process entrypoint.

Day 1: boots config, logging, and the database. With ``--dry-run`` it exits
cleanly after that — this is the smoke test. In future days each ``MODE``
will hand off to its own runner.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from app.logging_config import configure_logging, get_logger
from config.settings import Settings, get_settings, reload_settings
from storage.db import init_db


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="tradeify-bot",
        description="AI-assisted futures/crypto trading bot (paper-first MVP).",
    )
    parser.add_argument(
        "--mode",
        choices=["BACKTEST", "TRAIN", "PAPER", "LIVE"],
        default=None,
        help="Override MODE for this run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Boot, validate config, init DB, then exit. Safe smoke test.",
    )
    parser.add_argument(
        "--smoke-features",
        action="store_true",
        help="Run a Day-2 feature/strategy smoke pass against synthetic OHLCV.",
    )
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="Run a Day-3 train smoke pass: synthetic OHLCV -> setups -> labels -> train -> register -> predict.",
    )
    return parser.parse_args(argv)


def _apply_cli_overrides(args: argparse.Namespace) -> Settings:
    if args.mode is not None:
        os.environ["MODE"] = args.mode
        return reload_settings()
    return get_settings()


def _run_smoke_features(settings: Settings, log) -> int:
    """Day-2 smoke pass: synthetic OHLCV → features → strategy."""
    # Local imports keep CLI startup snappy and avoid hard deps from --dry-run.
    from features.feature_builder import FEATURE_COLUMNS, build_features
    from strategies.vwap_ema_pullback import VWAPEMAPullback
    from tests.fixtures.synthetic import synthetic_ohlcv

    df = synthetic_ohlcv(n_bars=500, tz=settings.TIMEZONE)
    log.info(
        "smoke.ohlcv",
        rows=len(df),
        first_ts=str(df.index.min()),
        last_ts=str(df.index.max()),
    )

    features = build_features(df, instrument=settings.INSTRUMENT, tz=settings.TIMEZONE)
    log.info(
        "smoke.features",
        rows=len(features),
        feature_columns=len(FEATURE_COLUMNS),
        first_ts=str(features.index.min()) if len(features) else None,
        last_ts=str(features.index.max()) if len(features) else None,
        non_null_cells=int(features[list(FEATURE_COLUMNS)].notna().sum().sum()),
    )

    strategy = VWAPEMAPullback(instrument=settings.INSTRUMENT)
    setups = strategy.detect_setups(features)
    by_dir: dict[str, int] = {"long": 0, "short": 0}
    for s in setups:
        by_dir[s.direction] = by_dir.get(s.direction, 0) + 1

    log.info(
        "smoke.setups",
        strategy=strategy.name,
        total=len(setups),
        long=by_dir["long"],
        short=by_dir["short"],
    )
    log.info("smoke.complete", message="Day 2 smoke run passed.")
    return 0


def _run_smoke_train(settings: Settings, log) -> int:
    """Day-3 train smoke: OHLCV -> features -> setups -> labels -> train -> register -> predict."""
    import pandas as pd

    from features.feature_builder import FEATURE_COLUMNS, build_features
    from labeling.tp_sl_labeler import label_setups
    from models.model_registry import load_model, save_model
    from models.predictor import Predictor
    from models.trainer import has_lightgbm, train
    from strategies.vwap_ema_pullback import VWAPEMAPullback
    from tests.fixtures.synthetic import synthetic_ohlcv
    from validation.time_split import chronological_split
    from validation.walk_forward import walk_forward_splits

    df = synthetic_ohlcv(n_bars=10_000, tz=settings.TIMEZONE)
    log.info("smoke.ohlcv", rows=len(df), first_ts=str(df.index.min()), last_ts=str(df.index.max()))

    features = build_features(df, instrument=settings.INSTRUMENT, tz=settings.TIMEZONE)
    log.info("smoke.features", rows=len(features))

    strategy = VWAPEMAPullback(instrument=settings.INSTRUMENT)
    setups = strategy.detect_setups(features)
    log.info(
        "smoke.setups",
        n=len(setups),
        long=sum(s.direction == "long" for s in setups),
        short=sum(s.direction == "short" for s in setups),
    )
    if len(setups) < 50:
        log.error("smoke.too_few_setups", n=len(setups), need=50)
        return 4

    labels = label_setups(setups, df, max_hold_bars=20)

    # Build the per-setup training table.
    rows = []
    for setup, lab in zip(setups, labels):
        row = dict(setup.features)
        row["_label"] = lab.label
        row["_ts"] = setup.timestamp
        rows.append(row)
    setup_df = pd.DataFrame(rows).set_index("_ts")
    setup_df.index = pd.DatetimeIndex(setup_df.index)
    if not setup_df.index.is_monotonic_increasing:
        setup_df = setup_df.sort_index()

    y = setup_df["_label"].astype(int)
    X = setup_df[list(FEATURE_COLUMNS)]
    log.info("smoke.label_distribution", positive_rate=float(y.mean()), n=int(len(y)))

    train_idx, val_idx, test_idx = chronological_split(setup_df, train_frac=0.7, val_frac=0.15)
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    # Walk-forward splits over (train + val) — only if there is enough data.
    n_combined = len(train_idx) + len(val_idx)
    test_bars = max(20, n_combined // 10)
    train_min = max(50, n_combined - 5 * test_bars)
    wf_splits: list = []
    try:
        wf_splits = list(
            walk_forward_splits(
                X.iloc[: n_combined],
                n_folds=3,
                train_min_bars=train_min,
                test_bars=test_bars,
                purge_bars=2,
                embargo_bars=2,
            )
        )
    except ValueError as e:
        log.warning("smoke.walk_forward_skipped", error=str(e))

    # Train logistic regression baseline.
    result_lr = train(
        X_train, y_train, X_val, y_val,
        model_kind="logreg",
        walk_forward_cv=wf_splits,
        features_full_df=X,
    )
    log.info(
        "smoke.train.lr.metrics",
        n_train=result_lr.n_train,
        n_val=result_lr.n_val,
        n_folds=len(result_lr.fold_metrics),
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in result_lr.aggregate_metrics.items()},
    )

    # Train LightGBM if installed (optional).
    if has_lightgbm():
        try:
            result_lgb = train(
                X_train, y_train, X_val, y_val,
                model_kind="lightgbm",
                walk_forward_cv=wf_splits,
                features_full_df=X,
            )
            log.info("smoke.train.lgb.metrics", **{k: round(v, 4) if isinstance(v, float) else v for k, v in result_lgb.aggregate_metrics.items()})
        except Exception as e:
            log.warning("smoke.train.lgb.failed", error=str(e))
    else:
        log.info("smoke.train.lgb.skipped", reason="lightgbm not installed")

    # Persist + reload + predict.
    version = save_model(result_lr, name="vwap_ema_pullback_lr")
    loaded = load_model("vwap_ema_pullback_lr", version=version)
    predictor = Predictor(loaded)

    test_setups = [s for s in setups if s.timestamp in set(X_test.index)][:3]
    for s in test_setups:
        pred = predictor.predict_setup(s)
        log.info(
            "smoke.predict",
            ts=str(s.timestamp),
            direction=s.direction,
            probability=round(pred.probability, 4),
            approved=pred.approved,
            threshold=pred.threshold,
        )

    log.info("smoke.complete", message="Day 3 smoke run passed.", model_version=version)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    try:
        settings = _apply_cli_overrides(args)
    except Exception as e:  # Pydantic ValidationError or similar.
        # Logging may not be configured yet, so go straight to stderr.
        print(f"FATAL: settings failed to load: {e}", file=sys.stderr)
        return 2

    configure_logging(level=settings.LOG_LEVEL, json_format=settings.LOG_JSON)
    log = get_logger("app.main")

    log.info(
        "boot",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        mode=settings.MODE,
        instrument=settings.INSTRUMENT,
        market_type=settings.MARKET_TYPE,
        timezone=settings.TIMEZONE,
        live_adapter_confirmed=settings.LIVE_ADAPTER_CONFIRMED,
        dry_run=args.dry_run,
    )

    try:
        init_db()
        log.info("db.initialized", url=settings.DATABASE_URL)
    except Exception as e:
        log.error("db.init_failed", error=str(e))
        return 3

    if args.dry_run:
        log.info("dry_run.complete", message="Day 1 smoke test passed.")
        return 0

    if args.smoke_features:
        return _run_smoke_features(settings, log)

    if args.smoke_train:
        return _run_smoke_train(settings, log)

    # Real mode runners arrive on Days 3–5. For now just say so and exit.
    if settings.MODE == "TRAIN":
        log.warning("mode.not_implemented", mode="TRAIN", note="Day 3 deliverable")
        return 0
    if settings.MODE == "BACKTEST":
        log.warning("mode.not_implemented", mode="BACKTEST", note="Day 4 deliverable")
        return 0
    if settings.MODE == "PAPER":
        log.warning("mode.not_implemented", mode="PAPER", note="Day 5 deliverable")
        return 0
    # LIVE has already been refused by the settings validator if not configured.
    log.warning("mode.live_unsupported", note="No live adapter implemented in this MVP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
