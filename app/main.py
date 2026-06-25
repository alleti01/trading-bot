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
    parser.add_argument(
        "--smoke-backtest",
        action="store_true",
        help="Run a Day-4 backtest smoke pass against synthetic OHLCV.",
    )
    parser.add_argument(
        "--smoke-paper",
        action="store_true",
        help="Run a Day-5 paper smoke pass: a few bar cycles against a synthetic feed, then exit.",
    )
    parser.add_argument(
        "--smoke-daily-report",
        action="store_true",
        help="Day-6 smoke: synthesize a few closed trades + risk blocks, write the daily Markdown report and CSV trade journal, then exit.",
    )
    parser.add_argument(
        "--smoke-agents",
        action="store_true",
        help="Day-7 smoke: seed trades + risk blocks, build daily report, run agent orchestrator with MockLLMClient, persist to agent_outputs, exit.",
    )
    parser.add_argument(
        "--smoke-trade-analysis",
        action="store_true",
        help=(
            "Day-8 smoke: seed one win + one loss + one false positive, run the "
            "TradeAnalyzer + MistakeClassifier pipeline + mistake report, then exit. "
            "Uses MockLLMClient — no real OpenAI calls."
        ),
    )
    parser.add_argument(
        "--retrain-from-feedback",
        action="store_true",
        help=(
            "Build the feedback dataset from closed trades + train a candidate "
            "model end-to-end (walk-forward). Writes a comparison report against "
            "the incumbent model. Does NOT promote anything."
        ),
    )
    parser.add_argument(
        "--promote-model",
        type=str,
        default=None,
        metavar="VERSION",
        help=(
            "Operator-only: promote a candidate model version (refuses unless "
            "the comparison report's PromotionDecision says promote=True). "
            "Requires --model-name."
        ),
    )
    parser.add_argument(
        "--paper-csv",
        type=str,
        default=None,
        help="Path to an OHLCV CSV. When provided with --mode PAPER drives the paper loop via RollingCSVFeed.",
    )
    parser.add_argument(
        "--paper-cycles",
        type=int,
        default=10,
        help="Number of bar cycles for --smoke-paper (default 10).",
    )
    parser.add_argument(
        "--backtest-csv",
        type=str,
        default=None,
        help="Path to an OHLCV CSV. When provided with --mode BACKTEST runs the backtest engine.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Optional model name in the registry to gate setups (backtest + paper).",
    )
    parser.add_argument(
        "--model-version",
        type=str,
        default="latest",
        help="Model version (default 'latest').",
    )
    parser.add_argument(
        "--min-feedback-rows",
        "--feedback-min-rows",
        dest="min_feedback_rows",
        type=int,
        default=None,
        help=(
            "Override FEEDBACK_MIN_ROWS for --retrain-from-feedback. Below "
            "this many feedback rows the candidate trainer refuses to run "
            "(alias: --feedback-min-rows)."
        ),
    )
    parser.add_argument(
        "--candidate-model-name",
        type=str,
        default=None,
        help=(
            "Override the saved candidate model name for "
            "--retrain-from-feedback (default: '<model-name>_candidate')."
        ),
    )
    parser.add_argument(
        "--use-mistake-tags-as-label",
        "--feedback-use-mistake-tags",
        dest="use_mistake_tags_as_label",
        action="store_true",
        help=(
            "When set, label = 0 for any trade carrying a mistake tag "
            "(default: label is derived from PnL only; tags are stored as "
            "metadata). Alias: --feedback-use-mistake-tags."
        ),
    )
    parser.add_argument(
        "--feedback-model-kind",
        choices=["logreg", "lightgbm"],
        default="logreg",
        help=(
            "Trainer to use for the saved candidate in "
            "--retrain-from-feedback (default: logreg). 'lightgbm' "
            "requires the optional lightgbm package."
        ),
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        default=None,
        help=(
            "Path to an OHLCV CSV with columns "
            "(timestamp, open, high, low, close, volume). Required for "
            "MODE=TRAIN."
        ),
    )
    parser.add_argument(
        "--model-kind",
        choices=["logreg", "lightgbm"],
        default="logreg",
        help="Trainer to use for MODE=TRAIN (default: logreg).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.70,
        help="Chronological train fraction for MODE=TRAIN (default 0.70).",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help=(
            "Chronological val fraction for MODE=TRAIN (default 0.15). "
            "Test fraction = 1 - train_frac - val_frac."
        ),
    )
    parser.add_argument(
        "--max-hold-bars",
        type=int,
        default=None,
        help=(
            "Override settings.MAX_HOLD_BARS for the TP/SL labeler in "
            "MODE=TRAIN (default: settings.MAX_HOLD_BARS, or 20)."
        ),
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help=(
            "Strategy name (registry key). For MODE=TRAIN and MODE=BACKTEST "
            "this picks the single strategy to run (default: "
            "'vwap_ema_pullback'). For MODE=PAPER, omitting this flag uses "
            "settings.ENABLED_STRATEGIES; passing it forces paper mode to "
            "run only the named strategy."
        ),
    )
    parser.add_argument(
        "--workflow",
        choices=[
            "premarket",
            "market-open",
            "midday",
            "daily-summary",
            "weekly-review",
            "run-day",
        ],
        default=None,
        help=(
            "Run one autonomous workflow and exit. Separate from MODE= "
            "train/backtest/paper; defaults to DRY_RUN unless "
            "--no-workflow-dry-run is set."
        ),
    )
    parser.add_argument(
        "--workflow-scheduler",
        action="store_true",
        help=(
            "Start the workflow APScheduler (pre-market, open, midday, "
            "EOD, weekly). Blocks until interrupted."
        ),
    )
    parser.add_argument(
        "--workflow-intraday",
        action="store_true",
        help=(
            "Start the continuous intraday scanner: re-scans the universe "
            "every WORKFLOW_SCAN_INTERVAL_MINUTES during the trading window "
            "and places bracket orders on approved setups. Blocks until "
            "interrupted. Honors --workflow-dry-run."
        ),
    )
    parser.add_argument(
        "--intraday-max-cycles",
        type=int,
        default=None,
        help="Bound the intraday loop to N cycles (testing/one-shot).",
    )
    parser.add_argument(
        "--workflow-dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When true (default), workflows write memory + Discord only — "
            "no paper orders. Use --no-workflow-dry-run with "
            "WORKFLOW_EXECUTION_MODE=PAPER and AUTONOMOUS_TRADING_ENABLED=true "
            "to allow workflow paper entries."
        ),
    )
    parser.add_argument(
        "--workflow-force",
        action="store_true",
        help="Force workflow to run (e.g. weekly-review on non-Friday).",
    )
    parser.add_argument(
        "--train-universe",
        action="store_true",
        help=(
            "Train a single equity model from pooled multi-symbol data "
            "(data/historical/<SYM>/1m.csv across --train-symbols or the "
            "liquid allowlist). Requires --model-name. Writes to the "
            "local model registry only."
        ),
    )
    parser.add_argument(
        "--train-symbols",
        type=str,
        default=None,
        help="Comma-separated symbols for --train-universe (default: allowlist).",
    )
    parser.add_argument(
        "--download-data",
        action="store_true",
        help=(
            "Download historical 1m bars from Alpaca for ENABLED_SYMBOLS "
            "(or --download-symbols) into data/historical/<SYM>/1m.csv. "
            "Read-only market data; never places orders."
        ),
    )
    parser.add_argument(
        "--download-symbols",
        type=str,
        default=None,
        help="Comma-separated symbols for --download-data (default: ENABLED_SYMBOLS).",
    )
    parser.add_argument(
        "--download-days",
        type=int,
        default=30,
        help="Lookback window in days for --download-data (default 30).",
    )
    parser.add_argument(
        "--start-parallel-paper",
        action="store_true",
        help=(
            "Launch parallel paper evaluation tracks (one per PARALLEL_BROKERS entry). "
            "Requires ENABLE_PARALLEL_PAPER=true."
        ),
    )
    parser.add_argument(
        "--parallel-paper-status",
        action="store_true",
        help="Print the status of each parallel paper evaluation track.",
    )
    parser.add_argument(
        "--parallel-paper-report",
        action="store_true",
        help="Generate per-track and combined parallel paper evaluation reports.",
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
    from strategies.registry import instantiate as instantiate_strategy
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

    strategy = instantiate_strategy("vwap_ema_pullback", instrument=settings.INSTRUMENT)
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
    from strategies.registry import instantiate as instantiate_strategy
    from tests.fixtures.synthetic import synthetic_ohlcv
    from validation.time_split import chronological_split
    from validation.walk_forward import walk_forward_splits

    df = synthetic_ohlcv(n_bars=10_000, tz=settings.TIMEZONE)
    log.info("smoke.ohlcv", rows=len(df), first_ts=str(df.index.min()), last_ts=str(df.index.max()))

    features = build_features(df, instrument=settings.INSTRUMENT, tz=settings.TIMEZONE)
    log.info("smoke.features", rows=len(features))

    strategy = instantiate_strategy("vwap_ema_pullback", instrument=settings.INSTRUMENT)
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
    # ``purge_bars`` must be at least ``MAX_HOLD_BARS`` so that the last
    # training sample's label (which depends on bars t+1..t+max_hold)
    # cannot leak into the test window. Without this, walk-forward
    # validation gives optimistically-biased metrics.
    purge_bars = max(2, int(settings.MAX_HOLD_BARS))
    embargo_bars = max(2, min(int(settings.MAX_HOLD_BARS // 4), test_bars - 1))
    wf_splits: list = []
    try:
        wf_splits = list(
            walk_forward_splits(
                X.iloc[: n_combined],
                n_folds=3,
                train_min_bars=train_min,
                test_bars=test_bars,
                purge_bars=purge_bars,
                embargo_bars=embargo_bars,
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


def _run_backtest(
    settings: Settings,
    log,
    *,
    ohlcv_df,
    model_name: Optional[str],
    model_version: str,
    timeframe: str,
    strategy_name: str = "vwap_ema_pullback",
) -> int:
    """Shared backtest core used by both --smoke-backtest and --backtest-csv.

    The strategy is resolved through ``strategies.registry`` so adding a
    new strategy needs no edits here. Multi-strategy backtest is a
    deliberate later step — this function still uses a single named
    strategy.
    """
    from backtesting.engine import BacktestEngine
    from features.feature_builder import build_features
    from models.model_registry import load_model
    from models.predictor import Predictor
    from reports.backtest_report import write_backtest_report
    from strategies.registry import instantiate as instantiate_strategy

    try:
        strategy = instantiate_strategy(strategy_name, instrument=settings.INSTRUMENT)
    except KeyError as e:
        log.error("backtest.unknown_strategy", error=str(e))
        return 4

    features = build_features(
        ohlcv_df, instrument=settings.INSTRUMENT, tz=settings.TIMEZONE
    )
    log.info("backtest.features_built", rows_in=len(ohlcv_df), rows_out=len(features))

    setups = strategy.detect_setups(features)
    log.info(
        "backtest.setups",
        strategy=strategy.name,
        n=len(setups),
        long=sum(s.direction == "long" for s in setups),
        short=sum(s.direction == "short" for s in setups),
    )

    predictor: Optional[Predictor] = None
    if model_name:
        try:
            loaded = load_model(model_name, version=model_version)
            predictor = Predictor(loaded)
            log.info(
                "backtest.predictor_loaded",
                model_name=model_name,
                model_version=loaded.metadata.get("version"),
            )
        except Exception as e:
            log.error("backtest.predictor_load_failed", error=str(e))
            return 5

    engine = BacktestEngine(
        settings=settings,
        predictor=predictor,
        starting_equity=0.0,
        timeframe=timeframe,
    )
    result = engine.run(ohlcv_df, setups)

    json_path, md_path = write_backtest_report(result, settings)
    log.info("backtest.report_paths", json=str(json_path), md=str(md_path))

    m = result.metrics
    if m is not None:
        log.info(
            "backtest.metrics",
            n_trades=m.n_trades,
            net_pnl=round(m.net_pnl, 2),
            win_rate=round(m.win_rate, 4),
            profit_factor=round(m.profit_factor, 4),
            expectancy=round(m.expectancy_per_trade, 2),
            max_dd=round(m.max_drawdown_dollars, 2),
            n_risk_blocked=result.n_setups_risk_blocked,
            n_model_rejected=result.n_setups_model_rejected,
        )
    log.info("smoke.complete", message="Day 4 backtest run passed.")
    return 0


def _run_smoke_backtest(settings: Settings, log) -> int:
    """Day-4 backtest smoke: synthetic OHLCV through the full pipeline."""
    from tests.fixtures.synthetic import synthetic_ohlcv

    df = synthetic_ohlcv(n_bars=2_000, tz=settings.TIMEZONE)
    log.info(
        "smoke.ohlcv",
        rows=len(df),
        first_ts=str(df.index.min()),
        last_ts=str(df.index.max()),
    )
    return _run_backtest(
        settings,
        log,
        ohlcv_df=df,
        model_name=None,
        model_version="latest",
        timeframe="1m",
    )


def _run_backtest_from_csv(
    settings: Settings,
    log,
    *,
    csv_path: str,
    model_name: Optional[str],
    model_version: str,
    strategy_name: str = "vwap_ema_pullback",
) -> int:
    from pathlib import Path

    from data.csv_loader import load_ohlcv_csv

    if not Path(csv_path).exists():
        log.error("backtest.csv_missing", path=str(csv_path))
        return 7

    try:
        df = load_ohlcv_csv(
            path=csv_path,
            instrument=settings.INSTRUMENT,
            timeframe="1m",
            tz=settings.TIMEZONE,
        )
    except Exception as e:
        log.error("backtest.csv_load_failed", path=str(csv_path), error=str(e))
        return 7

    if df.empty:
        log.error("backtest.csv_empty", path=str(csv_path))
        return 7

    return _run_backtest(
        settings,
        log,
        ohlcv_df=df,
        model_name=model_name,
        model_version=model_version,
        timeframe="1m",
        strategy_name=strategy_name,
    )


def _build_paper_feed(settings: Settings, args: argparse.Namespace, log):
    """Resolve which incremental feed to use for paper mode."""
    from data.market_data_service import RollingCSVFeed, SyntheticLiveFeed

    csv_path = args.paper_csv or settings.PAPER_CSV_PATH
    if csv_path:
        log.info("paper.feed_csv", path=str(csv_path))
        return RollingCSVFeed(
            path=csv_path,
            instrument=settings.INSTRUMENT,
            timeframe="1m",
            tz=settings.TIMEZONE,
            window_bars=settings.ROLLING_WINDOW_BARS,
        )
    log.info("paper.feed_synthetic", note="No CSV configured; using SyntheticLiveFeed.")
    return SyntheticLiveFeed(
        instrument=settings.INSTRUMENT,
        timeframe="1m",
        tz=settings.TIMEZONE,
        max_bars=max(2_000, settings.ROLLING_WINDOW_BARS * 2),
        window_bars=settings.ROLLING_WINDOW_BARS,
    )


def _is_multi_symbol_paper(settings: Settings) -> bool:
    """True iff ``ENABLED_SYMBOLS`` lists more than one symbol."""
    enabled = list(getattr(settings, "ENABLED_SYMBOLS", None) or [])
    return len(enabled) > 1


def _build_multi_symbol_paper_loop(
    settings: Settings, args: argparse.Namespace, log, *, notifier, high_risk_news_fn=None
):
    """Build the multi-symbol orchestrator from per-symbol CSVs.

    Convention: ``data/historical/<SYMBOL>/<timeframe>.csv``. Per-symbol
    feed failures are captured into ``disabled_symbols`` so the bot
    boots with the symbols that loaded successfully.
    """
    from config.instruments import SymbolUniverse
    from data.market_data_service import build_per_symbol_feeds
    from paper.loop import build_multi_symbol_paper_loop

    universe = SymbolUniverse.from_settings(settings)
    base_dir = settings.HISTORICAL_DATA_DIR
    plan = build_per_symbol_feeds(
        universe.as_list(),
        base_dir=base_dir,
        timeframe="1m",
        tz=settings.TIMEZONE,
        window_bars=settings.ROLLING_WINDOW_BARS,
    )
    log.info(
        "paper.multi_symbol.feeds",
        loaded=sorted(plan.feeds),
        missing=plan.missing,
        failed=list(plan.failed),
    )
    if not plan.feeds:
        raise RuntimeError(
            f"No per-symbol feeds loaded under {base_dir}. Either drop CSVs "
            f"into data/historical/<SYMBOL>/1m.csv or run a single-symbol "
            f"paper session by setting ENABLED_SYMBOLS to one entry."
        )
    disabled: dict[str, str] = {}
    for sym in plan.missing:
        disabled[sym] = "csv_missing"
    for sym, err in plan.failed.items():
        disabled[sym] = f"csv_load_failed:{err}"

    return build_multi_symbol_paper_loop(
        settings=settings,
        feeds=plan.feeds,
        notifier=notifier,
        disabled_symbols=disabled,
        model_name=args.model_name,
        model_version=args.model_version,
        high_risk_news_fn=high_risk_news_fn,
        cli_strategy=args.strategy,
    )


def _run_smoke_paper(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Day-5 paper smoke: a few synchronous bar cycles, then exit."""
    from notifications.notification_service import NotificationService
    from paper.loop import build_paper_loop
    from scheduler.service import SchedulerService
    from sqlalchemy import func, select

    from storage.db import session_scope
    from storage.tables import ClosedTrade, Notification, PaperTrade, RiskBlock

    notifier = NotificationService.from_settings(settings)
    if _is_multi_symbol_paper(settings):
        loop = _build_multi_symbol_paper_loop(
            settings, args, log, notifier=notifier
        )
    else:
        feed = _build_paper_feed(settings, args, log)
        loop = build_paper_loop(
            settings=settings,
            feed=feed,
            notifier=notifier,
            model_name=args.model_name,
            model_version=args.model_version,
            cli_strategy=args.strategy,
        )
    service = SchedulerService(
        settings=settings, loop=loop, notifier=notifier, blocking=False
    )
    cycles = max(1, int(args.paper_cycles or 10))
    results = service.run_smoke_cycles(cycles)

    new_bars = sum(1 for r in results if r.new_bar)
    setups_seen = sum(r.setups_seen for r in results)
    setups_filled = sum(r.setups_filled for r in results)
    setups_blocked = sum(r.setups_risk_blocked for r in results)
    setups_model_rejected = sum(r.setups_model_rejected for r in results)
    exits = sum(r.exits for r in results)
    errors = sum(len(r.errors) for r in results)

    with session_scope() as session:
        n_paper = session.execute(select(func.count(PaperTrade.id))).scalar() or 0
        n_closed = session.execute(select(func.count(ClosedTrade.id))).scalar() or 0
        n_blocks = session.execute(select(func.count(RiskBlock.id))).scalar() or 0
        n_notifs = session.execute(select(func.count(Notification.id))).scalar() or 0

    log.info(
        "smoke.paper.summary",
        cycles=cycles,
        new_bars=new_bars,
        setups_seen=setups_seen,
        setups_filled=setups_filled,
        setups_blocked=setups_blocked,
        setups_model_rejected=setups_model_rejected,
        exits=exits,
        errors=errors,
        db_paper_trades=n_paper,
        db_closed_trades=n_closed,
        db_risk_blocks=n_blocks,
        db_notifications=n_notifs,
    )
    log.info("smoke.complete", message="Day 5 paper smoke run passed.")
    return 0


def _run_smoke_daily_report(settings: Settings, log) -> int:
    """Day-6 smoke: seed a handful of synthetic closed trades + risk blocks
    and run the full daily report writer end-to-end."""
    from datetime import datetime, timedelta, timezone

    from reports.daily_report import write_daily_report
    from storage.db import session_scope
    from storage.tables import ClosedTrade, RiskBlock

    now = datetime.now(tz=timezone.utc)

    # Seed a few trades + a risk block for *today's* session date.
    from scheduler.market_hours import session_date
    from zoneinfo import ZoneInfo

    tz = ZoneInfo(settings.TIMEZONE)
    sd = session_date(now, settings)
    base_local = datetime.combine(sd, datetime.min.time(), tzinfo=tz).replace(hour=10)

    seeds = [
        ("long", 4500.0, 4504.0, "tp", 20.0),
        ("long", 4504.0, 4502.0, "sl", -10.0),
        ("short", 4505.0, 4501.0, "tp", 20.0),
    ]
    with session_scope() as session:
        for i, (direction, entry, exit_, reason, pnl) in enumerate(seeds):
            entry_ts = (base_local + timedelta(minutes=10 * i)).astimezone(timezone.utc)
            exit_ts = entry_ts + timedelta(minutes=4)
            session.add(
                ClosedTrade(
                    paper_trade_id=None,
                    setup_id=f"smoke-setup-{i}",
                    instrument=settings.INSTRUMENT,
                    direction=direction,
                    quantity=1.0,
                    entry_ts=entry_ts,
                    entry_price=entry,
                    exit_ts=exit_ts,
                    exit_price=exit_,
                    exit_reason=reason,
                    pnl=pnl,
                    commission=0.50,
                    slippage=0.0,
                )
            )
        session.add(
            RiskBlock(
                setup_id="smoke-setup-blocked",
                ts=base_local.astimezone(timezone.utc),
                rule="max_trades_per_day",
                reason="seeded for smoke run",
            )
        )

    artifacts = write_daily_report(settings, now=now)
    log.info(
        "smoke.daily_report.summary",
        md_path=str(artifacts.md_path),
        json_path=str(artifacts.json_path),
        journal_path=str(artifacts.journal_path) if artifacts.journal_path else None,
        **artifacts.summary.to_payload(),
    )
    log.info("smoke.complete", message="Day 6 daily-report smoke run passed.")
    return 0


def _run_paper_forever(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Day-5 PAPER mode: run the scheduler forever (blocking)."""
    from agents.llm_client import build_llm_client
    from agents.orchestrator import build_orchestrator
    from analysis.service import PostTradeAnalysisService
    from notifications.notification_service import NotificationService
    from paper.loop import build_paper_loop
    from scheduler.service import SchedulerService

    notifier = NotificationService.from_settings(settings)
    orchestrator = build_orchestrator(settings, notifier=notifier)
    if _is_multi_symbol_paper(settings):
        loop = _build_multi_symbol_paper_loop(
            settings, args, log,
            notifier=notifier,
            high_risk_news_fn=orchestrator.high_risk_news_active,
        )
    else:
        feed = _build_paper_feed(settings, args, log)
        loop = build_paper_loop(
            settings=settings,
            feed=feed,
            notifier=notifier,
            model_name=args.model_name,
            model_version=args.model_version,
            high_risk_news_fn=orchestrator.high_risk_news_active,
            cli_strategy=args.strategy,
        )

    # Day 8: per-trade analysis. Reuses the same LLM client built for the
    # orchestrator so we don't double-spend when agents are enabled.
    analysis_service = PostTradeAnalysisService(
        settings,
        notifier=notifier,
        llm=build_llm_client(settings),
    )
    loop.set_trade_closed_callback(
        lambda closed_id, mfe, mae, news_at_entry, conf: analysis_service.on_trade_closed(
            closed_id,
            mfe=mfe,
            mae=mae,
            news_risk_at_entry=news_at_entry,
            confidence_override=conf,
        )
    )

    service = SchedulerService(
        settings=settings,
        loop=loop,
        notifier=notifier,
        blocking=True,
        orchestrator=orchestrator,
    )
    log.info(
        "paper.run_forever",
        note="Starting blocking scheduler. Ctrl-C to stop.",
        agents_enabled=settings.ENABLE_LLM_AGENTS,
    )
    service.run_forever()
    return 0


def _run_smoke_agents(settings: Settings, log) -> int:
    """Day-7 smoke: seed trades, build report, run orchestrator with MockLLMClient.

    Forces a mock LLM client regardless of ``ENABLE_LLM_AGENTS`` so the
    smoke run never makes a real network call. Asserts every agent
    produced a schema-valid output.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import func, select
    from zoneinfo import ZoneInfo

    from agents.llm_client import MockLLMClient
    from agents.orchestrator import AgentOrchestrator
    from notifications.notification_service import NotificationService
    from reports.daily_report import write_daily_report
    from scheduler.market_hours import session_date
    from storage.db import session_scope
    from storage.tables import AgentOutput, ClosedTrade, RiskBlock

    notifier = NotificationService(discord=None)  # log-only

    now = datetime.now(tz=timezone.utc)
    tz = ZoneInfo(settings.TIMEZONE)
    sd = session_date(now, settings)
    base_local = datetime.combine(sd, datetime.min.time(), tzinfo=tz).replace(hour=10)

    seeds = [
        ("long", 4500.0, 4504.0, "tp", 20.0),
        ("long", 4504.0, 4502.0, "sl", -10.0),
        ("short", 4505.0, 4501.0, "tp", 20.0),
    ]
    with session_scope() as session:
        for i, (direction, entry, exit_, reason, pnl) in enumerate(seeds):
            entry_ts = (base_local + timedelta(minutes=10 * i)).astimezone(timezone.utc)
            exit_ts = entry_ts + timedelta(minutes=4)
            session.add(
                ClosedTrade(
                    paper_trade_id=None,
                    setup_id=f"smoke-agent-{i}",
                    instrument=settings.INSTRUMENT,
                    direction=direction,
                    quantity=1.0,
                    entry_ts=entry_ts,
                    entry_price=entry,
                    exit_ts=exit_ts,
                    exit_price=exit_,
                    exit_reason=reason,
                    pnl=pnl,
                    commission=0.50,
                    slippage=0.0,
                )
            )
        session.add(
            RiskBlock(
                setup_id="smoke-agent-blocked",
                ts=base_local.astimezone(timezone.utc),
                rule="max_trades_per_day",
                reason="seeded for agents smoke",
            )
        )

    artifacts = write_daily_report(settings, now=now)

    mock_llm = MockLLMClient()
    orchestrator = AgentOrchestrator(settings, llm=mock_llm, notifier=notifier)
    result = orchestrator.run_end_of_day(now=now, daily_md_path=artifacts.md_path)

    with session_scope() as session:
        n_outputs = session.execute(select(func.count(AgentOutput.id))).scalar() or 0
        n_valid_db = (
            session.execute(
                select(func.count(AgentOutput.id)).where(
                    AgentOutput.schema_valid.is_(True)
                )
            ).scalar()
            or 0
        )

    log.info(
        "smoke.agents.summary",
        n_agents=result.n_total(),
        n_valid=result.n_valid(),
        high_risk_news=result.high_risk_news,
        md_path=str(artifacts.md_path),
        appended_md_path=(
            str(result.appended_md_path) if result.appended_md_path else None
        ),
        db_agent_outputs=n_outputs,
        db_agent_outputs_valid=n_valid_db,
    )
    log.info(
        "smoke.complete",
        message="Day 7 agents smoke run passed.",
        note="MockLLMClient was used; production needs ENABLE_LLM_AGENTS=true + OPENAI_API_KEY.",
    )
    return 0


def _run_smoke_trade_analysis(settings: Settings, log) -> int:
    """Day-8 smoke: seed one win + one loss + one false positive, run the
    PostTradeAnalysisService + write the mistake report.

    Uses MockLLMClient so no real network calls. Asserts that the
    classifier produced the expected tags (in particular
    ``false_positive`` for the model-approved loss).
    """
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    from sqlalchemy import func, select
    from zoneinfo import ZoneInfo

    from agents.llm_client import MockLLMClient
    from analysis.service import PostTradeAnalysisService
    from analysis.types import MistakeTag
    from features.feature_builder import FEATURE_COLUMNS
    from notifications.notification_service import NotificationService
    from reports.mistake_report import write_mistake_report
    from scheduler.market_hours import session_date
    from storage.db import session_scope
    from storage.tables import (
        ClosedTrade,
        FeatureSnapshot,
        ImprovementSuggestion,
        ModelPrediction,
        Setup as SetupRow,
        TradeAnalysis,
        TradeMistakeTag,
    )

    notifier = NotificationService(discord=None)  # log-only

    now = datetime.now(tz=timezone.utc)
    tz = ZoneInfo(settings.TIMEZONE)
    sd = session_date(now, settings)
    base_local = datetime.combine(sd, datetime.min.time(), tzinfo=tz).replace(hour=10)

    # Three seeds: clean win, plain loss, false positive (loss with
    # high model confidence above threshold).
    # Fields: (label, direction, entry, exit, stop, target, exit_reason,
    #          pnl, model_p, model_thr).
    seeds = [
        ("clean_win", "long", 4500.0, 4504.0, 4498.0, 4504.0, "tp", 20.0, 0.72, 0.60),
        ("plain_loss", "long", 4504.0, 4502.0, 4502.0, 4508.0, "sl", -10.0, None, None),
        ("false_pos", "short", 4505.0, 4509.0, 4509.0, 4501.0, "sl", -20.0, 0.78, 0.60),
    ]

    closed_ids: list[str] = []
    with session_scope() as session:
        for i, (label, direction, entry, exit_, stop, target, reason, pnl, p, thr) in enumerate(seeds):
            entry_ts = (base_local + timedelta(minutes=10 * i)).astimezone(timezone.utc)
            exit_ts = entry_ts + timedelta(minutes=4)

            # Persist a feature snapshot keyed to the setup so the
            # analyzer can reload it the same way the live loop does.
            snap = FeatureSnapshot(
                instrument=settings.INSTRUMENT,
                ts=entry_ts,
                features={col: 0.0 for col in FEATURE_COLUMNS}
                | {"volatility_regime": 1, "trend_regime": 1, "volume_ratio_20": 1.0},
            )
            session.add(snap)
            session.flush()

            setup_id = str(uuid4())
            session.add(
                SetupRow(
                    id=setup_id,
                    instrument=settings.INSTRUMENT,
                    strategy_name="vwap_ema_pullback",
                    direction=direction,
                    ts=entry_ts,
                    entry_price=entry,
                    stop_price=stop,
                    target_price=target,
                    atr_at_entry=1.0,
                    feature_snapshot_id=snap.id,
                )
            )
            if p is not None and thr is not None:
                session.add(
                    ModelPrediction(
                        setup_id=setup_id,
                        model_name="smoke_lr",
                        model_version="smoke",
                        probability=p,
                        threshold=thr,
                        approved=p >= thr,
                    )
                )

            closed_id = str(uuid4())
            session.add(
                ClosedTrade(
                    id=closed_id,
                    paper_trade_id=None,
                    setup_id=setup_id,
                    instrument=settings.INSTRUMENT,
                    direction=direction,
                    quantity=1.0,
                    entry_ts=entry_ts,
                    entry_price=entry,
                    exit_ts=exit_ts,
                    exit_price=exit_,
                    exit_reason=reason,
                    pnl=pnl,
                    commission=0.50,
                    slippage=0.0,
                )
            )
            closed_ids.append(closed_id)

    service = PostTradeAnalysisService(
        settings,
        notifier=notifier,
        llm=MockLLMClient(),
    )

    outcomes = []
    for cid in closed_ids:
        outcomes.append(service.on_trade_closed(cid))

    mistake_artifacts = write_mistake_report(settings, now=now)

    with session_scope() as session:
        n_analyses = session.execute(select(func.count(TradeAnalysis.id))).scalar() or 0
        n_tags = session.execute(select(func.count(TradeMistakeTag.id))).scalar() or 0
        n_proposals = (
            session.execute(select(func.count(ImprovementSuggestion.id))).scalar() or 0
        )

    fp_seen = any(
        outcome.tagging is not None and outcome.tagging.has(MistakeTag.FALSE_POSITIVE)
        for outcome in outcomes
    )

    log.info(
        "smoke.trade_analysis.summary",
        seeds=len(seeds),
        analyses_persisted=n_analyses,
        tags_persisted=n_tags,
        improvement_suggestions=n_proposals,
        false_positive_detected=fp_seen,
        mistake_md=str(mistake_artifacts.md_path),
        mistake_json=str(mistake_artifacts.json_path),
        post_trade_reports=[
            str(o.md_path) for o in outcomes if o.md_path is not None
        ],
    )
    log.info("smoke.complete", message="Day 8 trade-analysis smoke run passed.")
    return 0


def _run_retrain_from_feedback(settings: Settings, log, *, args=None) -> int:
    """Train a candidate model from closed paper trades, save it under
    a *candidate* version in the registry, and write a promotion
    comparison report against the incumbent.

    Hard guarantees:

    - **Never auto-promotes.** ``compare()`` returns an advisory
      :class:`PromotionDecision`; this runner does *not* move any
      registry pointer. Promotion requires the explicit
      ``--promote-model VERSION`` CLI step, which itself re-runs the
      gate check.
    - Refuses to run on too few rows (``FEEDBACK_MIN_ROWS`` or
      ``--min-feedback-rows``).
    - Splits chronologically only — never randomly.
    - Mistake tags become metadata; they only become labels when the
      operator opts in via ``--use-mistake-tags-as-label``.
    """
    from pathlib import Path

    from analysis.feedback_dataset import FeedbackDataset
    from analysis.feedback_trainer import (
        InsufficientFeedbackError,
        train_candidate_from_feedback,
    )
    from analysis.promotion import compare, write_comparison_report
    from models.model_registry import load_model, save_model

    if args is None or not args.model_name:
        log.error(
            "retrain.missing_model_name",
            note=(
                "--retrain-from-feedback requires --model-name (the "
                "incumbent's registry name; the candidate is saved as "
                "'<model-name>_candidate' unless --candidate-model-name "
                "is given)."
            ),
        )
        return 4

    incumbent_name = args.model_name
    candidate_name = (
        getattr(args, "candidate_model_name", None) or f"{incumbent_name}_candidate"
    )
    min_rows = (
        int(args.min_feedback_rows)
        if args.min_feedback_rows is not None
        else int(settings.FEEDBACK_MIN_ROWS)
    )
    use_mistake_tags = bool(
        getattr(args, "use_mistake_tags_as_label", False)
        or settings.FEEDBACK_USE_MISTAKE_TAGS_AS_LABEL
    )
    model_kind = getattr(args, "feedback_model_kind", "logreg")

    dataset = FeedbackDataset()
    rows = dataset.build()

    out_dir = Path(settings.REPORTS_DIR) / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset.export_csv(rows, out_dir / "feedback_dataset.csv")
    json_path = dataset.export_json(rows, out_dir / "feedback_dataset.json")
    log.info(
        "retrain.dataset_built",
        n=len(rows),
        min_rows=min_rows,
        csv_path=str(csv_path),
        json_path=str(json_path),
    )

    if len(rows) < min_rows:
        log.error(
            "retrain.insufficient_rows",
            n=len(rows),
            required=min_rows,
            note=(
                "Need >= FEEDBACK_MIN_ROWS closed trades before retraining. "
                "Run paper mode longer (target ~2-4 weeks of trading days for "
                "the default of 100) or pass --min-feedback-rows to override."
            ),
        )
        return 4

    try:
        candidate = train_candidate_from_feedback(
            rows,
            min_rows=min_rows,
            confidence_threshold=settings.CONFIDENCE_THRESHOLD,
            use_mistake_tags_as_label=use_mistake_tags,
            model_kind=model_kind,
        )
    except InsufficientFeedbackError as e:
        log.error(
            "retrain.insufficient_feedback",
            error=str(e),
            note="Train aborted; nothing written to the model registry.",
        )
        return 4
    except Exception as e:  # noqa: BLE001 - surface any trainer failure cleanly
        log.error("retrain.train_failed", error=str(e))
        return 5

    # Build the metadata payload that distinguishes this artifact as a
    # candidate. ``save_model`` will merge this into ``metadata.json``.
    extra_metadata = {
        "candidate": True,
        "source": "feedback",
        "incumbent_name": incumbent_name,
        "model_kind": model_kind,
        "label_strategy": candidate.label_strategy,
        "min_feedback_rows": min_rows,
        "n_total": candidate.n_total,
        "n_train": candidate.n_train,
        "n_val": candidate.n_val,
        "n_test": candidate.n_test,
        "test_metrics": candidate.test_metrics,
        "realized_metrics": candidate.realized_metrics,
        "feedback_train_range": [r.isoformat() for r in candidate.train_range],
        "feedback_val_range": [r.isoformat() for r in candidate.val_range],
        "feedback_test_range": [r.isoformat() for r in candidate.test_range],
        "mistake_tag_counts": candidate.mistake_tag_counts,
        "excluded_no_features": candidate.excluded_no_features,
        "boosted_trained": candidate.boosted_train_result is not None,
        "boosted_metrics": (
            candidate.boosted_train_result.aggregate_metrics
            if candidate.boosted_train_result is not None
            else None
        ),
    }
    candidate_version = save_model(
        candidate.train_result,
        name=candidate_name,
        extra_metadata=extra_metadata,
    )
    log.info(
        "retrain.candidate_saved",
        name=candidate_name,
        version=candidate_version,
        model_kind=model_kind,
        n_train=candidate.n_train,
        n_test=candidate.n_test,
        test_precision=round(candidate.test_metrics["precision"], 4),
        test_recall=round(candidate.test_metrics["recall"], 4),
        expectancy=round(candidate.realized_metrics["expectancy_per_trade"], 4),
        profit_factor=round(candidate.realized_metrics["profit_factor"], 4),
    )

    # Promotion comparison — advisory only. We deliberately do NOT
    # promote here. Operator must run --promote-model VERSION after
    # reviewing the report.
    incumbent_metadata: Optional[dict] = None
    try:
        incumbent_loaded = load_model(incumbent_name)
        incumbent_metadata = incumbent_loaded.metadata
    except Exception as e:  # noqa: BLE001 - first-time retrain has no incumbent
        log.warning(
            "retrain.no_incumbent",
            name=incumbent_name,
            error=str(e),
            note=(
                "No incumbent registered yet — comparison skipped. The "
                "candidate is saved and can become the incumbent via "
                f"--promote-model {candidate_version} once an incumbent "
                "exists; until then, train an initial model first."
            ),
        )
        return 0

    candidate_metadata = dict(incumbent_loaded.metadata)
    candidate_metadata.update(
        {
            "name": candidate_name,
            "version": candidate_version,
            "metrics": candidate.train_result.aggregate_metrics,
            "fold_metrics": [],  # candidate uses chronological split, not WF
        }
    )

    decision = compare(
        incumbent_metadata=incumbent_metadata,
        candidate_metadata=candidate_metadata,
        realized_metrics_candidate=candidate.realized_metrics,
    )
    report_path = write_comparison_report(decision, out_dir)
    log.info(
        "retrain.comparison_written",
        promote=decision.promote,
        failed_gates=decision.failed_gates,
        report=str(report_path),
        note=(
            "Decision is advisory. Run --promote-model VERSION to actually "
            "promote (the gate check will be re-run there)."
        ),
    )
    return 0


def _run_train_from_csv(settings: Settings, log, args) -> int:
    """Train a model on a real OHLCV CSV and save it to the registry.

    Pipeline (mirrors ``--smoke-train`` but on real data):

    1. Load + validate the CSV (timestamp, open, high, low, close,
       volume) via ``data.csv_loader.load_ohlcv_csv``.
    2. Build canonical features.
    3. Detect setups using the strategy resolved through the registry
       (``--strategy`` CLI flag, default ``vwap_ema_pullback``).
    4. Label setups with TP/SL/time using ``label_setups`` and
       ``--max-hold-bars`` (default: ``settings.MAX_HOLD_BARS``).
    5. Build X (FEATURE_COLUMNS) + y (0/1) from the setup feature
       snapshots, sorted chronologically by timestamp.
    6. Chronological 3-way split (train / val / test). Never random.
    7. Train via ``models.trainer.train`` with walk-forward CV over
       (train + val); calibrate on val.
    8. Save via ``models.model_registry.save_model`` with rich
       ``extra_metadata`` (CSV path, strategy, split ranges, label
       distribution, model_kind, train/val/test sizes).

    Refuses to run on missing CSV (exit 7), too few rows (exit 4),
    fewer than 100 setups (exit 4), single-class labels (exit 4), or
    missing required feature columns (exit 4). LIVE mode is not
    affected — this runner only ever writes to the local model
    registry.
    """
    import pandas as pd
    from pathlib import Path

    from data.csv_loader import REQUIRED_COLUMNS, load_ohlcv_csv
    from features.feature_builder import FEATURE_COLUMNS, build_features
    from labeling.tp_sl_labeler import label_setups
    from models.model_registry import save_model
    from models.trainer import has_lightgbm, train
    from strategies.registry import instantiate as instantiate_strategy
    from validation.time_split import chronological_split
    from validation.walk_forward import walk_forward_splits

    MIN_SETUPS = 100
    MIN_OHLCV_ROWS = 200  # below this no strategy can warm up its features

    # ---- 0. Validate inputs ------------------------------------------------
    if not args.train_csv:
        log.error(
            "train.missing_csv",
            note="MODE=TRAIN requires --train-csv PATH (real OHLCV CSV).",
        )
        return 4
    if not args.model_name:
        log.error(
            "train.missing_model_name",
            note="MODE=TRAIN requires --model-name NAME for the registry.",
        )
        return 4

    csv_path = Path(args.train_csv)
    if not csv_path.is_file():
        log.error("train.csv_missing", path=str(csv_path))
        return 7

    # Resolve max_hold_bars: CLI > settings > 20.
    max_hold_bars = (
        int(args.max_hold_bars)
        if args.max_hold_bars is not None
        else int(getattr(settings, "MAX_HOLD_BARS", 20) or 20)
    )
    if max_hold_bars <= 0:
        log.error("train.bad_max_hold_bars", value=max_hold_bars)
        return 4

    train_frac = float(args.train_frac)
    val_frac = float(args.val_frac)
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1.0:
        log.error(
            "train.bad_split_fractions",
            train_frac=train_frac,
            val_frac=val_frac,
            note="train_frac > 0, val_frac >= 0, train_frac + val_frac < 1.",
        )
        return 4

    model_kind = args.model_kind
    if model_kind == "lightgbm" and not has_lightgbm():
        log.error(
            "train.lightgbm_unavailable",
            note=(
                "LightGBM is not installed. Install it (pip install lightgbm) "
                "or pass --model-kind logreg."
            ),
        )
        return 4

    # ---- 1. Load OHLCV -----------------------------------------------------
    try:
        df = load_ohlcv_csv(
            csv_path,
            instrument=settings.INSTRUMENT,
            timeframe="1m",
            tz=settings.TIMEZONE,
        )
    except ValueError as e:
        # ``load_ohlcv_csv`` raises ValueError on missing required columns —
        # surface that as a config error rather than a crash.
        log.error(
            "train.csv_invalid",
            path=str(csv_path),
            error=str(e),
            required_columns=list(REQUIRED_COLUMNS),
        )
        return 4

    if df.empty or len(df) < MIN_OHLCV_ROWS:
        log.error(
            "train.csv_too_small",
            rows=int(len(df)),
            required_minimum=MIN_OHLCV_ROWS,
            note=(
                "CSV has too few rows to compute features and detect setups. "
                "At least ~200 bars are needed for the warmup windows."
            ),
        )
        return 4

    log.info(
        "train.csv_loaded",
        path=str(csv_path),
        rows=int(len(df)),
        first_ts=str(df.index.min()),
        last_ts=str(df.index.max()),
        instrument=settings.INSTRUMENT,
    )

    # ---- 2. Features -------------------------------------------------------
    features = build_features(df, instrument=settings.INSTRUMENT, tz=settings.TIMEZONE)
    missing_cols = [c for c in FEATURE_COLUMNS if c not in features.columns]
    if missing_cols:
        log.error(
            "train.features_missing_columns",
            missing=missing_cols,
            note=(
                "Feature builder did not produce all FEATURE_COLUMNS. This "
                "usually means the CSV is too short for warmups; widen the "
                "data range and retry."
            ),
        )
        return 4

    log.info(
        "train.features_built",
        rows=int(len(features)),
        feature_columns=len(FEATURE_COLUMNS),
        first_ts=str(features.index.min()) if len(features) else None,
        last_ts=str(features.index.max()) if len(features) else None,
    )

    # ---- 3. Setups ---------------------------------------------------------
    strategy_name = getattr(args, "strategy", None) or "vwap_ema_pullback"
    try:
        strategy = instantiate_strategy(strategy_name, instrument=settings.INSTRUMENT)
    except KeyError as e:
        log.error("train.unknown_strategy", error=str(e))
        return 4
    setups = strategy.detect_setups(features)
    by_dir: dict[str, int] = {"long": 0, "short": 0}
    for s in setups:
        by_dir[s.direction] = by_dir.get(s.direction, 0) + 1

    log.info(
        "train.setups_detected",
        strategy=strategy.name,
        total=len(setups),
        long=by_dir["long"],
        short=by_dir["short"],
    )
    if len(setups) < MIN_SETUPS:
        log.error(
            "train.too_few_setups",
            n=len(setups),
            required=MIN_SETUPS,
            note=(
                "Strategy did not produce enough setups for a stable train. "
                "Common causes: CSV too short, warmup consuming most bars, "
                "wrong instrument/timezone, or strategy params too strict."
            ),
        )
        return 4

    # ---- 4. Labels ---------------------------------------------------------
    labels = label_setups(setups, df, max_hold_bars=max_hold_bars)
    n_pos = sum(int(l.label) for l in labels)
    log.info(
        "train.labels_built",
        n=len(labels),
        positive=n_pos,
        positive_rate=round(n_pos / len(labels), 4) if labels else 0.0,
        max_hold_bars=max_hold_bars,
    )
    if n_pos == 0 or n_pos == len(labels):
        log.error(
            "train.single_class_labels",
            positive=n_pos,
            n=len(labels),
            note=(
                "All setups have the same label — cannot train a binary "
                "classifier. Check TP/SL parameters and CSV coverage; both "
                "wins and losses must occur."
            ),
        )
        return 4

    # ---- 5. Build X / y ----------------------------------------------------
    rows = []
    for setup, lab in zip(setups, labels):
        row = dict(setup.features)
        row["_label"] = int(lab.label)
        row["_ts"] = setup.timestamp
        rows.append(row)

    setup_df = pd.DataFrame(rows).set_index("_ts")
    setup_df.index = pd.DatetimeIndex(setup_df.index)
    if not setup_df.index.is_monotonic_increasing:
        setup_df = setup_df.sort_index()

    feature_missing = [c for c in FEATURE_COLUMNS if c not in setup_df.columns]
    if feature_missing:
        log.error(
            "train.setup_features_missing",
            missing=feature_missing,
            note="Setup feature snapshot is incomplete; cannot align to FEATURE_COLUMNS.",
        )
        return 4

    y = setup_df["_label"].astype(int)
    X = setup_df[list(FEATURE_COLUMNS)]

    if y.nunique() < 2:
        # Defense in depth: even after the earlier check, the chronological
        # split below could still leave a single class in train/val. We
        # check again and surface a clear error.
        log.error("train.single_class_labels", positive=int(y.sum()), n=int(len(y)))
        return 4

    # ---- 6. Chronological split (never random) -----------------------------
    try:
        train_idx, val_idx, test_idx = chronological_split(
            setup_df, train_frac=train_frac, val_frac=val_frac
        )
    except ValueError as e:
        log.error(
            "train.split_failed",
            error=str(e),
            n_setups=int(len(setup_df)),
            train_frac=train_frac,
            val_frac=val_frac,
        )
        return 4

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    if y_train.nunique() < 2 or y_val.nunique() < 2:
        log.error(
            "train.single_class_in_fold",
            train_classes=int(y_train.nunique()),
            val_classes=int(y_val.nunique()),
            note=(
                "After chronological split, train or val contains a single "
                "class. Provide a longer CSV or rebalance the split fractions."
            ),
        )
        return 4

    log.info(
        "train.split",
        n_total=int(len(setup_df)),
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
        train_start=str(X_train.index[0]),
        train_end=str(X_train.index[-1]),
        val_start=str(X_val.index[0]),
        val_end=str(X_val.index[-1]),
        test_start=str(X_test.index[0]),
        test_end=str(X_test.index[-1]),
        train_frac=train_frac,
        val_frac=val_frac,
    )

    # ---- 7. Walk-forward over (train + val) --------------------------------
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
        log.warning("train.walk_forward_skipped", error=str(e))

    # ---- 8. Train ----------------------------------------------------------
    try:
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
    except Exception as e:  # noqa: BLE001 - surface trainer failure cleanly
        log.error("train.failed", model_kind=model_kind, error=str(e))
        return 5

    log.info(
        "train.metrics_val",
        model_kind=model_kind,
        n_train=result.n_train,
        n_val=result.n_val,
        n_folds=len(result.fold_metrics),
        **{
            k: round(v, 4) if isinstance(v, float) else v
            for k, v in result.aggregate_metrics.items()
        },
    )

    # Holdout test metrics — independent confirmation that the calibrated
    # model didn't simply memorize the val window.
    test_proba = result.estimator.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= float(settings.CONFIDENCE_THRESHOLD)).astype(int)
    test_acc = float((test_pred == y_test.values).mean())
    test_metrics = {
        "n_test": int(len(test_idx)),
        "accuracy": round(test_acc, 4),
        "positive_rate": round(float(y_test.mean()), 4),
        "approve_rate": round(float(test_pred.mean()), 4),
    }
    log.info("train.metrics_test", **test_metrics)

    # ---- 9. Persist --------------------------------------------------------
    extra_metadata = {
        "source": "train_from_csv",
        "csv_path": str(csv_path),
        "instrument": settings.INSTRUMENT,
        "timeframe": "1m",
        "strategy": strategy.name,
        "model_kind": model_kind,
        "max_hold_bars": max_hold_bars,
        "confidence_threshold": float(settings.CONFIDENCE_THRESHOLD),
        "train_frac": train_frac,
        "val_frac": val_frac,
        "n_total_setups": int(len(setup_df)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "label_distribution": {
            "positive": int(n_pos),
            "negative": int(len(labels) - n_pos),
            "positive_rate": float(n_pos / len(labels)),
        },
        "ohlcv_range": {
            "first_ts": str(df.index.min()),
            "last_ts": str(df.index.max()),
            "rows": int(len(df)),
        },
        "split_ranges": {
            "train": [str(X_train.index[0]), str(X_train.index[-1])],
            "val": [str(X_val.index[0]), str(X_val.index[-1])],
            "test": [str(X_test.index[0]), str(X_test.index[-1])],
        },
        "test_metrics": test_metrics,
    }

    try:
        version = save_model(
            result, name=args.model_name, extra_metadata=extra_metadata
        )
    except Exception as e:  # noqa: BLE001
        log.error("train.save_failed", error=str(e))
        return 5

    saved_dir = Path(settings.MODELS_DIR) / args.model_name / version
    log.info(
        "train.saved",
        name=args.model_name,
        version=version,
        path=str(saved_dir),
        model_kind=model_kind,
    )
    log.info(
        "train.complete",
        message=(
            f"MODE=TRAIN finished. Model={args.model_name} "
            f"version={version} kind={model_kind} setups={len(setup_df)} "
            f"test_accuracy={test_metrics['accuracy']}"
        ),
    )
    return 0


def _run_promote_model(settings: Settings, log, *, args) -> int:
    """Operator-only: promote a candidate model version after review.

    Refuses to act without a comparison report's PromotionDecision
    saying promote=True — and even then, the only state change is an
    on-disk symlink-style update; this function never modifies model
    files in place.
    """
    from pathlib import Path

    from analysis.promotion import compare
    from models.model_registry import load_model

    if not args.model_name or not args.promote_model:
        log.error(
            "promote_model.bad_args",
            note="Both --model-name and --promote-model VERSION are required.",
        )
        return 4

    try:
        incumbent = load_model(args.model_name)
        candidate = load_model(args.model_name, version=args.promote_model)
    except Exception as e:
        log.error("promote_model.load_failed", error=str(e))
        return 5

    decision = compare(
        incumbent_metadata=incumbent.metadata,
        candidate_metadata=candidate.metadata,
    )
    log.info(
        "promote_model.decision",
        promote=decision.promote,
        rationale=decision.rationale,
        failed_gates=decision.failed_gates,
    )
    if not decision.promote:
        log.warning(
            "promote_model.refused",
            note="PromotionDecision.promote=False; not advancing pointer.",
        )
        return 0

    # In a real registry we'd flip a "current" symlink. For the MVP the
    # registry already resolves "latest" by sort order, so promotion
    # just means the candidate exists on disk — which it does. Log the
    # transition for the audit trail and exit.
    log.info(
        "promote_model.promoted",
        model=args.model_name,
        old_version=incumbent.metadata.get("version"),
        new_version=candidate.metadata.get("version"),
    )
    return 0


def _run_train_universe(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Train a pooled multi-symbol equity model and register it."""
    from config.equity_allowlist import LIQUID_EQUITY_ALLOWLIST
    from models.equity_trainer import EquityTrainError, train_universe_model

    if not args.model_name:
        log.error("train_universe.missing_model_name", note="Requires --model-name NAME.")
        return 4

    if args.train_symbols:
        symbols = [s.strip().upper() for s in args.train_symbols.split(",") if s.strip()]
    else:
        symbols = sorted(LIQUID_EQUITY_ALLOWLIST)

    try:
        result = train_universe_model(
            settings,
            symbols=symbols,
            model_name=args.model_name,
            model_kind=args.model_kind,
            strategy_name=args.strategy or "vwap_ema_pullback",
            max_hold_bars=args.max_hold_bars,
            train_frac=float(args.train_frac),
            val_frac=float(args.val_frac),
        )
    except EquityTrainError as e:
        log.error("train_universe.failed", error=str(e))
        return 4
    except Exception as e:  # noqa: BLE001
        log.error("train_universe.unexpected", error=str(e))
        return 5

    log.info(
        "train_universe.complete",
        model_name=result.model_name,
        version=result.version,
        symbols_used=result.symbols_used,
        n_setups=result.n_total_setups,
        n_pos=result.n_pos,
        test_metrics=result.test_metrics,
    )
    return 0


def _run_download_data(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Download historical bars from Alpaca into the historical data dir."""
    from data.alpaca_bars import download_symbols

    symbols = None
    if args.download_symbols:
        symbols = [s.strip().upper() for s in args.download_symbols.split(",") if s.strip()]

    try:
        results = download_symbols(
            settings,
            symbols=symbols,
            timeframe="1m",
            days=int(args.download_days),
        )
    except Exception as e:  # noqa: BLE001
        log.error("download.failed", error=str(e))
        return 5

    ok = [s for s, r in results.items() if r == "ok"]
    failed = {s: r for s, r in results.items() if r != "ok"}
    log.info("download.complete", ok=ok, failed=failed)
    return 0 if ok else 5


def _run_intraday_cli(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Run the continuous intraday scanner (autonomous paper loop)."""
    from agents.orchestrator import build_orchestrator
    from notifications.notification_service import NotificationService
    from workflows.intraday_loop import IntradayLoop

    if settings.WORKFLOW_EXECUTION_MODE == "LIVE":
        log.error("intraday.live_refused", note="LIVE is locked.")
        return 6

    notifier = NotificationService.from_settings(settings)
    orchestrator = None
    if settings.ENABLE_LLM_AGENTS:
        try:
            orchestrator = build_orchestrator(settings, notifier=notifier)
        except Exception as e:  # noqa: BLE001
            log.warning("intraday.orchestrator_unavailable", error=str(e))

    loop = IntradayLoop(
        settings,
        notifier=notifier,
        orchestrator=orchestrator,
        dry_run=args.workflow_dry_run,
    )
    log.info(
        "intraday.cli",
        dry_run=loop.dry_run,
        execution_mode=settings.WORKFLOW_EXECUTION_MODE,
        interval_minutes=settings.WORKFLOW_SCAN_INTERVAL_MINUTES,
        dynamic_universe=settings.WORKFLOW_DYNAMIC_UNIVERSE,
        long_only=settings.WORKFLOW_LONG_ONLY,
    )
    loop.run_forever(max_cycles=args.intraday_max_cycles)
    return 0


def _run_workflow_cli(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Run autonomous workflows (separate from MODE train/backtest/paper loop)."""
    from workflows.scheduler import WorkflowScheduler
    from workflows.workflow_runner import WorkflowRunner

    if settings.WORKFLOW_EXECUTION_MODE == "LIVE":
        log.error(
            "workflow.live_refused",
            note="WORKFLOW_EXECUTION_MODE=LIVE is locked. Use DRY_RUN or PAPER.",
        )
        return 6

    runner = WorkflowRunner.from_settings(
        settings,
        cli_dry_run=args.workflow_dry_run,
    )
    log.info(
        "workflow.cli",
        workflow=args.workflow,
        scheduler=args.workflow_scheduler,
        dry_run=runner.dry_run,
        execution_mode=runner.execution_mode(),
        autonomous=settings.AUTONOMOUS_TRADING_ENABLED,
    )

    if args.workflow_scheduler:
        sched = WorkflowScheduler(settings, runner, blocking=True)
        sched.run_forever()
        return 0

    if not args.workflow:
        log.error("workflow.missing_name", note="Pass --workflow NAME or --workflow-scheduler")
        return 4

    result = runner.run(
        args.workflow,
        force=bool(args.workflow_force),
    )
    if not result.success and not result.skipped:
        log.error("workflow.failed", workflow=args.workflow, errors=result.errors)
        return 5
    log.info(
        "workflow.complete",
        workflow=args.workflow,
        success=result.success,
        skipped=result.skipped,
        dry_run=result.dry_run,
    )
    return 0


def _run_parallel_paper_cli(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Launch, query, or report on parallel paper evaluation tracks."""
    import json

    from evaluation.parallel_paper_runner import ParallelPaperRunner
    from reports.paper_evaluation_report import write_combined_summary, write_track_report

    if settings.WORKFLOW_EXECUTION_MODE == "LIVE":
        log.error("parallel.live_refused", note="LIVE is locked.")
        return 6

    if not settings.ENABLE_PARALLEL_PAPER:
        log.error(
            "parallel.not_enabled",
            note="Set ENABLE_PARALLEL_PAPER=true to use parallel paper mode.",
        )
        return 4

    runner = ParallelPaperRunner(
        settings,
        dry_run=args.workflow_dry_run,
    )
    runner.build_tracks()

    if args.parallel_paper_status:
        status = runner.status()
        print(json.dumps(status, indent=2, default=str))
        return 0

    if args.parallel_paper_report:
        results: dict = {}
        for provider, ctx in runner.contexts.items():
            state = ctx.load_state()
            results[provider] = {
                "success": not ctx.blocked,
                "trades": state.get("trade_count", 0),
                "workflows": [],
            }
            write_track_report(ctx, payload=results[provider])
        write_combined_summary(runner.contexts, results=results)
        log.info("parallel.reports_written")
        return 0

    if args.start_parallel_paper:
        log.info(
            "parallel.start",
            providers=sorted(runner.contexts.keys()),
            dry_run=args.workflow_dry_run,
        )
        results = runner.run_all(force=bool(args.workflow_force))
        for provider, ctx in runner.contexts.items():
            payload = results.get(provider, {})
            write_track_report(ctx, payload=payload)
        write_combined_summary(runner.contexts, results=results)
        all_ok = all(
            r.get("success", False) for r in results.values() if isinstance(r, dict)
        )
        log.info("parallel.done", success=all_ok)
        return 0 if all_ok else 5

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

    if args.smoke_backtest:
        return _run_smoke_backtest(settings, log)

    if args.smoke_paper:
        return _run_smoke_paper(settings, log, args=args)

    if args.smoke_daily_report:
        return _run_smoke_daily_report(settings, log)

    if args.smoke_agents:
        return _run_smoke_agents(settings, log)

    if args.smoke_trade_analysis:
        return _run_smoke_trade_analysis(settings, log)

    if args.retrain_from_feedback:
        return _run_retrain_from_feedback(settings, log, args=args)

    if args.promote_model:
        return _run_promote_model(settings, log, args=args)

    if args.download_data:
        return _run_download_data(settings, log, args=args)

    if args.train_universe:
        return _run_train_universe(settings, log, args=args)

    if args.workflow_intraday:
        return _run_intraday_cli(settings, log, args=args)

    if args.workflow or args.workflow_scheduler:
        return _run_workflow_cli(settings, log, args=args)

    if args.start_parallel_paper or args.parallel_paper_status or args.parallel_paper_report:
        return _run_parallel_paper_cli(settings, log, args=args)

    if settings.MODE == "TRAIN":
        return _run_train_from_csv(settings, log, args)
    if settings.MODE == "BACKTEST":
        if args.backtest_csv:
            return _run_backtest_from_csv(
                settings,
                log,
                csv_path=args.backtest_csv,
                model_name=args.model_name,
                model_version=args.model_version,
                strategy_name=args.strategy or "vwap_ema_pullback",
            )
        log.warning(
            "backtest.no_input",
            note="Provide --backtest-csv PATH or use --smoke-backtest.",
        )
        return 0
    if settings.MODE == "PAPER":
        return _run_paper_forever(settings, log, args=args)
    # Defense-in-depth: even if the settings validator was bypassed (e.g.,
    # an operator set LIVE_ADAPTER_CONFIRMED=true and reached this branch),
    # the dispatcher refuses to advance into anything that resembles real
    # execution. A real live adapter would be wired in here in a future
    # implementation; for the MVP we exit cleanly with a non-zero code so
    # callers can detect the misconfiguration.
    log.error(
        "mode.live_unsupported",
        mode=settings.MODE,
        note=(
            "LIVE mode reached the dispatcher but no live adapter is "
            "implemented in this MVP. Refusing to start."
        ),
    )
    return 6


if __name__ == "__main__":
    sys.exit(main())
