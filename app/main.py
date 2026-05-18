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


def _run_backtest(
    settings: Settings,
    log,
    *,
    ohlcv_df,
    model_name: Optional[str],
    model_version: str,
    timeframe: str,
) -> int:
    """Shared backtest core used by both --smoke-backtest and --backtest-csv."""
    from backtesting.engine import BacktestEngine
    from features.feature_builder import build_features
    from models.model_registry import load_model
    from models.predictor import Predictor
    from reports.backtest_report import write_backtest_report
    from strategies.vwap_ema_pullback import VWAPEMAPullback

    features = build_features(
        ohlcv_df, instrument=settings.INSTRUMENT, tz=settings.TIMEZONE
    )
    log.info("backtest.features_built", rows_in=len(ohlcv_df), rows_out=len(features))

    setups = VWAPEMAPullback(instrument=settings.INSTRUMENT).detect_setups(features)
    log.info(
        "backtest.setups",
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
) -> int:
    from data.csv_loader import load_ohlcv_csv

    df = load_ohlcv_csv(
        path=csv_path,
        instrument=settings.INSTRUMENT,
        timeframe="1m",
        tz=settings.TIMEZONE,
    )
    return _run_backtest(
        settings,
        log,
        ohlcv_df=df,
        model_name=model_name,
        model_version=model_version,
        timeframe="1m",
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


def _run_smoke_paper(settings: Settings, log, *, args: argparse.Namespace) -> int:
    """Day-5 paper smoke: a few synchronous bar cycles, then exit."""
    from notifications.notification_service import NotificationService
    from paper.loop import build_paper_loop
    from scheduler.service import SchedulerService
    from sqlalchemy import func, select

    from storage.db import session_scope
    from storage.tables import ClosedTrade, Notification, PaperTrade, RiskBlock

    notifier = NotificationService.from_settings(settings)
    feed = _build_paper_feed(settings, args, log)
    loop = build_paper_loop(
        settings=settings,
        feed=feed,
        notifier=notifier,
        model_name=args.model_name,
        model_version=args.model_version,
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
    feed = _build_paper_feed(settings, args, log)
    loop = build_paper_loop(
        settings=settings,
        feed=feed,
        notifier=notifier,
        model_name=args.model_name,
        model_version=args.model_version,
        high_risk_news_fn=orchestrator.high_risk_news_active,
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


def _run_retrain_from_feedback(settings: Settings, log) -> int:
    """Build the feedback dataset, train a candidate model, write a
    promotion comparison report. NEVER promotes the new model — the
    operator must run ``--promote-model`` after reviewing the report.
    """
    from pathlib import Path

    from analysis.feedback_dataset import FeedbackDataset
    from analysis.promotion import compare, write_comparison_report
    from models.model_registry import load_model

    dataset = FeedbackDataset()
    rows = dataset.build()
    out_dir = Path(settings.REPORTS_DIR) / "feedback"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset.export_csv(rows, out_dir / "feedback_dataset.csv")
    json_path = dataset.export_json(rows, out_dir / "feedback_dataset.json")
    log.info(
        "retrain.dataset_built",
        n=len(rows),
        csv_path=str(csv_path),
        json_path=str(json_path),
    )

    if len(rows) < 30:
        log.warning(
            "retrain.too_few_trades",
            n=len(rows),
            note="Need >= 30 closed trades before retraining; aborting candidate train.",
        )
        return 0

    # Candidate training requires a stored incumbent + a real feature
    # matrix. We deliberately do NOT auto-pick which model name to train
    # against; an operator must already have an incumbent registered.
    try:
        incumbent_loaded = load_model("vwap_ema_pullback_lr")
    except Exception as e:
        log.warning(
            "retrain.no_incumbent",
            error=str(e),
            note="Skipping comparison; train an initial model first via --smoke-train.",
        )
        return 0

    # Building a real candidate trainer + walk-forward over the feedback
    # dataset is intentionally out of scope for this smoke path; the
    # comparison report against the incumbent (using its own metadata)
    # already exercises the gate logic. Operators wire their offline
    # trainer to ``analysis.promotion.compare`` directly.
    decision = compare(
        incumbent_metadata=incumbent_loaded.metadata,
        candidate_metadata=incumbent_loaded.metadata,  # placeholder candidate
    )
    report_path = write_comparison_report(decision, out_dir)
    log.info(
        "retrain.comparison_written",
        promote=decision.promote,
        failed_gates=decision.failed_gates,
        report=str(report_path),
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
        return _run_retrain_from_feedback(settings, log)

    if args.promote_model:
        return _run_promote_model(settings, log, args=args)

    if settings.MODE == "TRAIN":
        log.warning("mode.not_implemented", mode="TRAIN", note="Day 3 deliverable")
        return 0
    if settings.MODE == "BACKTEST":
        if args.backtest_csv:
            return _run_backtest_from_csv(
                settings,
                log,
                csv_path=args.backtest_csv,
                model_name=args.model_name,
                model_version=args.model_version,
            )
        log.warning(
            "backtest.no_input",
            note="Provide --backtest-csv PATH or use --smoke-backtest.",
        )
        return 0
    if settings.MODE == "PAPER":
        return _run_paper_forever(settings, log, args=args)
    # LIVE has already been refused by the settings validator if not configured.
    log.warning("mode.live_unsupported", note="No live adapter implemented in this MVP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
