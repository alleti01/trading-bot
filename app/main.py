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
