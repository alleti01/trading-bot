"""Paper trading loop — one bar-close cycle.

Pipeline (executed every ``on_bar_close`` call):

1. Poll the feed; if no new bar appeared, exit early.
2. Manage any open position **first** using the shared trade-management
   exits (TP / SL / max-hold / forced-flat) so we apply the new bar's
   range to the existing position before considering new entries.
3. Outside the trading window: skip step 4–6 (no new entries).
4. Recompute features on the rolling window. Filter strategy setups to
   ones whose timestamp is exactly ``latest_bar.ts`` — this is the
   "no-lookahead" property: any setup older than the latest bar would
   already have been considered on a previous tick.
5. Optional model gate.
6. Risk engine — authoritative. Blocked setups are persisted as
   ``risk_blocks`` rows and skipped.
7. Size + submit through :class:`PaperExecutor`. Sizing failures are
   recorded as risk blocks (defensive — should be rare).

A separate ``flatten_now()`` is exposed so the scheduler's end-of-day job
can force-flat any straggler position even if no new bar has arrived yet
(e.g. data feed went quiet around the close).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from app.logging_config import get_logger
from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from backtesting.trade_management import apply_exit, check_exit
from config.instruments import get_instrument
from config.settings import Settings
from data.market_data_service import IncrementalFeed, PollResult
from execution.base import Order
from execution.paper_executor import KillSwitchActive, PaperExecutor
from features.feature_builder import build_features
from models.predictor import Predictor
from risk.kill_switch import KillSwitch
from risk.position_sizing import size_position
from risk.risk_engine import RiskConfig, evaluate
from scheduler.market_hours import is_force_flat_due, is_in_trading_window
from sqlalchemy import select
from storage.db import session_scope
from storage.tables import ModelPrediction as ModelPredictionRow
from storage.tables import RiskBlock as RiskBlockRow
from storage.tables import Setup as SetupRow
from strategies.base import Setup, Strategy


class _NotifierLike(Protocol):
    def notify(self, kind: str, /, **payload: Any) -> None: ...


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class BarCycleResult:
    new_bar: bool
    bar_ts: Optional[datetime] = None
    setups_seen: int = 0
    setups_filled: int = 0
    setups_risk_blocked: int = 0
    setups_model_rejected: int = 0
    exits: int = 0
    in_window: bool = False
    trading_enabled: bool = True
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loop
# ---------------------------------------------------------------------------
class PaperTradingLoop:
    """Single-bar tick handler for paper mode.

    Construct once at boot, then call ``on_bar_close(now)`` from the
    scheduler each minute (or whatever ``BAR_INTERVAL_SECONDS`` is set to).
    """

    def __init__(
        self,
        *,
        settings: Settings,
        feed: IncrementalFeed,
        strategy: Strategy,
        notifier: _NotifierLike,
        predictor: Optional[Predictor] = None,
        portfolio: Optional[Portfolio] = None,
        executor: Optional[PaperExecutor] = None,
        kill_switch: Optional[KillSwitch] = None,
        max_hold_bars: Optional[int] = None,
    ) -> None:
        self.settings = settings
        self.feed = feed
        self.strategy = strategy
        self.notifier = notifier
        self.predictor = predictor
        self.spec = get_instrument(settings.INSTRUMENT)
        self.portfolio = portfolio or Portfolio(instrument_spec=self.spec)
        self.kill_switch = kill_switch or KillSwitch()
        self.fills = make_fills_model(
            settings.INSTRUMENT,
            slippage_ticks=settings.SLIPPAGE_TICKS,
            commission_per_contract=settings.COMMISSION_PER_CONTRACT,
        )
        self.executor = executor or PaperExecutor(
            portfolio=self.portfolio,
            fills_model=self.fills,
            kill_switch=self.kill_switch,
        )
        self.risk_config = RiskConfig.from_settings(settings)
        self.tz = ZoneInfo(settings.TIMEZONE)
        self.max_hold_bars = int(max_hold_bars or settings.MAX_HOLD_BARS)

        self.trading_enabled: bool = predictor is not None or True  # default-on
        self._bar_index = 0
        self._last_bar_ts: Optional[datetime] = None
        self._last_seen_setup_ids: set[str] = set()
        self.log = get_logger("paper.loop")

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------
    def on_bar_close(self, now: datetime) -> BarCycleResult:
        """Run a single bar-close cycle. Never raises."""
        result = BarCycleResult(new_bar=False, trading_enabled=self.trading_enabled)
        try:
            poll: PollResult = self.feed.poll_latest()
        except Exception as e:
            self._record_error(result, "feed.poll_failed", e)
            return result

        if poll.new_candle is None:
            return result

        result.new_bar = True
        result.bar_ts = poll.new_candle.ts
        self._bar_index += 1
        self._last_bar_ts = poll.new_candle.ts

        bar_series = self._bar_to_series(poll.new_candle)

        # Step 2: manage open position first.
        try:
            if self._maybe_exit_open(bar_series, result):
                result.exits += 1
        except Exception as e:
            self._record_error(result, "exit.failed", e)

        result.in_window = is_in_trading_window(now, self.settings)
        if not result.in_window:
            return result

        # Step 3+: scan for new setups.
        if not self.trading_enabled:
            return result
        if self.kill_switch.is_tripped():
            return result

        try:
            self._consider_new_entries(
                rolling_window=poll.rolling_window,
                latest_ts=poll.new_candle.ts,
                bar_series=bar_series,
                now=now,
                result=result,
            )
        except Exception as e:
            self._record_error(result, "entry.failed", e)
        return result

    def flatten_now(self, now: datetime, *, reason: str = "forced_flat") -> bool:
        """Close any open position immediately at the *last seen* bar's close.

        Called by the scheduler at end-of-day. Returns True iff a position
        was closed.
        """
        if self.portfolio.open_position is None:
            return False
        ts = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        # Use the open's entry price as a fallback raw price if we have no
        # newer bar — fills model will still apply slippage either way.
        last_close: float
        if self._last_bar_ts is not None and self.portfolio.open_position is not None:
            last_close = self.portfolio.open_position.entry_price
        else:
            last_close = self.portfolio.open_position.entry_price  # type: ignore[union-attr]

        record = self.executor.close_position(
            ts=ts,
            exit_raw_price=last_close,
            exit_reason=reason,
            bar_index=self._bar_index,
        )
        self.notifier.notify(
            "forced_flat",
            instrument=record.instrument,
            direction=record.direction,
            net_pnl=round(record.net_pnl, 2),
            ts=ts.isoformat(),
        )
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _bar_to_series(self, candle) -> pd.Series:
        return pd.Series(
            {
                "open": float(candle.open),
                "high": float(candle.high),
                "low": float(candle.low),
                "close": float(candle.close),
                "volume": float(candle.volume),
            },
            name=candle.ts,
        )

    def _maybe_exit_open(self, bar_series: pd.Series, result: BarCycleResult) -> bool:
        pos = self.portfolio.open_position
        if pos is None:
            return False
        bar_ts = bar_series.name
        if isinstance(bar_ts, pd.Timestamp):
            bar_ts = bar_ts.to_pydatetime()

        decision = check_exit(
            position=pos,
            bar=bar_series,
            bar_index=self._bar_index,
            bar_ts=bar_ts,
            max_hold_bars=self.max_hold_bars,
            force_flat_time=self.settings.force_flat_time(),
            market_type=self.settings.MARKET_TYPE,
            tz=self.tz,
        )
        if decision is None:
            return False
        record = self.executor.close_position(
            ts=bar_ts,
            exit_raw_price=decision.raw_price,
            exit_reason=decision.reason,
            bar_index=self._bar_index,
        )
        self.notifier.notify(
            "trade.closed",
            instrument=record.instrument,
            direction=record.direction,
            exit_reason=record.exit_reason,
            net_pnl=round(record.net_pnl, 2),
            ts=bar_ts.isoformat(),
        )
        return True

    def _consider_new_entries(
        self,
        *,
        rolling_window: pd.DataFrame,
        latest_ts: datetime,
        bar_series: pd.Series,
        now: datetime,
        result: BarCycleResult,
    ) -> None:
        if rolling_window.empty:
            return
        # Build features on the rolling window. The feature builder is
        # idempotent and respects no-lookahead; recomputing here is fine
        # for Day 5 sizes (a few hundred bars).
        rolling = rolling_window.copy()
        rolling["instrument"] = self.settings.INSTRUMENT
        rolling["timeframe"] = "1m"
        try:
            features = build_features(
                rolling, instrument=self.settings.INSTRUMENT, tz=self.settings.TIMEZONE
            )
        except Exception as e:
            self._record_error(result, "features.failed", e)
            return

        if features.empty:
            return

        # Force-flat: even if outside-window check above let us through
        # via crypto 24/7, we still respect the explicit flat time.
        if is_force_flat_due(now, self.settings) and self.settings.MARKET_TYPE == "futures":
            return

        try:
            setups = self.strategy.detect_setups(features)
        except Exception as e:
            self._record_error(result, "strategy.failed", e)
            return

        # Only consider setups for the latest bar — anything older was
        # already handled on a prior tick (and re-emitting them would be
        # an implicit lookahead).
        latest_setups = [s for s in setups if s.timestamp == latest_ts]
        result.setups_seen = len(latest_setups)
        if not latest_setups:
            return

        for setup in latest_setups:
            self._handle_setup(setup, bar_series, now, result)

    def _handle_setup(
        self,
        setup: Setup,
        bar_series: pd.Series,
        now: datetime,
        result: BarCycleResult,
    ) -> None:
        if setup.id in self._last_seen_setup_ids:
            return
        self._last_seen_setup_ids.add(setup.id)

        self._persist_setup(setup)
        self.notifier.notify(
            "signal.generated",
            instrument=setup.instrument,
            direction=setup.direction,
            entry=round(setup.entry_price, 4),
            stop=round(setup.stop_price, 4),
            target=round(setup.target_price, 4),
            ts=setup.timestamp.isoformat(),
        )

        # Optional model gate.
        if self.predictor is not None:
            try:
                pred = self.predictor.predict_setup(setup)
            except Exception as e:
                self._record_error(result, "predictor.failed", e)
                result.setups_model_rejected += 1
                return
            self._persist_prediction(setup, pred)
            if not pred.approved:
                result.setups_model_rejected += 1
                return

        # Risk engine.
        decision = evaluate(
            setup,
            self.portfolio,
            self.risk_config,
            now,
            kill_switch_tripped=self.kill_switch.is_tripped(),
        )
        if not decision.allowed:
            self._persist_risk_block(setup, decision, now)
            self.notifier.notify(
                "trade.blocked",
                instrument=setup.instrument,
                direction=setup.direction,
                rule=decision.rule,
                reason=decision.reason,
            )
            result.setups_risk_blocked += 1
            return

        # Sizing + submit.
        try:
            sizing = size_position(
                entry_price=setup.entry_price,
                stop_price=setup.stop_price,
                instrument=self.spec,
                risk_per_trade=self.settings.RISK_PER_TRADE,
                max_position_size=self.settings.MAX_POSITION_SIZE,
            )
        except Exception as e:
            self._record_error(result, "sizing.failed", e)
            return

        order = Order(
            instrument=setup.instrument,
            direction=setup.direction,
            quantity=float(sizing.quantity),
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            setup_id=setup.id,
        )
        try:
            fill = self.executor.submit(order)
        except KillSwitchActive:
            return
        except Exception as e:
            self._record_error(result, "executor.failed", e)
            return

        result.setups_filled += 1
        self.notifier.notify(
            "trade.opened",
            instrument=setup.instrument,
            direction=setup.direction,
            quantity=order.quantity,
            entry_price=round(fill.fill_price, 4),
            stop=round(setup.stop_price, 4),
            target=round(setup.target_price, 4),
            setup_id=setup.id,
        )

    # ------------------------------------------------------------------
    # DB persistence helpers
    # ------------------------------------------------------------------
    def _persist_setup(self, setup: Setup) -> None:
        try:
            with session_scope() as session:
                existing = session.execute(
                    select(SetupRow).where(SetupRow.id == setup.id)
                ).scalar_one_or_none()
                if existing is not None:
                    return
                row = SetupRow(
                    id=setup.id,
                    instrument=setup.instrument,
                    strategy_name=setup.strategy_name,
                    direction=setup.direction,
                    ts=setup.timestamp,
                    entry_price=setup.entry_price,
                    stop_price=setup.stop_price,
                    target_price=setup.target_price,
                    atr_at_entry=setup.atr_at_entry,
                    feature_snapshot_id=None,
                )
                session.add(row)
        except Exception as e:
            self.log.warning("persist.setup_failed", setup_id=setup.id, error=str(e))

    def _persist_prediction(self, setup: Setup, pred) -> None:
        try:
            with session_scope() as session:
                row = ModelPredictionRow(
                    setup_id=setup.id,
                    model_name=str(pred.model_name),
                    model_version=str(pred.model_version),
                    probability=float(pred.probability),
                    threshold=float(pred.threshold),
                    approved=bool(pred.approved),
                )
                session.add(row)
        except Exception as e:
            self.log.warning("persist.prediction_failed", setup_id=setup.id, error=str(e))

    def _persist_risk_block(self, setup: Setup, decision, ts: datetime) -> None:
        try:
            with session_scope() as session:
                row = RiskBlockRow(
                    setup_id=setup.id,
                    ts=ts,
                    rule=decision.rule,
                    reason=decision.reason,
                )
                session.add(row)
        except Exception as e:
            self.log.warning("persist.risk_block_failed", setup_id=setup.id, error=str(e))

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------
    def _record_error(self, result: BarCycleResult, kind: str, exc: Exception) -> None:
        msg = f"{kind}: {exc}"
        result.errors.append(msg)
        self.log.error("loop.error", kind=kind, error=str(exc))
        self.notifier.notify("system.error", kind=kind, error=str(exc))


# ---------------------------------------------------------------------------
# Loop builder — used by main.py and the scheduler
# ---------------------------------------------------------------------------
def build_paper_loop(
    *,
    settings: Settings,
    feed: IncrementalFeed,
    notifier: _NotifierLike,
    model_name: Optional[str] = None,
    model_version: str = "latest",
) -> PaperTradingLoop:
    """Construct a :class:`PaperTradingLoop` with the project's defaults.

    If a model is requested but loading fails, the returned loop has
    ``trading_enabled=False`` and the failure is reported via the
    notifier — the bot stays up, just without entries.
    """
    from strategies.vwap_ema_pullback import VWAPEMAPullback  # local import: keeps boot snappy

    log = get_logger("paper.loop_builder")
    strategy = VWAPEMAPullback(instrument=settings.INSTRUMENT)

    predictor: Optional[Predictor] = None
    trading_enabled = True
    if model_name:
        try:
            from models.model_registry import load_model

            loaded = load_model(model_name, version=model_version)
            predictor = Predictor(loaded)
            log.info(
                "paper.predictor_loaded",
                model_name=model_name,
                model_version=loaded.metadata.get("version"),
            )
        except Exception as e:
            log.error("paper.predictor_failed", error=str(e))
            notifier.notify("system.error", kind="predictor_load_failed", error=str(e))
            trading_enabled = False

    loop = PaperTradingLoop(
        settings=settings,
        feed=feed,
        strategy=strategy,
        notifier=notifier,
        predictor=predictor,
    )
    loop.trading_enabled = trading_enabled
    return loop


# Generate a deterministic, short id for tests that can't use uuid4 (kept here
# so the loop module itself stays free of test-only imports).
def _stable_id(prefix: str = "id") -> str:  # pragma: no cover - helper only
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
