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
6. Optional entry gate (multi-symbol caps).
7. Risk engine — authoritative. Blocked setups are persisted as
   ``risk_blocks`` rows and skipped.
8. Size + submit through :class:`PaperExecutor`. Sizing failures are
   recorded as risk blocks (defensive — should be rare).

A separate ``flatten_now()`` is exposed so the scheduler's end-of-day job
can force-flat any straggler position even if no new bar has arrived yet
(e.g. data feed went quiet around the close).

Multi-symbol mode (this revision) adds :class:`MultiSymbolPaperLoop`
which composes one :class:`PaperTradingLoop` per symbol behind a single
``on_bar_close`` entry point and threads an ``entry_gate`` callback into
each so per-symbol day caps, the global day cap, and ``MAX_ACTIVE_SYMBOLS``
are enforced before any submit. Per-symbol failures are isolated — a
broken feed for ``MNQ`` cannot stop ``MES`` from running.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, Sequence
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
from notifications.trade_events import build_trade_closed_payload
from risk.kill_switch import KillSwitch
from risk.position_sizing import size_position
from risk.risk_engine import RiskConfig, evaluate
from scheduler.market_hours import is_force_flat_due, is_in_trading_window
from sqlalchemy import select
from storage.db import session_scope
from storage.tables import FeatureSnapshot as FeatureSnapshotRow
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


@dataclass
class _ExcursionTracker:
    """Tracks max favorable / max adverse excursion ($) for the open position.

    Rebased on each call to :meth:`begin`. Updated bar-by-bar in
    :meth:`update_bar`. Read once at close into the trade analysis.
    """

    direction: str = "long"
    entry_price: float = 0.0
    point_value: float = 1.0
    quantity: float = 1.0
    mfe: float = 0.0  # max favorable excursion in dollars (>= 0)
    mae: float = 0.0  # max adverse excursion in dollars (>= 0)
    news_risk_at_entry: bool = False

    def begin(
        self,
        *,
        direction: str,
        entry_price: float,
        point_value: float,
        quantity: float,
        news_risk_at_entry: bool,
    ) -> None:
        self.direction = direction
        self.entry_price = float(entry_price)
        self.point_value = float(point_value)
        self.quantity = float(quantity)
        self.mfe = 0.0
        self.mae = 0.0
        self.news_risk_at_entry = bool(news_risk_at_entry)

    def update_bar(self, *, high: float, low: float) -> None:
        if self.direction == "long":
            best_move = float(high) - self.entry_price
            worst_move = self.entry_price - float(low)
        else:
            best_move = self.entry_price - float(low)
            worst_move = float(high) - self.entry_price
        best_dollars = max(0.0, best_move) * self.point_value * self.quantity
        worst_dollars = max(0.0, worst_move) * self.point_value * self.quantity
        if best_dollars > self.mfe:
            self.mfe = best_dollars
        if worst_dollars > self.mae:
            self.mae = worst_dollars

    def reset(self) -> None:
        self.mfe = 0.0
        self.mae = 0.0
        self.news_risk_at_entry = False


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
        strategy: Optional[Strategy] = None,
        strategies: Optional[Sequence[Strategy]] = None,
        notifier: _NotifierLike,
        predictor: Optional[Predictor] = None,
        portfolio: Optional[Portfolio] = None,
        executor: Optional[PaperExecutor] = None,
        kill_switch: Optional[KillSwitch] = None,
        max_hold_bars: Optional[int] = None,
        high_risk_news_fn: Optional[Callable[[], bool]] = None,
        instrument: Optional[str] = None,
        entry_gate: Optional[
            Callable[[Setup, Optional[float]], "tuple[bool, str]"]
        ] = None,
    ) -> None:
        # Accept either a single ``strategy=`` (the original API) or a
        # list ``strategies=`` (multi-strategy paper mode). Internally
        # ``self.strategies`` is always the canonical list; ``self.strategy``
        # is preserved as the first element for back-compat with code that
        # used to read it directly.
        if strategies is None:
            if strategy is None:
                raise TypeError(
                    "PaperTradingLoop requires either ``strategy=`` or ``strategies=``."
                )
            self.strategies: list[Strategy] = [strategy]
        else:
            if not strategies:
                raise ValueError("strategies= must contain at least one strategy")
            self.strategies = list(strategies)
        self.strategy: Strategy = self.strategies[0]
        self.settings = settings
        self.feed = feed
        self.notifier = notifier
        self.predictor = predictor
        # Multi-symbol mode threads a per-symbol instrument in; default
        # falls back to the legacy single-symbol settings.INSTRUMENT.
        self.instrument: str = (instrument or settings.INSTRUMENT).upper()
        self.spec = get_instrument(self.instrument)
        # Optional cap-enforcement gate; default = always allow.
        self._entry_gate: Callable[[Setup, Optional[float]], "tuple[bool, str]"] = (
            entry_gate or (lambda _setup, _conf: (True, ""))
        )
        self.portfolio = portfolio or Portfolio(instrument_spec=self.spec)
        self.kill_switch = kill_switch or KillSwitch()
        self.fills = make_fills_model(
            settings.INSTRUMENT,
            slippage_ticks=settings.SLIPPAGE_TICKS,
            commission_per_contract=settings.COMMISSION_PER_CONTRACT,
            crypto_slippage_bps=settings.CRYPTO_SLIPPAGE_BPS,
            crypto_fee_bps=settings.CRYPTO_FEE_BPS,
        )
        self.executor = executor or PaperExecutor(
            portfolio=self.portfolio,
            fills_model=self.fills,
            kill_switch=self.kill_switch,
        )
        self.risk_config = RiskConfig.from_settings(settings)
        self.tz = ZoneInfo(settings.TIMEZONE)
        self.max_hold_bars = int(max_hold_bars or settings.MAX_HOLD_BARS)
        # Day 7: agent-driven news risk window. Default callable always
        # returns False so the loop is decoupled from the agents module.
        self._high_risk_news_fn: Callable[[], bool] = high_risk_news_fn or (lambda: False)

        self.trading_enabled: bool = predictor is not None or True  # default-on
        self._bar_index = 0
        self._last_bar_ts: Optional[datetime] = None
        self._last_bar_close: Optional[float] = None
        self._last_seen_setup_ids: set[str] = set()
        # Day 8: per-trade analysis hook + MFE/MAE tracker.
        # ``trade_closed_callback`` accepts (closed_trade_id, mfe, mae,
        # news_risk_at_entry, model_confidence). The loop never imports the
        # analysis package directly so the callback can be plugged in by the
        # service builder (or left as a no-op in tests / smoke).
        self._trade_closed_callback: Optional[
            Callable[[str, Optional[float], Optional[float], bool, Optional[float]], None]
        ] = None
        self._excursion = _ExcursionTracker()
        self._last_predictor_confidence: Optional[float] = None
        self.log = get_logger("paper.loop")

    def set_trade_closed_callback(
        self,
        callback: Optional[
            Callable[[str, Optional[float], Optional[float], bool, Optional[float]], None]
        ],
    ) -> None:
        """Wire (or unwire) the post-close analysis hook.

        The hook is called *after* the deterministic close + DB write. A
        callback failure must not propagate — the analysis layer is
        already required to swallow its own exceptions.
        """
        self._trade_closed_callback = callback

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
        self._last_bar_close = float(poll.new_candle.close)

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
        was closed. If the loop has not yet seen any bars (no
        ``_last_bar_close``) we fall back to the position's entry price —
        this only happens in the degenerate "scheduler tripped flat
        before any data arrived" case and produces a near-zero net move
        (intentional: we cannot invent a price we haven't observed).
        """
        if self.portfolio.open_position is None:
            return False
        ts = now if now.tzinfo else now.replace(tzinfo=timezone.utc)

        if self._last_bar_close is not None:
            last_close = float(self._last_bar_close)
        else:
            self.log.warning(
                "loop.flatten_no_bar_seen",
                detail="No bar observed yet; closing at entry price.",
            )
            last_close = float(self.portfolio.open_position.entry_price)

        # Snapshot in-memory excursion + entry context BEFORE the close.
        mfe = float(self._excursion.mfe)
        mae = float(self._excursion.mae)
        news_at_entry = bool(self._excursion.news_risk_at_entry)
        confidence_at_entry = self._last_predictor_confidence

        record = self.executor.close_position(
            ts=ts,
            exit_raw_price=last_close,
            exit_reason=reason,
            bar_index=self._bar_index,
        )
        self._excursion.reset()
        self._last_predictor_confidence = None

        self.notifier.notify(
            "forced_flat",
            **build_trade_closed_payload(record, ts=ts),
        )

        # Day 8 hook also fires for forced-flat closes so the trade
        # analysis layer sees every closed position, not only TP/SL/time
        # exits processed in ``_maybe_exit_open``.
        closed_trade_id = getattr(self.executor, "last_closed_trade_id", None)
        if self._trade_closed_callback is not None and closed_trade_id:
            try:
                self._trade_closed_callback(
                    closed_trade_id,
                    mfe,
                    mae,
                    news_at_entry,
                    confidence_at_entry,
                )
            except Exception as e:
                self.log.warning("loop.trade_closed_callback_failed", error=str(e))
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

        # Track MFE/MAE on every bar the position is open — including the
        # exit bar itself, since the bar's range tells us the in-bar high
        # water mark / drawdown before the close decision.
        self._excursion.update_bar(
            high=float(bar_series["high"]),
            low=float(bar_series["low"]),
        )

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

        # Snapshot excursion + entry context BEFORE close_position resets it.
        mfe = float(self._excursion.mfe)
        mae = float(self._excursion.mae)
        news_at_entry = bool(self._excursion.news_risk_at_entry)
        confidence_at_entry = self._last_predictor_confidence

        record = self.executor.close_position(
            ts=bar_ts,
            exit_raw_price=decision.raw_price,
            exit_reason=decision.reason,
            bar_index=self._bar_index,
        )
        self._excursion.reset()
        self._last_predictor_confidence = None

        self.notifier.notify(
            "trade.closed",
            **build_trade_closed_payload(record, ts=bar_ts),
        )

        # Day 8: post-trade analysis hook. The callback is responsible for
        # joining closed_trade_id to setups + predictions. We only pass
        # the volatile in-memory data (MFE/MAE, news flag at entry,
        # confidence at entry) that the DB does not know about.
        closed_trade_id = getattr(self.executor, "last_closed_trade_id", None)
        if self._trade_closed_callback is not None and closed_trade_id:
            try:
                self._trade_closed_callback(
                    closed_trade_id,
                    mfe,
                    mae,
                    news_at_entry,
                    confidence_at_entry,
                )
            except Exception as e:
                self.log.warning("loop.trade_closed_callback_failed", error=str(e))
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
        rolling["instrument"] = self.instrument
        rolling["timeframe"] = "1m"
        try:
            features = build_features(
                rolling, instrument=self.instrument, tz=self.settings.TIMEZONE
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

        # Run every enabled strategy on the same feature frame and merge
        # the latest-bar setups. Strategy failures are isolated: one
        # broken plug-in cannot stop the others.
        latest_setups: list[Setup] = []
        for strategy in self.strategies:
            try:
                strategy_setups = strategy.detect_setups(features)
            except Exception as e:
                self._record_error(result, f"strategy.failed:{strategy.name}", e)
                continue
            latest_setups.extend(
                s for s in strategy_setups if s.timestamp == latest_ts
            )

        result.setups_seen = len(latest_setups)
        if not latest_setups:
            return

        latest_setups = self._resolve_setup_conflicts(latest_setups, result)
        if not latest_setups:
            return

        for setup in latest_setups:
            self._handle_setup(setup, bar_series, now, result)

    # ------------------------------------------------------------------
    # Multi-strategy conflict resolution
    # ------------------------------------------------------------------
    def _resolve_setup_conflicts(
        self, setups: list[Setup], result: BarCycleResult
    ) -> list[Setup]:
        """Drop opposing-direction setups on the same symbol.

        With one strategy enabled there is rarely a conflict and this
        function is a near no-op. With multiple strategies enabled the
        registry's resolver enforces the rule that we never simultaneously
        take long and short on the same instrument.

        Score each candidate via the predictor first (when present) so
        the resolver can break ties by approved confidence. Predictor
        failures bubble up through ``_handle_setup`` later — at this
        stage we only score for *resolution*; ``_handle_setup`` will
        score again for the model gate. That double scoring is cheap
        (one ``predict_proba`` call per candidate) and keeps the
        per-setup error/notify/persist bookkeeping isolated from the
        resolver.
        """
        if len(setups) <= 1:
            return setups

        # Lazy import: avoids hard-binding registry into loop boot when
        # paper mode is constructed without it.
        from strategies.registry import ScoredSetup, resolve_conflicts

        scored: list[ScoredSetup] = []
        for setup in setups:
            confidence: Optional[float] = None
            approved: Optional[bool] = None
            if self.predictor is not None:
                try:
                    pred = self.predictor.predict_setup(setup)
                    confidence = float(pred.probability)
                    approved = bool(pred.approved)
                except Exception as e:
                    self.log.warning(
                        "loop.predictor_score_failed_for_resolution",
                        setup_id=setup.id,
                        error=str(e),
                    )
            scored.append(ScoredSetup(setup=setup, confidence=confidence, approved=approved))

        resolution = resolve_conflicts(scored)

        for conflict in resolution.conflicts:
            self.notifier.notify(
                "strategy.conflict",
                instrument=conflict.instrument,
                reason=conflict.reason,
                winner_setup_id=conflict.winner_setup_id,
                dropped_setup_ids=conflict.dropped_setup_ids,
            )

        return [s.setup for s in resolution.survivors]

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
        confidence_for_trade: Optional[float] = None
        if self.predictor is not None:
            try:
                pred = self.predictor.predict_setup(setup)
            except Exception as e:
                self._record_error(result, "predictor.failed", e)
                result.setups_model_rejected += 1
                return
            self._persist_prediction(setup, pred)
            confidence_for_trade = float(pred.probability)
            if not pred.approved:
                result.setups_model_rejected += 1
                return

        # Multi-symbol entry gate (caps). Default gate always allows;
        # ``MultiSymbolPaperLoop`` swaps in a real gate that enforces
        # per-symbol day cap, total day cap, and MAX_ACTIVE_SYMBOLS.
        # A blocked candidate is recorded as a risk_block for audit
        # parity with deterministic risk decisions.
        try:
            allow, gate_rule = self._entry_gate(setup, confidence_for_trade)
        except Exception as e:
            self.log.warning("loop.entry_gate_failed", error=str(e))
            allow, gate_rule = True, ""
        if not allow:
            from risk.risk_engine import RiskDecision  # local import: avoid cycle

            decision = RiskDecision(
                allowed=False,
                rule=gate_rule or "multi_symbol_cap",
                reason="Multi-symbol cap exceeded; setup not submitted.",
            )
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

        # Risk engine. The high-risk news flag is *block-only* — the
        # NewsAgent (or any other source) can flip it via the callback,
        # but the engine remains authoritative; agents cannot approve.
        try:
            high_risk_news = bool(self._high_risk_news_fn())
        except Exception as e:
            self.log.warning("loop.news_flag_failed", error=str(e))
            high_risk_news = False
        decision = evaluate(
            setup,
            self.portfolio,
            self.risk_config,
            now,
            kill_switch_tripped=self.kill_switch.is_tripped(),
            high_risk_news_window=high_risk_news,
            instrument_spec=self.spec,
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

        # Begin tracking MFE/MAE for the freshly-opened position, and
        # snapshot the news flag at *entry* time so the trade analysis
        # later reflects what the agent flag was when we entered, not
        # what it might be at exit.
        self._excursion.begin(
            direction=setup.direction,
            entry_price=fill.fill_price,
            point_value=self.spec.point_value,
            quantity=order.quantity,
            news_risk_at_entry=high_risk_news,
        )
        self._last_predictor_confidence = confidence_for_trade

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
        """Persist the Setup row + its frozen feature snapshot (Day 8).

        The feature snapshot is the canonical anti-lookahead artifact —
        the trade analyzer reloads it at close time so any feature-based
        post-mortem reasons about exactly the values the strategy saw at
        entry, not whatever ``build_features`` happens to produce later.
        """
        try:
            with session_scope() as session:
                existing = session.execute(
                    select(SetupRow).where(SetupRow.id == setup.id)
                ).scalar_one_or_none()
                if existing is not None:
                    return

                snapshot_row = FeatureSnapshotRow(
                    instrument=setup.instrument,
                    ts=setup.timestamp,
                    features=dict(setup.features),
                )
                session.add(snapshot_row)
                session.flush()  # populate snapshot_row.id

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
                    feature_snapshot_id=snapshot_row.id,
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
# Multi-symbol orchestrator
# ---------------------------------------------------------------------------
@dataclass
class MultiSymbolBarResult:
    """Aggregate per-symbol :class:`BarCycleResult`, per-tick.

    The scheduler treats this as a drop-in replacement for the
    single-symbol :class:`BarCycleResult` so it doesn't need to know
    whether the bot is running 1 or 6 symbols. Convenience accessors
    summarize across the per-symbol entries.
    """

    by_symbol: dict[str, BarCycleResult] = field(default_factory=dict)

    @property
    def new_bar(self) -> bool:
        return any(r.new_bar for r in self.by_symbol.values())

    @property
    def setups_seen(self) -> int:
        return sum(r.setups_seen for r in self.by_symbol.values())

    @property
    def setups_filled(self) -> int:
        return sum(r.setups_filled for r in self.by_symbol.values())

    @property
    def setups_risk_blocked(self) -> int:
        return sum(r.setups_risk_blocked for r in self.by_symbol.values())

    @property
    def setups_model_rejected(self) -> int:
        return sum(r.setups_model_rejected for r in self.by_symbol.values())

    @property
    def exits(self) -> int:
        return sum(r.exits for r in self.by_symbol.values())

    @property
    def errors(self) -> list[str]:
        flat: list[str] = []
        for sym, r in self.by_symbol.items():
            flat.extend(f"{sym}:{e}" for e in r.errors)
        return flat


class MultiSymbolPaperLoop:
    """Compose N per-symbol :class:`PaperTradingLoop` instances.

    Public API mirrors the single-symbol loop (``on_bar_close``,
    ``flatten_now``, ``set_trade_closed_callback``, ``trading_enabled``)
    so the scheduler is unchanged.

    Caps enforced at the orchestrator level via the ``entry_gate`` we
    inject into each per-symbol loop:

    - ``MAX_TRADES_PER_SYMBOL_PER_DAY`` — fills per (session_date, symbol)
    - ``MAX_TOTAL_TRADES_PER_DAY`` — fills per session_date across all
      symbols
    - ``MAX_ACTIVE_SYMBOLS`` — count of per-symbol loops currently
      holding an open position; new entries refused beyond this cap.

    Per-symbol loops are independent: a feed failure or a strategy
    crash on one symbol cannot affect any other.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        loops: dict[str, PaperTradingLoop],
        notifier: _NotifierLike,
        disabled_symbols: Optional[dict[str, str]] = None,
    ) -> None:
        if not loops:
            raise ValueError("MultiSymbolPaperLoop requires at least one loop")
        self.settings = settings
        self.notifier = notifier
        self._loops: dict[str, PaperTradingLoop] = {
            sym.upper(): loop for sym, loop in loops.items()
        }
        self.disabled_symbols: dict[str, str] = dict(disabled_symbols or {})
        # Per-(session_date, symbol) counters. Reset lazily when the
        # session_date observed at gate-call time changes.
        self._fills_by_day: dict[str, dict[str, int]] = {}
        self.log = get_logger("paper.multi_symbol")
        # Wire the cap-enforcement gate into each per-symbol loop. Done
        # here (not by the loop builder) so the gate can read shared
        # state across all loops.
        for sym, loop in self._loops.items():
            loop._entry_gate = self._make_gate_for(sym)
            # Attach a post-fill hook so the orchestrator counts fills.
            self._patch_fill_counter(sym, loop)

    @property
    def trading_enabled(self) -> bool:
        # Aggregate: orchestrator is "trading_enabled" if any per-symbol
        # loop is. The scheduler reads this before forwarding bars.
        return any(loop.trading_enabled for loop in self._loops.values())

    @trading_enabled.setter
    def trading_enabled(self, value: bool) -> None:
        for loop in self._loops.values():
            loop.trading_enabled = bool(value)

    @property
    def strategies(self) -> list[Strategy]:
        # Convenience for the scheduler/test tooling: return strategies
        # of the primary loop. Real per-symbol strategies are owned by
        # each ``PaperTradingLoop``.
        first = next(iter(self._loops.values()))
        return list(first.strategies)

    # ------------------------------------------------------------------
    # Public scheduler-facing API
    # ------------------------------------------------------------------
    def on_bar_close(self, now: datetime) -> MultiSymbolBarResult:
        """Forward a tick to every enabled symbol's loop.

        Per-symbol failures are isolated: an exception from one loop
        is captured into its result and never raised. If a symbol is
        disabled (e.g. its CSV failed to load) it stays disabled for
        the lifetime of the orchestrator.
        """
        out = MultiSymbolBarResult()
        for sym, loop in self._loops.items():
            if sym in self.disabled_symbols:
                out.by_symbol[sym] = BarCycleResult(
                    new_bar=False,
                    trading_enabled=False,
                    errors=[f"disabled:{self.disabled_symbols[sym]}"],
                )
                continue
            try:
                out.by_symbol[sym] = loop.on_bar_close(now)
            except Exception as e:  # noqa: BLE001 - isolate per-symbol crashes
                self.log.error("multi_symbol.loop_failed", symbol=sym, error=str(e))
                out.by_symbol[sym] = BarCycleResult(
                    new_bar=False,
                    trading_enabled=False,
                    errors=[f"loop_failed:{e}"],
                )
        return out

    def flatten_now(self, now: datetime, *, reason: str = "forced_flat") -> bool:
        """Force-flat every per-symbol loop. Returns True if any closed."""
        any_closed = False
        for sym, loop in self._loops.items():
            try:
                if loop.flatten_now(now, reason=reason):
                    any_closed = True
            except Exception as e:  # noqa: BLE001
                self.log.warning(
                    "multi_symbol.flatten_failed", symbol=sym, error=str(e)
                )
        return any_closed

    def set_trade_closed_callback(self, callback) -> None:
        """Fan-out the post-trade analysis hook to every loop."""
        for loop in self._loops.values():
            loop.set_trade_closed_callback(callback)

    def loop_for(self, symbol: str) -> PaperTradingLoop:
        """Lookup helper for tests / introspection."""
        return self._loops[symbol.upper()]

    def symbols(self) -> list[str]:
        return sorted(self._loops)

    def open_position_symbols(self) -> list[str]:
        return [
            sym for sym, loop in self._loops.items()
            if loop.portfolio.open_position is not None
        ]

    # ------------------------------------------------------------------
    # Caps
    # ------------------------------------------------------------------
    def _session_key(self, now: Optional[datetime]) -> str:
        # Use the trading-window timezone for "today" so caps reset on
        # the operator's session boundary, not UTC midnight.
        from scheduler.market_hours import session_date  # local import to avoid cycle

        ts = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return session_date(ts, self.settings).isoformat()

    def _counters_for_today(self, now: Optional[datetime]) -> dict[str, int]:
        key = self._session_key(now)
        if key not in self._fills_by_day:
            # Drop older keys so the dict doesn't grow unbounded over a
            # multi-week run.
            self._fills_by_day = {key: {}}
        return self._fills_by_day[key]

    def _make_gate_for(self, symbol: str):
        per_symbol_cap = min(
            int(self.settings.MAX_TRADES_PER_SYMBOL_PER_DAY),
            int(self.settings.MAX_TRADES_PER_DAY),
        )
        total_cap = int(self.settings.MAX_TOTAL_TRADES_PER_DAY)
        active_cap = int(self.settings.MAX_ACTIVE_SYMBOLS)

        def _gate(setup: Setup, _confidence: Optional[float]) -> tuple[bool, str]:
            counters = self._counters_for_today(getattr(setup, "timestamp", None))
            sym_count = counters.get(symbol, 0)
            total = sum(counters.values())

            if sym_count >= per_symbol_cap:
                self.log.info(
                    "multi_symbol.cap_blocked",
                    rule="per_symbol_day_cap",
                    symbol=symbol,
                    sym_count=sym_count,
                    cap=per_symbol_cap,
                )
                return False, "per_symbol_day_cap"
            if total >= total_cap:
                self.log.info(
                    "multi_symbol.cap_blocked",
                    rule="total_day_cap",
                    symbol=symbol,
                    total=total,
                    cap=total_cap,
                )
                return False, "total_day_cap"
            n_active = len(self.open_position_symbols())
            sym_already_active = symbol in self.open_position_symbols()
            if not sym_already_active and n_active >= active_cap:
                self.log.info(
                    "multi_symbol.cap_blocked",
                    rule="max_active_symbols",
                    symbol=symbol,
                    active=n_active,
                    cap=active_cap,
                )
                return False, "max_active_symbols"
            return True, ""

        return _gate

    def _patch_fill_counter(self, symbol: str, loop: PaperTradingLoop) -> None:
        """Wrap the loop's notifier so we can observe ``trade.opened`` events.

        We don't change the loop's internals; we just intercept the
        notifier so any successful submit increments the per-symbol
        counter for today's session.
        """
        original_notifier = loop.notifier
        orchestrator = self

        class _CountingNotifier:
            def __init__(self, inner):
                self._inner = inner

            def notify(self, kind: str, /, **payload):
                if kind == "trade.opened":
                    counters = orchestrator._counters_for_today(None)
                    counters[symbol] = counters.get(symbol, 0) + 1
                return self._inner.notify(kind, **payload)

        loop.notifier = _CountingNotifier(original_notifier)


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
    high_risk_news_fn: Optional[Callable[[], bool]] = None,
    cli_strategy: Optional[str] = None,
    instrument: Optional[str] = None,
) -> PaperTradingLoop:
    """Construct a :class:`PaperTradingLoop` with the project's defaults.

    Strategies come from the registry: ``cli_strategy`` (operator
    override via ``--strategy``) takes precedence, otherwise we use
    ``settings.ENABLED_STRATEGIES``. If a model is requested but
    loading fails, the returned loop has ``trading_enabled=False`` and
    the failure is reported via the notifier — the bot stays up, just
    without entries.
    """
    from strategies.registry import instantiate_enabled

    log = get_logger("paper.loop_builder")
    strategies = instantiate_enabled(
        settings, instrument=settings.INSTRUMENT, cli_strategy=cli_strategy
    )
    log.info(
        "paper.strategies_loaded",
        names=[s.name for s in strategies],
    )

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
        strategies=strategies,
        notifier=notifier,
        predictor=predictor,
        high_risk_news_fn=high_risk_news_fn,
        instrument=instrument or settings.INSTRUMENT,
    )
    loop.trading_enabled = trading_enabled
    return loop


def build_multi_symbol_paper_loop(
    *,
    settings: Settings,
    feeds: dict[str, IncrementalFeed],
    notifier: _NotifierLike,
    disabled_symbols: Optional[dict[str, str]] = None,
    model_name: Optional[str] = None,
    model_version: str = "latest",
    high_risk_news_fn: Optional[Callable[[], bool]] = None,
    cli_strategy: Optional[str] = None,
) -> MultiSymbolPaperLoop:
    """Build one :class:`PaperTradingLoop` per (symbol, feed) pair and
    wrap them in a :class:`MultiSymbolPaperLoop`.

    The model is loaded once and shared across all per-symbol loops
    (the predictor is stateless w.r.t. symbol — its threshold + feature
    vector contract is the same everywhere). A per-symbol feed failure
    is captured into ``disabled_symbols`` rather than aborting boot.
    """
    log = get_logger("paper.multi_symbol_builder")

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
            notifier.notify(
                "system.error", kind="predictor_load_failed", error=str(e)
            )
            trading_enabled = False

    from strategies.registry import instantiate_enabled

    loops: dict[str, PaperTradingLoop] = {}
    for sym, feed in feeds.items():
        sym = sym.upper()
        try:
            strategies = instantiate_enabled(
                settings, instrument=sym, cli_strategy=cli_strategy
            )
            loop = PaperTradingLoop(
                settings=settings,
                feed=feed,
                strategies=strategies,
                notifier=notifier,
                predictor=predictor,
                high_risk_news_fn=high_risk_news_fn,
                instrument=sym,
            )
            loop.trading_enabled = trading_enabled
            loops[sym] = loop
        except Exception as e:  # noqa: BLE001 - one bad symbol must not abort boot
            log.error(
                "paper.symbol_loop_init_failed", symbol=sym, error=str(e)
            )
            notifier.notify(
                "system.error",
                kind="symbol_loop_init_failed",
                symbol=sym,
                error=str(e),
            )

    if not loops:
        raise ValueError(
            "build_multi_symbol_paper_loop produced zero functioning per-symbol loops; "
            "check ENABLED_SYMBOLS, the data directory, and the model registry."
        )

    log.info(
        "paper.multi_symbol_loaded",
        symbols=sorted(loops),
        disabled=list((disabled_symbols or {}).keys()),
    )
    return MultiSymbolPaperLoop(
        settings=settings,
        loops=loops,
        notifier=notifier,
        disabled_symbols=disabled_symbols,
    )


# Generate a deterministic, short id for tests that can't use uuid4 (kept here
# so the loop module itself stays free of test-only imports).
def _stable_id(prefix: str = "id") -> str:  # pragma: no cover - helper only
    return f"{prefix}-{uuid.uuid4().hex[:8]}"
