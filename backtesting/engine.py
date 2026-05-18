"""Bar-by-bar backtest engine.

Fill convention (enforced)
--------------------------
- A setup detected on the close of bar ``t`` enters on the **open of bar
  ``t+1``** with adverse slippage. We never let it fill at the close of
  the signal bar — that is a free 1-bar lookahead.
- Same-bar TP and SL ambiguity resolves to **SL first** (label 0). Same
  conservative rule the labeler uses on Day 3.
- ``MAX_HOLD_BARS`` matches the labeler's max_hold by default.
- ``FORCE_FLAT_TIME``: any open position is closed at the bar that
  crosses the configured local-time flat boundary.

Risk gate
---------
Every candidate setup goes through ``risk.risk_engine.evaluate``. A
``RiskDecision(allowed=False, ...)`` is **final** — the engine records it
and moves on, even if the model approved it.

Outputs
-------
``BacktestResult`` carries every closed trade, the equity curve, the
per-day PnL table, the list of risk blocks, and the aggregate metrics.
The report writer (``reports/backtest_report.py``) is what turns this
into Markdown + JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

from typing import Callable

from app.logging_config import get_logger
from backtesting.fills import FillsModel, make_fills_model
from backtesting.metrics import BacktestMetrics, compute_metrics, daily_pnl_table
from backtesting.portfolio import ClosedTradeRecord, Portfolio
from backtesting.trade_management import apply_exit, check_exit
from config.instruments import get_instrument
from config.settings import Settings
from models.predictor import Predictor
from risk.position_sizing import size_position
from risk.risk_engine import RiskConfig, RiskDecision, evaluate
from strategies.base import Setup


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass
class RiskBlockRecord:
    setup_id: str
    instrument: str
    direction: str
    timestamp: datetime
    rule: str
    reason: str

    def to_dict(self) -> dict:
        return {
            "setup_id": self.setup_id,
            "instrument": self.instrument,
            "direction": self.direction,
            "timestamp": self.timestamp.isoformat(),
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass
class BacktestResult:
    instrument: str
    timeframe: str
    closed_trades: list[ClosedTradeRecord] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)
    daily_pnl: list[dict] = field(default_factory=list)
    risk_blocks: list[RiskBlockRecord] = field(default_factory=list)
    metrics: Optional[BacktestMetrics] = None
    starting_equity: float = 0.0
    n_setups_total: int = 0
    n_setups_model_rejected: int = 0
    n_setups_risk_blocked: int = 0
    n_setups_filled: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class BacktestEngine:
    def __init__(
        self,
        *,
        settings: Settings,
        predictor: Optional[Predictor] = None,
        fills_model: Optional[FillsModel] = None,
        starting_equity: float = 0.0,
        max_hold_bars: Optional[int] = None,
        timeframe: str = "1m",
        on_trade_closed: Optional[Callable[[ClosedTradeRecord], None]] = None,
    ) -> None:
        self.settings = settings
        self.predictor = predictor
        self.spec = get_instrument(settings.INSTRUMENT)
        self.portfolio = Portfolio(instrument_spec=self.spec, starting_equity=starting_equity)
        self.risk_config = RiskConfig.from_settings(settings)
        self.timeframe = timeframe
        self.max_hold_bars = int(max_hold_bars or settings.MAX_HOLD_BARS)

        self.fills = fills_model or make_fills_model(
            settings.INSTRUMENT,
            slippage_ticks=settings.SLIPPAGE_TICKS,
            commission_per_contract=settings.COMMISSION_PER_CONTRACT,
            crypto_slippage_bps=settings.CRYPTO_SLIPPAGE_BPS,
            crypto_fee_bps=settings.CRYPTO_FEE_BPS,
        )
        self.tz = ZoneInfo(settings.TIMEZONE)
        self.log = get_logger("backtesting.engine")
        # Day 8: optional in-memory hook for backtest-only callers that
        # want each closed trade as it lands. Defaults to a no-op so the
        # backtester is unchanged for everyone else (including tests).
        self._on_trade_closed: Callable[[ClosedTradeRecord], None] = (
            on_trade_closed or (lambda _r: None)
        )

    # ----------------------------------------------------------------------
    def run(
        self,
        ohlcv_df: pd.DataFrame,
        setups: list[Setup],
    ) -> BacktestResult:
        if not isinstance(ohlcv_df.index, pd.DatetimeIndex):
            raise ValueError("ohlcv_df must have a DatetimeIndex")
        if not ohlcv_df.index.is_monotonic_increasing:
            raise ValueError("ohlcv_df index must be monotonically increasing")

        result = BacktestResult(
            instrument=self.settings.INSTRUMENT,
            timeframe=self.timeframe,
            starting_equity=self.portfolio.starting_equity,
            n_setups_total=len(setups),
        )

        # Index setups by their bar position in ohlcv_df. A setup detected
        # on bar ``t`` enters on bar ``t+1`` open.
        setup_by_bar: dict[int, list[Setup]] = {}
        for s in setups:
            try:
                i = ohlcv_df.index.get_loc(s.timestamp)
            except KeyError:
                # Setup timestamp isn't in the OHLCV — skip silently.
                continue
            setup_by_bar.setdefault(int(i), []).append(s)

        n_bars = len(ohlcv_df)
        timestamps = ohlcv_df.index

        for i in range(n_bars):
            bar = ohlcv_df.iloc[i]
            ts = timestamps[i]

            # 1) Manage any open position FIRST. Exits happen on this bar.
            self._maybe_exit_open_position(ts=ts, bar=bar, bar_index=i, result=result)

            # 2) Process pending entries from the previous bar's setups.
            #    Setups detected on bar i-1 fill on bar i open. After entry,
            #    we re-run the exit check on the *same* bar so an intrabar
            #    TP/SL on the entry bar is honored (otherwise the position
            #    silently survives until the next bar — a favorable bias).
            if i > 0:
                pending = setup_by_bar.get(i - 1, [])
                for setup in pending:
                    self._consider_entry(
                        setup=setup,
                        bar=bar,
                        ts=ts,
                        bar_index=i,
                        result=result,
                    )
                    if not self.portfolio.is_flat():
                        self._maybe_exit_open_position(
                            ts=ts, bar=bar, bar_index=i, result=result
                        )

        # If anything is still open at end-of-data, close at the last bar's close.
        if not self.portfolio.is_flat():
            last_ts = timestamps[-1]
            last_close = float(ohlcv_df.iloc[-1]["close"])
            pos = self.portfolio.open_position
            assert pos is not None
            exit_costs = self.fills.exit(
                direction=pos.direction, raw_price=last_close, quantity=pos.quantity
            )
            self.portfolio.close(
                ts=last_ts.to_pydatetime() if isinstance(last_ts, pd.Timestamp) else last_ts,
                exit_price=exit_costs.fill_price,
                exit_reason="end_of_data",
                commission=exit_costs.commission,
                slippage=exit_costs.slippage,
                bar_index=n_bars - 1,
            )

        # Finalize result.
        result.closed_trades = list(self.portfolio.closed_trades)
        result.equity_curve = list(self.portfolio.equity_curve)
        result.daily_pnl = daily_pnl_table(result.closed_trades)
        result.metrics = compute_metrics(
            result.closed_trades,
            result.equity_curve,
            starting_equity=self.portfolio.starting_equity,
        )

        # Day 8: emit each closed trade to an optional in-memory hook.
        # Backtest does not write closed_trades to the DB, so the
        # PostTradeAnalysisService's DB-driven path does not apply here.
        # Callers that want analysis on backtest results can adapt the
        # records into ``PostTradeAnalysis`` directly via the hook.
        for record in result.closed_trades:
            try:
                self._on_trade_closed(record)
            except Exception as e:
                self.log.warning("backtest.on_trade_closed_failed", error=str(e))

        self.log.info(
            "backtest.complete",
            instrument=self.settings.INSTRUMENT,
            n_setups=result.n_setups_total,
            n_filled=result.n_setups_filled,
            n_risk_blocked=result.n_setups_risk_blocked,
            n_model_rejected=result.n_setups_model_rejected,
            n_trades=result.metrics.n_trades,
            net_pnl=round(result.metrics.net_pnl, 2),
            win_rate=round(result.metrics.win_rate, 4),
            max_dd=round(result.metrics.max_drawdown_dollars, 2),
        )
        return result

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------
    def _maybe_exit_open_position(
        self,
        *,
        ts,
        bar: pd.Series,
        bar_index: int,
        result: BacktestResult,
    ) -> None:
        pos = self.portfolio.open_position
        if pos is None:
            return

        ts_py = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts

        decision = check_exit(
            position=pos,
            bar=bar,
            bar_index=bar_index,
            bar_ts=ts_py,
            max_hold_bars=self.max_hold_bars,
            force_flat_time=self.settings.force_flat_time(),
            market_type=self.settings.MARKET_TYPE,
            tz=self.tz,
        )
        if decision is None:
            return
        apply_exit(
            portfolio=self.portfolio,
            fills=self.fills,
            decision=decision,
            ts=ts_py,
            bar_index=bar_index,
        )

    def _consider_entry(
        self,
        *,
        setup: Setup,
        bar: pd.Series,
        ts,
        bar_index: int,
        result: BacktestResult,
    ) -> None:
        ts_py = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts

        # 1) Optional model gate.
        if self.predictor is not None:
            try:
                prediction = self.predictor.predict_setup(setup)
            except Exception as e:
                self.log.warning(
                    "backtest.predictor_failed",
                    setup_id=setup.id,
                    error=str(e),
                )
                # Refuse the trade if the model fails — safer default than
                # silently letting it through.
                result.n_setups_model_rejected += 1
                return
            if not prediction.approved:
                result.n_setups_model_rejected += 1
                return

        # 2) Risk engine.
        decision: RiskDecision = evaluate(
            setup,
            self.portfolio,
            self.risk_config,
            ts_py,
            instrument_spec=self.spec,
        )
        if not decision.allowed:
            result.n_setups_risk_blocked += 1
            result.risk_blocks.append(
                RiskBlockRecord(
                    setup_id=setup.id,
                    instrument=setup.instrument,
                    direction=setup.direction,
                    timestamp=ts_py,
                    rule=decision.rule,
                    reason=decision.reason,
                )
            )
            return

        # 3) Fill on this bar's open with slippage.
        raw_open = float(bar["open"])
        entry = self.fills.entry(
            direction=setup.direction, raw_price=raw_open, quantity=1.0
        )
        sizing = size_position(
            entry_price=entry.fill_price,
            stop_price=setup.stop_price,
            instrument=self.spec,
            risk_per_trade=self.settings.RISK_PER_TRADE,
            max_position_size=self.settings.MAX_POSITION_SIZE,
        )

        # Recompute per-side commission with the actual size.
        entry = self.fills.entry(
            direction=setup.direction,
            raw_price=raw_open,
            quantity=float(sizing.quantity),
        )

        self.portfolio.open(
            setup_id=setup.id,
            instrument=setup.instrument,
            direction=setup.direction,
            quantity=sizing.quantity,
            ts=ts_py,
            entry_price=entry.fill_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            commission=entry.commission,
            slippage=entry.slippage,
            bar_index=bar_index,
        )
        result.n_setups_filled += 1
