"""Continuous intraday scanner — the autonomous paper-trading loop.

During the trading window it re-scans the (optionally dynamic) universe
every ``WORKFLOW_SCAN_INTERVAL_MINUTES`` minutes:

  for each symbol:
    refresh bars (optional) → SignalEngine.generate_signal
      → long-only filter → risk caps → bracket order (via order_execution)
    manage open positions (stop/target/expiry handled by broker/options)

Safety:
- DRY_RUN never places orders (resolve_order_broker returns None).
- LIVE is refused by the workflow runner / broker router.
- LLM agents stay advisory: they can only rank allowlist names for the
  scan universe, never decide a trade.
- One symbol failing never aborts the cycle; the loop logs and continues.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime, timezone
from typing import Any, Optional

from app.logging_config import get_logger
from config.settings import Settings
from notifications.notification_service import NotificationService
from scheduler.market_hours import is_in_trading_window
from workflows.watchlist import build_scan_universe, propose_agent_watchlist

_log = get_logger("workflows.intraday")


class IntradayLoop:
    """Owns the scan cadence and per-symbol pipeline for autonomous paper."""

    def __init__(
        self,
        settings: Settings,
        *,
        notifier: Optional[NotificationService] = None,
        orchestrator=None,  # noqa: ANN001 — optional AgentOrchestrator
        dry_run: Optional[bool] = None,
    ) -> None:
        self.settings = settings
        self.notifier = notifier or NotificationService.from_settings(settings)
        self.orchestrator = orchestrator
        self.dry_run = (
            dry_run
            if dry_run is not None
            else settings.WORKFLOW_EXECUTION_MODE == "DRY_RUN"
        )
        self.log = _log
        self._stop = threading.Event()
        self._scans = 0
        self._orders = 0
        # Agent watchlist is expensive (a live research call); compute it
        # once per session date and reuse across the 5-min scans.
        self._watchlist_cache: list[str] = []
        self._watchlist_day: Optional[str] = None
        # Per-session daily risk state (reset on new session date).
        self._risk_day: Optional[str] = None
        self._trades_today = 0
        self._halted_today = False
        self._halt_reason = ""

    # ------------------------------------------------------------------
    # Daily risk + reconciliation helpers
    # ------------------------------------------------------------------
    def _reset_daily_state(self, now: datetime) -> None:
        from scheduler.market_hours import session_date

        day = session_date(now, self.settings).isoformat()
        if day != self._risk_day:
            self._risk_day = day
            self._trades_today = 0
            self._halted_today = False
            self._halt_reason = ""
            self.log.info("intraday.session_reset", day=day)

    def _day_pnl(self, broker) -> float:  # noqa: ANN001
        """Best-effort day P&L from the broker account (0.0 if unknown)."""
        try:
            acct = broker.get_account()
            return float(getattr(acct, "realized_pnl", 0.0) or 0.0)
        except Exception:  # noqa: BLE001
            return 0.0

    def _daily_risk_block(self, broker, now: datetime) -> tuple[bool, bool, str]:
        """Return (block_entries, flatten_now, reason).

        Enforces the deterministic daily caps the autonomous loop was
        previously missing: max trades/day, daily loss (flatten + halt),
        daily profit (stop for the day), and force-flat at session close.
        """
        if self._halted_today:
            return True, False, self._halt_reason

        from scheduler.market_hours import is_force_flat_due

        if is_force_flat_due(now, self.settings):
            return True, True, "force_flat_eod"

        if self._trades_today >= self.settings.MAX_TRADES_PER_DAY:
            return True, False, "max_trades_per_day"

        day_pnl = self._day_pnl(broker)
        if self.settings.MAX_DAILY_LOSS > 0 and day_pnl <= -abs(self.settings.MAX_DAILY_LOSS):
            return True, True, f"daily_loss_limit ({day_pnl:.2f})"
        if self.settings.MAX_DAILY_PROFIT > 0 and day_pnl >= abs(self.settings.MAX_DAILY_PROFIT):
            return True, False, f"daily_profit_target ({day_pnl:.2f})"

        return False, False, ""

    @staticmethod
    def _held_or_pending(reconcile: dict) -> set[str]:
        held: set[str] = set()
        for p in reconcile.get("positions", []) or []:
            sym = str(p.get("symbol", "")).upper()
            if sym:
                held.add(sym)
        for o in reconcile.get("orders", []) or []:
            sym = str(o.get("symbol", "")).upper()
            if sym:
                held.add(sym)
        return held

    def _flatten_all(self, broker, reconcile: dict, *, reason: str) -> int:
        from integrations.broker_base import BrokerError

        closed = 0
        for p in reconcile.get("positions", []) or []:
            sym = str(p.get("symbol", "")).upper()
            if not sym:
                continue
            try:
                result = broker.close_position(symbol=sym)
                if getattr(result, "success", False):
                    closed += 1
                    self._notify_safe(
                        "trade.closed", instrument=sym, reason=reason,
                        source="workflow.intraday",
                    )
            except BrokerError as e:
                self.log.warning("intraday.flatten_failed", symbol=sym, error=str(e))
        if closed:
            self.log.info("intraday.flattened", closed=closed, reason=reason)
        return closed

    # ------------------------------------------------------------------
    # Universe
    # ------------------------------------------------------------------
    def _agent_candidates(self, now: datetime) -> list[str]:
        if not (
            self.settings.WORKFLOW_DYNAMIC_UNIVERSE
            and self.settings.WORKFLOW_AGENT_WATCHLIST
        ):
            return []
        from scheduler.market_hours import session_date

        day = session_date(now, self.settings).isoformat()
        if day != self._watchlist_day:
            self._watchlist_cache = propose_agent_watchlist(
                self.settings, orchestrator=self.orchestrator
            )
            self._watchlist_day = day
            self.log.info("intraday.watchlist_refreshed", day=day, picks=self._watchlist_cache)
        return self._watchlist_cache

    def _universe(self, *, now: Optional[datetime] = None) -> list[str]:
        now = now or datetime.now(tz=timezone.utc)
        candidates = self._agent_candidates(now)
        return build_scan_universe(self.settings, agent_candidates=candidates)

    # ------------------------------------------------------------------
    # One scan cycle
    # ------------------------------------------------------------------
    def scan_once(self, *, now: Optional[datetime] = None) -> dict[str, Any]:
        """Run a single scan cycle across the universe. Returns a summary."""
        # Lazy imports keep the module import graph thin + avoid cycles.
        from workflows.base import WorkflowContext
        from workflows.broker_interface import PaperBrokerInterface
        from workflows.gates import WorkflowGates
        from workflows.memory import ensure_memory_files
        from workflows.order_execution import execute_entry_with_stops, resolve_order_broker, prepare_broker
        from workflows.signal_engine import SignalEngine

        now = now or datetime.now(tz=timezone.utc)
        self._scans += 1
        self._reset_daily_state(now)
        universe = self._universe(now=now)

        # Data refresh is read-only market data — safe in dry-run too, so
        # a dry run evaluates the full universe on fresh bars.
        if self.settings.WORKFLOW_REFRESH_DATA_EACH_SCAN:
            self._refresh_data(universe)

        execution_mode = "DRY_RUN" if self.dry_run else self.settings.WORKFLOW_EXECUTION_MODE
        if str(execution_mode).upper() == "LIVE":
            return {"scanned": 0, "skipped": "live_refused"}

        memory = ensure_memory_files(self.settings)
        ctx = WorkflowContext(
            settings=self.settings,
            notifier=self.notifier,
            broker=PaperBrokerInterface(self.settings),
            gates=WorkflowGates(self.settings),
            memory=memory,
            orchestrator=self.orchestrator,
            execution_mode=execution_mode,  # type: ignore[arg-type]
            dry_run=self.dry_run,
            now=now,
            force=True,
        )

        # Reconcile once per cycle before placing new orders.
        reconcile_info: dict[str, Any] = {}
        order_broker = resolve_order_broker(ctx)
        if order_broker is not None:
            reconcile_info = prepare_broker(ctx)

        # ---- Daily risk gate (deterministic, applies before any entry) ---
        risk_block = False
        risk_reason = ""
        if order_broker is not None and not self.dry_run:
            risk_block, flatten_now, risk_reason = self._daily_risk_block(order_broker, now)
            if flatten_now:
                self._flatten_all(order_broker, reconcile_info, reason=risk_reason)
            if risk_block:
                if risk_reason.startswith("daily_loss") or risk_reason == "force_flat_eod":
                    self._halted_today = True
                    self._halt_reason = risk_reason
                self.log.info("intraday.risk_block", reason=risk_reason, scan=self._scans)
                return {
                    "scan": self._scans,
                    "universe_size": len(universe),
                    "entered": 0,
                    "risk_block": risk_reason,
                    "results": [],
                }

        held_or_pending = self._held_or_pending(reconcile_info)

        engine = SignalEngine(
            self.settings,
            model_name=self.settings.WORKFLOW_MODEL_NAME,
            model_version=self.settings.WORKFLOW_MODEL_VERSION,
        )

        results: list[dict[str, Any]] = []
        broker_state = ctx.broker.pull_state(now=now)
        open_positions = max(
            broker_state.account.open_positions, len(held_or_pending)
        )

        for sym in universe:
            if ctx.entries_blocked:
                results.append({"symbol": sym, "action": "skip", "reason": "entries_blocked"})
                continue
            if open_positions >= self.settings.MAX_OPEN_POSITIONS:
                results.append({"symbol": sym, "action": "skip", "reason": "max_open_positions"})
                continue
            # Dedupe: never stack a new order on a symbol we already hold
            # or have a working order for.
            if sym.upper() in held_or_pending:
                results.append({"symbol": sym, "action": "skip", "reason": "already_held_or_pending"})
                continue
            if not self.dry_run and self._trades_today >= self.settings.MAX_TRADES_PER_DAY:
                results.append({"symbol": sym, "action": "skip", "reason": "max_trades_per_day"})
                continue

            try:
                signal = engine.generate_signal(sym)
            except Exception as e:  # noqa: BLE001
                self.log.warning("intraday.signal_error", symbol=sym, error=str(e))
                results.append({"symbol": sym, "action": "error", "reason": str(e)})
                continue

            if signal is None:
                results.append({"symbol": sym, "action": "skip", "reason": "no_setup"})
                continue
            if self.settings.WORKFLOW_LONG_ONLY and signal.direction != "long":
                results.append({"symbol": sym, "action": "skip", "reason": "long_only"})
                continue
            if not signal.approved:
                results.append(
                    {"symbol": sym, "action": "skip", "reason": f"not_approved:{signal.reason}"}
                )
                continue

            if self.dry_run:
                results.append(
                    {
                        "symbol": sym,
                        "action": "would_enter",
                        "direction": signal.direction,
                        "entry": signal.entry_price,
                        "stop": signal.stop_price,
                        "target": signal.target_price,
                    }
                )
                continue

            qty = self._size_position(sym, signal)
            ok, entry, _ = execute_entry_with_stops(
                ctx,
                symbol=sym,
                side=signal.direction,
                quantity=qty,
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                target_price=signal.target_price,
                thesis=f"intraday {signal.direction} {sym} ({signal.reason})",
            )
            if ok:
                self._orders += 1
                self._trades_today += 1
                open_positions += 1
                held_or_pending.add(sym.upper())
                results.append(
                    {"symbol": sym, "action": "enter", "direction": signal.direction,
                     "order_id": entry.order_id if entry else None}
                )
                self._notify_safe(
                    "trade.opened",
                    instrument=sym,
                    direction=signal.direction,
                    entry_price=signal.entry_price,
                    stop_price=signal.stop_price,
                    target_price=signal.target_price,
                    confidence=signal.confidence,
                    source="workflow.intraday",
                )
            else:
                results.append({"symbol": sym, "action": "skip", "reason": "order_failed"})

        summary = {
            "scan": self._scans,
            "universe_size": len(universe),
            "entered": sum(1 for r in results if r["action"] == "enter"),
            "results": results,
        }
        self.log.info(
            "intraday.scan_complete",
            scan=self._scans,
            universe=len(universe),
            entered=summary["entered"],
            dry_run=self.dry_run,
        )
        return summary

    def _size_position(self, symbol: str, signal) -> float:  # noqa: ANN001
        """Risk-based share count for equities; fixed size otherwise."""
        if not self.settings.USE_RISK_BASED_SIZING:
            return float(self.settings.MAX_POSITION_SIZE)
        try:
            from config.instruments import is_supported_symbol, register_equity
            from risk.position_sizing import size_equity_shares

            if not is_supported_symbol(symbol):
                register_equity(symbol)
            res = size_equity_shares(
                entry_price=signal.entry_price,
                stop_price=signal.stop_price,
                instrument=symbol,
                risk_per_trade=float(self.settings.RISK_PER_TRADE),
                max_shares=int(self.settings.MAX_SHARES_PER_TRADE),
                max_notional=float(self.settings.MAX_NOTIONAL_PER_TRADE),
            )
            return float(res.quantity)
        except Exception as e:  # noqa: BLE001
            self.log.warning("intraday.sizing_failed", symbol=symbol, error=str(e))
            return float(self.settings.MAX_POSITION_SIZE)

    def _refresh_data(self, symbols: list[str]) -> None:
        try:
            from data.alpaca_bars import download_symbols

            download_symbols(self.settings, symbols=symbols, timeframe="1m", days=30)
        except Exception as e:  # noqa: BLE001
            self.log.warning("intraday.data_refresh_failed", error=str(e))

    def _notify_safe(self, kind: str, **payload: Any) -> None:
        try:
            self.notifier.notify(kind, **payload)
        except Exception as e:  # noqa: BLE001
            self.log.warning("intraday.notify_failed", kind=kind, error=str(e))

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run_forever(self, *, max_cycles: Optional[int] = None) -> None:
        """Block, scanning every interval during the trading window.

        Outside the trading window the loop sleeps without scanning. Set
        ``max_cycles`` in tests to bound the run.
        """
        interval_s = self.settings.WORKFLOW_SCAN_INTERVAL_MINUTES * 60
        self.log.info(
            "intraday.start",
            interval_minutes=self.settings.WORKFLOW_SCAN_INTERVAL_MINUTES,
            dry_run=self.dry_run,
            execution_mode=self.settings.WORKFLOW_EXECUTION_MODE,
        )
        self._notify_safe(
            "system.info",
            source="workflow.intraday",
            message=(
                f"Intraday loop started ({self.settings.WORKFLOW_EXECUTION_MODE}, "
                f"every {self.settings.WORKFLOW_SCAN_INTERVAL_MINUTES}m, "
                f"long_only={self.settings.WORKFLOW_LONG_ONLY})"
            ),
        )
        cycles = 0
        while not self._stop.is_set():
            now = datetime.now(tz=timezone.utc)
            if is_in_trading_window(now, self.settings):
                try:
                    self.scan_once(now=now)
                except Exception as e:  # noqa: BLE001
                    self.log.exception("intraday.cycle_failed", error=str(e))
                    self._notify_safe(
                        "system.error", source="workflow.intraday", error=str(e)
                    )
            else:
                self.log.info("intraday.outside_window", now=now.isoformat())
            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                break
            self._stop.wait(interval_s)
        self.log.info("intraday.stopped", scans=self._scans, orders=self._orders)

    def stop(self) -> None:
        self._stop.set()
