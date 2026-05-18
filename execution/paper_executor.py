"""Paper trading executor.

Simulates fills using the same :class:`FillsModel` the backtester uses, so
paper-mode economics match backtest economics. Persists every open and
close to SQLite so the daily report (Day 6) and any forensic post-mortem
have a real audit trail.

Architectural rules (enforced):

- The executor does not talk to the model and does not consult the risk
  engine. It is a pass-through that handles fills + bookkeeping.
- The executor does not auto-decide exits. The :class:`PaperTradingLoop`
  calls ``close_position`` with an explicit reason after running the
  shared trade-management exit check.
- If the kill switch is tripped, ``submit`` raises ``KillSwitchActive``.
  The loop catches this so a tripped switch can never produce a trade.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlalchemy import select

from app.logging_config import get_logger
from backtesting.fills import FillsModel
from backtesting.portfolio import ClosedTradeRecord, Portfolio
from execution.base import Executor, Fill, Order
from risk.kill_switch import KillSwitch
from storage.db import session_scope
from storage.tables import ClosedTrade as ClosedTradeRow
from storage.tables import PaperTrade as PaperTradeRow


class KillSwitchActive(RuntimeError):
    """Raised by :meth:`PaperExecutor.submit` when the kill switch is tripped."""


@dataclass(frozen=True)
class PaperOpen:
    """Result of a successful paper open. Carries the DB row id for later close."""

    fill: Fill
    paper_trade_id: str


class PaperExecutor(Executor):
    """Single-account paper executor.

    Holds a reference to a :class:`Portfolio` so every open/close also
    updates in-process state (the loop reads day stats from the portfolio
    when calling ``risk_engine.evaluate``). The portfolio and DB rows are
    kept in sync; tests can assert on either.
    """

    def __init__(
        self,
        *,
        portfolio: Portfolio,
        fills_model: FillsModel,
        kill_switch: Optional[KillSwitch] = None,
    ) -> None:
        self.portfolio = portfolio
        self.fills = fills_model
        self.kill_switch = kill_switch or KillSwitch()
        self._open_paper_id: Optional[str] = None
        # Day 8: post-trade analysis hook needs the closed_trades.id of
        # the most recent close so it can join setups + predictions
        # without re-querying by (setup_id, exit_ts).
        self.last_closed_trade_id: Optional[str] = None
        self.log = get_logger("execution.paper_executor")

    # ---------------- Order submission ------------------------------------
    def submit(self, order: Order) -> Fill:
        """Open a new paper position.

        Refuses if the kill switch is tripped or another position is open.
        Returns a :class:`Fill` whose price already includes adverse
        slippage from the configured :class:`FillsModel`.
        """
        if self.kill_switch.is_tripped():
            raise KillSwitchActive("Kill switch is tripped — refusing paper submit.")
        if self.portfolio.open_position is not None:
            raise RuntimeError(
                "PaperExecutor.submit called while another position is open."
            )

        costs = self.fills.entry(
            direction=order.direction,
            raw_price=order.entry_price,
            quantity=order.quantity,
        )
        ts = datetime.now().astimezone()  # local-tz wall clock; loop also passes ts

        # Stage in the portfolio FIRST. If the DB write fails we'll roll
        # back and reset the portfolio.
        self.portfolio.open(
            setup_id=str(order.setup_id or ""),
            instrument=order.instrument,
            direction=order.direction,
            quantity=order.quantity,
            ts=ts,
            entry_price=costs.fill_price,
            stop_price=order.stop_price,
            target_price=order.target_price,
            commission=costs.commission,
            slippage=costs.slippage,
            bar_index=0,
        )

        try:
            paper_id = self._persist_open(order=order, ts=ts, fill_price=costs.fill_price)
        except Exception:
            # Roll back the in-memory open to keep portfolio + DB consistent.
            self.portfolio.open_position = None
            raise

        self._open_paper_id = paper_id
        fill = Fill(
            order=order,
            fill_ts=ts,
            fill_price=costs.fill_price,
            commission=costs.commission,
            slippage=costs.slippage,
        )
        self.log.info(
            "paper.opened",
            paper_trade_id=paper_id,
            instrument=order.instrument,
            direction=order.direction,
            quantity=order.quantity,
            entry_price=round(costs.fill_price, 4),
            commission=round(costs.commission, 4),
            slippage=round(costs.slippage, 6),
        )
        return fill

    # ---------------- Position close --------------------------------------
    def close_position(
        self,
        *,
        ts: datetime,
        exit_raw_price: float,
        exit_reason: str,
        bar_index: int = 0,
    ) -> ClosedTradeRecord:
        """Close the open paper position.

        ``exit_raw_price`` is the *pre-slippage* price (e.g. bar close, stop,
        target). The fills model adds adverse slippage as in backtest.
        """
        pos = self.portfolio.open_position
        if pos is None:
            raise RuntimeError("close_position called with no open position.")

        costs = self.fills.exit(
            direction=pos.direction,
            raw_price=exit_raw_price,
            quantity=pos.quantity,
        )
        record = self.portfolio.close(
            ts=ts,
            exit_price=costs.fill_price,
            exit_reason=exit_reason,
            commission=costs.commission,
            slippage=costs.slippage,
            bar_index=bar_index,
        )

        paper_id = self._open_paper_id
        closed_id: Optional[str] = None
        if paper_id is None:
            self.log.warning(
                "paper.close_without_open_id",
                exit_reason=exit_reason,
                detail="Portfolio had open position but no DB paper_trade_id",
            )
        else:
            closed_id = self._persist_close(paper_trade_id=paper_id, record=record)
        self._open_paper_id = None
        self.last_closed_trade_id = closed_id

        self.log.info(
            "paper.closed",
            paper_trade_id=paper_id,
            instrument=record.instrument,
            direction=record.direction,
            exit_reason=exit_reason,
            net_pnl=round(record.net_pnl, 4),
            exit_price=round(record.exit_price, 4),
        )
        return record

    # ---------------- Internal DB writes ----------------------------------
    def _persist_open(self, *, order: Order, ts: datetime, fill_price: float) -> str:
        with session_scope() as session:
            row = PaperTradeRow(
                setup_id=str(order.setup_id) if order.setup_id else None,
                instrument=order.instrument,
                direction=order.direction,
                quantity=float(order.quantity),
                entry_ts=ts,
                entry_price=float(fill_price),
                stop_price=float(order.stop_price),
                target_price=float(order.target_price),
                status="open",
            )
            session.add(row)
            session.flush()
            return str(row.id)

    def _persist_close(
        self, *, paper_trade_id: str, record: ClosedTradeRecord
    ) -> str:
        with session_scope() as session:
            paper = session.execute(
                select(PaperTradeRow).where(PaperTradeRow.id == paper_trade_id)
            ).scalar_one_or_none()
            if paper is not None:
                paper.status = "closed"

            closed = ClosedTradeRow(
                paper_trade_id=paper_trade_id,
                setup_id=record.setup_id or None,
                instrument=record.instrument,
                direction=record.direction,
                quantity=float(record.quantity),
                entry_ts=record.entry_ts,
                entry_price=float(record.entry_price),
                exit_ts=record.exit_ts,
                exit_price=float(record.exit_price),
                exit_reason=record.exit_reason,
                pnl=float(record.net_pnl),
                commission=float(record.commission),
                slippage=float(record.slippage),
            )
            session.add(closed)
            session.flush()  # populate closed.id for the post-trade analysis hook.
            return str(closed.id)
