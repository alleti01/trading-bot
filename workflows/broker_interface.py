"""Broker state interface for workflows (paper DB-backed MVP)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select

from backtesting.fills import FillsModel, make_fills_model
from backtesting.portfolio import OpenPosition, Portfolio
from config.instruments import get_instrument
from config.settings import Settings
from execution.base import Order
from execution.paper_executor import KillSwitchActive, PaperExecutor
from risk.kill_switch import KillSwitch
from scheduler.market_hours import session_date
from storage.db import session_scope
from storage.tables import ClosedTrade as ClosedTradeRow
from storage.tables import PaperTrade as PaperTradeRow
from workflows.schemas import (
    AccountSnapshot,
    BrokerState,
    OrderSnapshot,
    PositionSnapshot,
)


class BrokerInterface(ABC):
    @abstractmethod
    def pull_state(
        self, *, now: datetime, quotes: Optional[dict[str, float]] = None
    ) -> BrokerState:
        ...


class PaperBrokerInterface(BrokerInterface):
    """Reads open paper trades + closed trades from SQLite."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def pull_state(
        self, *, now: datetime, quotes: Optional[dict[str, float]] = None
    ) -> BrokerState:
        sd = session_date(now, self.settings).isoformat()
        quotes = quotes or {}
        tz = self.settings.workflow_tz()
        window = _session_window(now, self.settings, tz)
        day_pnl = 0.0
        cum_pnl = 0.0
        trades_today = 0
        positions: list[PositionSnapshot] = []

        with session_scope() as session:
            closed_rows = session.execute(select(ClosedTradeRow)).scalars().all()
            for row in closed_rows:
                cum_pnl += float(row.pnl)
                if window[0] <= row.exit_ts.astimezone(tz) <= window[1]:
                    day_pnl += float(row.pnl)
                    trades_today += 1

            open_rows = session.execute(
                select(PaperTradeRow).where(PaperTradeRow.status == "open")
            ).scalars().all()
            for row in open_rows:
                spec = get_instrument(row.instrument)
                current = quotes.get(row.instrument.upper())
                unrealized = None
                unrealized_pct = None
                if current is not None:
                    unrealized = _unrealized_dollars(
                        direction=row.direction,
                        entry=float(row.entry_price),
                        current=float(current),
                        quantity=float(row.quantity),
                        point_value=spec.point_value,
                    )
                    if row.entry_price:
                        if row.direction == "long":
                            unrealized_pct = (
                                (current - row.entry_price) / row.entry_price
                            ) * 100.0
                        else:
                            unrealized_pct = (
                                (row.entry_price - current) / row.entry_price
                            ) * 100.0
                positions.append(
                    PositionSnapshot(
                        paper_trade_id=str(row.id),
                        instrument=row.instrument,
                        direction=row.direction,  # type: ignore[arg-type]
                        quantity=float(row.quantity),
                        entry_price=float(row.entry_price),
                        stop_price=float(row.stop_price),
                        target_price=float(row.target_price),
                        current_price=current,
                        unrealized_pnl=unrealized,
                        unrealized_pnl_pct=unrealized_pct,
                    )
                )

        account = AccountSnapshot(
            as_of=now,
            session_date=sd,
            day_pnl=round(day_pnl, 2),
            cumulative_pnl=round(cum_pnl, 2),
            trades_today=trades_today,
            open_positions=len(positions),
            open_orders=0,
            equity_estimate=round(cum_pnl, 2),
        )
        return BrokerState(account=account, positions=positions, orders=[])


def build_paper_executor(
    settings: Settings,
    *,
    kill_switch: Optional[KillSwitch] = None,
) -> PaperExecutor:
    spec = get_instrument(settings.INSTRUMENT)
    portfolio = Portfolio(instrument_spec=spec)
    fills: FillsModel = make_fills_model(
        spec.symbol,
        slippage_ticks=settings.SLIPPAGE_TICKS,
        commission_per_contract=settings.COMMISSION_PER_CONTRACT,
        crypto_slippage_bps=settings.CRYPTO_SLIPPAGE_BPS,
        crypto_fee_bps=settings.CRYPTO_FEE_BPS,
    )
    executor = PaperExecutor(
        portfolio=portfolio,
        fills_model=fills,
        kill_switch=kill_switch or KillSwitch(),
    )
    _hydrate_open_position(executor)
    return executor


def _hydrate_open_position(executor: PaperExecutor) -> None:
    with session_scope() as session:
        row = session.execute(
            select(PaperTradeRow).where(PaperTradeRow.status == "open")
        ).scalars().first()
        if row is None:
            return
        ts = row.entry_ts
        executor.portfolio.open(
            setup_id=str(row.setup_id or "workflow"),
            instrument=row.instrument,
            direction=row.direction,
            quantity=float(row.quantity),
            ts=ts,
            entry_price=float(row.entry_price),
            stop_price=float(row.stop_price),
            target_price=float(row.target_price),
            commission=0.0,
            slippage=0.0,
            bar_index=0,
        )
        executor._open_paper_id = str(row.id)  # noqa: SLF001 — workflow bridge


def submit_paper_order(
    executor: PaperExecutor,
    *,
    order: Order,
) -> bool:
    try:
        executor.submit(order)
        return True
    except (KillSwitchActive, RuntimeError):
        return False


def close_paper_position(
    executor: PaperExecutor,
    *,
    now: datetime,
    exit_price: float,
    reason: str,
) -> bool:
    if executor.portfolio.open_position is None:
        return False
    try:
        executor.close_position(
            ts=now,
            exit_raw_price=exit_price,
            exit_reason=reason,
        )
        return True
    except RuntimeError:
        return False


def _session_window(
    now: datetime, settings: Settings, tz: ZoneInfo
) -> tuple[datetime, datetime]:
    local = now.astimezone(tz)
    start = datetime.combine(
        local.date(),
        settings.trading_window_start_time(),
        tzinfo=tz,
    )
    end = datetime.combine(
        local.date(),
        settings.trading_window_end_time(),
        tzinfo=tz,
    )
    if end < start:
        end += timedelta(days=1)
    return start, end


def _unrealized_dollars(
    *,
    direction: str,
    entry: float,
    current: float,
    quantity: float,
    point_value: float,
) -> float:
    if direction == "long":
        return (current - entry) * point_value * quantity
    return (entry - current) * point_value * quantity
