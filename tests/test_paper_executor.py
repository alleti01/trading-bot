"""PaperExecutor: persists open + close, applies fill costs, refuses on kill switch."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from config.instruments import get_instrument
from execution.base import Order
from execution.paper_executor import KillSwitchActive, PaperExecutor
from risk.kill_switch import KillSwitch
from storage.db import init_db, session_scope
from storage.tables import ClosedTrade, PaperTrade


NY = ZoneInfo("America/New_York")


def _make_executor(*, slippage_ticks: float = 1.0):
    init_db()
    spec = get_instrument("MES")
    portfolio = Portfolio(instrument_spec=spec)
    fills = make_fills_model(
        "MES", slippage_ticks=slippage_ticks, commission_per_contract=0.50
    )
    return PaperExecutor(portfolio=portfolio, fills_model=fills), portfolio


def _order(direction: str = "long", *, entry: float = 4500.0) -> Order:
    return Order(
        instrument="MES",
        direction=direction,
        quantity=1.0,
        entry_price=entry,
        stop_price=entry - 2.0 if direction == "long" else entry + 2.0,
        target_price=entry + 4.0 if direction == "long" else entry - 4.0,
        setup_id="setup-test-1",
    )


def test_submit_persists_paper_trade_and_applies_slippage() -> None:
    executor, portfolio = _make_executor(slippage_ticks=2.0)
    fill = executor.submit(_order(entry=4500.0))

    spec = get_instrument("MES")
    expected_slip = 2.0 * spec.tick_size
    assert fill.fill_price == pytest.approx(4500.0 + expected_slip)
    assert fill.commission == pytest.approx(0.50)
    assert portfolio.open_position is not None

    with session_scope() as session:
        rows = session.execute(select(PaperTrade)).scalars().all()
    assert len(rows) == 1
    assert rows[0].status == "open"
    assert rows[0].direction == "long"
    assert rows[0].entry_price == pytest.approx(fill.fill_price)


def test_close_persists_closed_trade_and_marks_paper_closed() -> None:
    executor, portfolio = _make_executor(slippage_ticks=0.0)
    executor.submit(_order(entry=4500.0))

    record = executor.close_position(
        ts=datetime(2024, 1, 15, 10, 5, tzinfo=NY),
        exit_raw_price=4504.0,
        exit_reason="tp",
    )
    assert record.exit_reason == "tp"
    assert portfolio.open_position is None

    with session_scope() as session:
        paper_rows = session.execute(select(PaperTrade)).scalars().all()
        closed_rows = session.execute(select(ClosedTrade)).scalars().all()

    assert paper_rows[0].status == "closed"
    assert len(closed_rows) == 1
    assert closed_rows[0].exit_reason == "tp"
    assert closed_rows[0].pnl != 0  # MES point value $5 → +$20 gross


def test_submit_refuses_when_kill_switch_tripped() -> None:
    executor, _ = _make_executor()
    KillSwitch().trip("test")
    with pytest.raises(KillSwitchActive):
        executor.submit(_order())


def test_submit_refuses_with_open_position() -> None:
    executor, _ = _make_executor()
    executor.submit(_order(entry=4500))
    with pytest.raises(RuntimeError, match="another position is open"):
        executor.submit(_order(entry=4501))


def test_close_without_open_position_raises() -> None:
    executor, _ = _make_executor()
    with pytest.raises(RuntimeError, match="no open position"):
        executor.close_position(
            ts=datetime(2024, 1, 15, 10, 5, tzinfo=NY),
            exit_raw_price=4500.0,
            exit_reason="manual",
        )
