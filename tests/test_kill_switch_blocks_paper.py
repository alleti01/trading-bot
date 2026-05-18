"""Kill switch: tripped state must block paper trading at every layer."""

from __future__ import annotations

import os

import pytest

from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from config.instruments import get_instrument
from execution.base import Order
from execution.paper_executor import KillSwitchActive, PaperExecutor
from risk.kill_switch import KillSwitch
from storage.db import init_db


def _executor():
    init_db()
    spec = get_instrument("MES")
    portfolio = Portfolio(instrument_spec=spec)
    fills = make_fills_model("MES", slippage_ticks=0.0, commission_per_contract=0.0)
    return PaperExecutor(portfolio=portfolio, fills_model=fills), portfolio


def test_executor_refuses_when_tripped() -> None:
    executor, _ = _executor()
    KillSwitch().trip("integration_test")
    order = Order(
        instrument="MES", direction="long", quantity=1.0,
        entry_price=4500.0, stop_price=4498.0, target_price=4504.0,
        setup_id="abc",
    )
    with pytest.raises(KillSwitchActive):
        executor.submit(order)


def test_state_persists_across_kill_switch_instances() -> None:
    init_db()
    KillSwitch().trip("persists")
    # A fresh KillSwitch reads from the same DB row.
    other = KillSwitch()
    assert other.is_tripped() is True
    other.reset_manual()
    assert KillSwitch().is_tripped() is False
