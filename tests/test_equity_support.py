"""Equity instrument + fills support tests."""

from __future__ import annotations

import pytest

from backtesting.fills import EquityFillsModel, make_fills_model
from config.instruments import get_instrument, register_equity


def test_spy_registered_as_equity() -> None:
    spec = get_instrument("SPY")
    assert spec.market_type == "equity"
    assert spec.tick_size == 0.01
    assert spec.point_value == 1.0


def test_register_equity_on_demand() -> None:
    spec = register_equity("GOOG")
    assert spec.market_type == "equity"
    assert get_instrument("GOOG").symbol == "GOOG"


def test_equity_fills_model_adverse_slippage() -> None:
    fm = make_fills_model(
        "SPY", slippage_ticks=1, commission_per_contract=0.0
    )
    assert isinstance(fm, EquityFillsModel)
    entry = fm.entry(direction="long", raw_price=550.0, quantity=10)
    # 1 cent adverse on a long entry
    assert entry.fill_price == pytest.approx(550.01)
    assert entry.commission == 0.0


def test_equity_fills_model_rejects_futures() -> None:
    with pytest.raises(ValueError):
        EquityFillsModel("MES", slippage_cents=1.0)


def test_equity_fills_short_exit_slippage() -> None:
    fm = EquityFillsModel("AAPL", slippage_cents=2.0)
    ex = fm.exit(direction="short", raw_price=210.0, quantity=5)
    # short exit gets a worse (higher) fill
    assert ex.fill_price == pytest.approx(210.02)
