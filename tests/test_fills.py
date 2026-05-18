"""Slippage + commission models."""

from __future__ import annotations

import pytest

from backtesting.fills import (
    CryptoFillsModel,
    FuturesFillsModel,
    make_fills_model,
)
from config.instruments import get_instrument


def test_futures_long_entry_pays_higher_than_raw() -> None:
    fm = FuturesFillsModel(get_instrument("MES"), slippage_ticks=1, commission_per_contract=1.50)
    out = fm.entry(direction="long", raw_price=4500.00, quantity=2)
    # 1 tick of MES = 0.25; long entry is adverse → pays more.
    assert out.fill_price == 4500.25
    assert out.slippage == 0.25
    assert out.commission == 3.00


def test_futures_short_entry_receives_lower_than_raw() -> None:
    fm = FuturesFillsModel(get_instrument("MES"), slippage_ticks=2, commission_per_contract=1.50)
    out = fm.entry(direction="short", raw_price=4500.00, quantity=1)
    assert out.fill_price == 4499.50
    assert out.slippage == 0.50
    assert out.commission == 1.50


def test_futures_exit_mirrors_entry_direction() -> None:
    fm = FuturesFillsModel(get_instrument("MES"), slippage_ticks=1, commission_per_contract=1.50)
    long_exit = fm.exit(direction="long", raw_price=4500.00, quantity=1)
    short_exit = fm.exit(direction="short", raw_price=4500.00, quantity=1)
    # Long exit gets less; short exit pays more.
    assert long_exit.fill_price == 4499.75
    assert short_exit.fill_price == 4500.25


def test_futures_rejects_unknown_direction() -> None:
    fm = FuturesFillsModel(get_instrument("MES"), slippage_ticks=1, commission_per_contract=1.0)
    with pytest.raises(ValueError):
        fm.entry(direction="sideways", raw_price=4500, quantity=1)  # type: ignore[arg-type]


def test_futures_rejects_negative_inputs() -> None:
    with pytest.raises(ValueError):
        FuturesFillsModel(get_instrument("MES"), slippage_ticks=-1, commission_per_contract=0)
    with pytest.raises(ValueError):
        FuturesFillsModel(get_instrument("MES"), slippage_ticks=0, commission_per_contract=-1)


def test_crypto_fills_use_bps_slippage() -> None:
    fm = CryptoFillsModel(get_instrument("BTC"), slippage_bps=10, fee_bps=5)
    out = fm.entry(direction="long", raw_price=10_000, quantity=0.1)
    assert out.fill_price == pytest.approx(10_010.0)
    assert out.slippage == pytest.approx(10.0)
    # Commission = qty * fill_price * 5bps.
    assert out.commission == pytest.approx(0.1 * 10_010.0 * 0.0005)


def test_make_fills_model_returns_correct_type() -> None:
    fm = make_fills_model("MES", slippage_ticks=1, commission_per_contract=1.5)
    assert isinstance(fm, FuturesFillsModel)
    fm2 = make_fills_model("BTC", slippage_ticks=0, commission_per_contract=0)
    assert isinstance(fm2, CryptoFillsModel)


def test_futures_model_rejects_crypto_instrument() -> None:
    with pytest.raises(ValueError, match="non-futures"):
        FuturesFillsModel(get_instrument("BTC"), slippage_ticks=1, commission_per_contract=1.0)
