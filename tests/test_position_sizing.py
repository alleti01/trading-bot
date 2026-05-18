"""Position sizing math."""

from __future__ import annotations

import pytest

from risk.position_sizing import size_position


def test_size_capped_by_max_position() -> None:
    # MES point_value=$5; stop 1pt away → $5/contract risk; risk_per_trade $50
    # → naive = 10, cap=2 → expect 2.
    out = size_position(
        entry_price=4500, stop_price=4499,
        instrument="MES", risk_per_trade=50, max_position_size=2,
    )
    assert out.quantity == 2
    assert out.risk_per_contract_dollars == pytest.approx(5.0)
    assert out.expected_risk_dollars == pytest.approx(10.0)


def test_size_floored_at_one_when_distance_too_wide() -> None:
    # Stop 100pt away; per-contract risk = $500; risk_per_trade = $100 → naive 0,
    # floored to 1 with exceeds_risk_per_trade=True.
    out = size_position(
        entry_price=4500, stop_price=4400,
        instrument="MES", risk_per_trade=100, max_position_size=5,
    )
    assert out.quantity == 1
    assert out.exceeds_risk_per_trade is True


def test_rejects_zero_distance() -> None:
    with pytest.raises(ValueError):
        size_position(
            entry_price=4500, stop_price=4500,
            instrument="MES", risk_per_trade=100, max_position_size=2,
        )


def test_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        size_position(
            entry_price=4500, stop_price=4499,
            instrument="MES", risk_per_trade=0, max_position_size=2,
        )
    with pytest.raises(ValueError):
        size_position(
            entry_price=4500, stop_price=4499,
            instrument="MES", risk_per_trade=100, max_position_size=0,
        )
