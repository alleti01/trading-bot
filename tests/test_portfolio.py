"""Portfolio state, PnL math, day rollover."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from backtesting.portfolio import Portfolio
from config.instruments import get_instrument


def _portfolio() -> Portfolio:
    return Portfolio(instrument_spec=get_instrument("MES"), starting_equity=0.0)


def test_long_round_trip_pnl_uses_point_value() -> None:
    """MES point_value=$5: 4500 → 4502 long w/ 1 contract = $10 gross."""
    p = _portfolio()
    t0 = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="s1", instrument="MES", direction="long", quantity=1,
        ts=t0, entry_price=4500.0, stop_price=4495.0, target_price=4510.0,
        commission=1.5, slippage=0.0, bar_index=0,
    )
    closed = p.close(
        ts=t0 + timedelta(minutes=5), exit_price=4502.0,
        exit_reason="tp", commission=1.5, slippage=0.0, bar_index=5,
    )
    assert closed.gross_pnl == pytest.approx(10.0)
    # net = 10 - (1.5 + 1.5) = 7
    assert closed.net_pnl == pytest.approx(7.0)
    assert closed.bars_held == 5
    assert p.is_flat()
    assert p.equity == pytest.approx(7.0)


def test_short_round_trip_pnl_negation() -> None:
    p = _portfolio()
    t0 = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="s2", instrument="MES", direction="short", quantity=2,
        ts=t0, entry_price=4500.0, stop_price=4505.0, target_price=4490.0,
        commission=3.0, slippage=0.0, bar_index=0,
    )
    closed = p.close(
        ts=t0 + timedelta(minutes=3), exit_price=4498.0,
        exit_reason="tp", commission=3.0, slippage=0.0, bar_index=3,
    )
    # Short: (entry - exit) * 5 * 2 = (4500-4498)*5*2 = 20
    assert closed.gross_pnl == pytest.approx(20.0)
    assert closed.net_pnl == pytest.approx(20.0 - 6.0)


def test_cannot_open_two_positions() -> None:
    p = _portfolio()
    t0 = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="s1", instrument="MES", direction="long", quantity=1,
        ts=t0, entry_price=4500, stop_price=4495, target_price=4510,
        commission=1.5, slippage=0.0, bar_index=0,
    )
    with pytest.raises(RuntimeError, match="while"):
        p.open(
            setup_id="s2", instrument="MES", direction="short", quantity=1,
            ts=t0, entry_price=4500, stop_price=4505, target_price=4490,
            commission=1.5, slippage=0.0, bar_index=0,
        )


def test_day_rolls_over() -> None:
    p = _portfolio()
    t0 = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.maybe_roll_day(t0)
    p.day.trades = 5
    rolled = p.maybe_roll_day(t0 + timedelta(days=1))
    assert rolled is True
    assert p.day.trades == 0


def test_close_updates_day_state() -> None:
    p = _portfolio()
    t0 = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="s1", instrument="MES", direction="long", quantity=1,
        ts=t0, entry_price=4500, stop_price=4495, target_price=4510,
        commission=1.5, slippage=0.0, bar_index=0,
    )
    p.close(
        ts=t0 + timedelta(minutes=2), exit_price=4505,
        exit_reason="tp", commission=1.5, slippage=0.0, bar_index=2,
    )
    assert p.day.trades == 1
    assert p.day.wins == 1
    assert p.day.losses == 0
    assert p.day.last_trade_net_pnl == pytest.approx((25.0) - 3.0)


def test_close_without_open_raises() -> None:
    p = _portfolio()
    with pytest.raises(RuntimeError, match="no open position"):
        p.close(
            ts=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
            exit_price=4500, exit_reason="tp",
            commission=1.5, slippage=0, bar_index=0,
        )


def test_equity_curve_tracks_close_events() -> None:
    p = _portfolio()
    t0 = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="s1", instrument="MES", direction="long", quantity=1,
        ts=t0, entry_price=4500, stop_price=4495, target_price=4510,
        commission=1.5, slippage=0.0, bar_index=0,
    )
    p.close(
        ts=t0 + timedelta(minutes=1), exit_price=4505,
        exit_reason="tp", commission=1.5, slippage=0.0, bar_index=1,
    )
    p.open(
        setup_id="s2", instrument="MES", direction="short", quantity=1,
        ts=t0 + timedelta(minutes=2), entry_price=4500, stop_price=4505, target_price=4490,
        commission=1.5, slippage=0.0, bar_index=2,
    )
    p.close(
        ts=t0 + timedelta(minutes=3), exit_price=4505,
        exit_reason="sl", commission=1.5, slippage=0.0, bar_index=3,
    )
    assert len(p.equity_curve) == 2
    assert p.equity_curve[0][1] != p.equity_curve[1][1]
