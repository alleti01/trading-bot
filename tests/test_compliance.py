"""Compliance + Tradeify rule checks on synthetic trade ledgers."""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

from backtesting.portfolio import ClosedTradeRecord
from compliance.rules import hft_profit_fraction, hft_trade_fraction
from compliance.tradeify_rules import (
    consistency_rule,
    forced_flat_compliance,
    no_overnight_futures,
)


def _trade(
    *,
    entry: datetime,
    hold_seconds: float,
    net_pnl: float,
    direction: str = "long",
) -> ClosedTradeRecord:
    return ClosedTradeRecord(
        setup_id="s",
        instrument="MES",
        direction=direction,
        quantity=1.0,
        entry_ts=entry,
        entry_price=4500.0,
        exit_ts=entry + timedelta(seconds=hold_seconds),
        exit_price=4501.0,
        exit_reason="tp" if net_pnl > 0 else "sl",
        gross_pnl=net_pnl + 1.5,
        commission=1.5,
        slippage=0.0,
        net_pnl=net_pnl,
        bars_held=int(hold_seconds // 60),
    )


# ---------------------------------------------------------------------------
# HFT
# ---------------------------------------------------------------------------
def test_hft_trade_fraction_flags_majority_fast_trades() -> None:
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    trades = [
        _trade(entry=base, hold_seconds=2, net_pnl=10),
        _trade(entry=base, hold_seconds=3, net_pnl=10),
        _trade(entry=base, hold_seconds=4, net_pnl=10),
        _trade(entry=base, hold_seconds=120, net_pnl=10),
    ]
    flag = hft_trade_fraction(trades)
    assert flag.triggered is True
    assert flag.value > 0.5


def test_hft_trade_fraction_clean() -> None:
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    trades = [_trade(entry=base, hold_seconds=120, net_pnl=10) for _ in range(5)]
    flag = hft_trade_fraction(trades)
    assert flag.triggered is False


def test_hft_profit_fraction_flags_when_majority_profit_from_fast_wins() -> None:
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    trades = [
        _trade(entry=base, hold_seconds=2, net_pnl=80),    # fast win
        _trade(entry=base, hold_seconds=3, net_pnl=20),    # fast win
        _trade(entry=base, hold_seconds=300, net_pnl=10),  # slow win
    ]
    flag = hft_profit_fraction(trades)
    assert flag.triggered is True


# ---------------------------------------------------------------------------
# Consistency
# ---------------------------------------------------------------------------
def test_consistency_rule_flags_when_one_day_dominates() -> None:
    # Day A: $400 profit. Day B: $50 profit. Limit = 30% → Day A is 88.8% → flag.
    day_a = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    day_b = datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc)
    trades = [
        _trade(entry=day_a, hold_seconds=120, net_pnl=400),
        _trade(entry=day_b, hold_seconds=120, net_pnl=50),
    ]
    flag = consistency_rule(trades, limit_percent=30.0)
    assert flag.triggered is True


def test_consistency_rule_does_not_flag_when_total_is_negative() -> None:
    day = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    trades = [_trade(entry=day, hold_seconds=120, net_pnl=-100)]
    flag = consistency_rule(trades, limit_percent=30.0)
    assert flag.triggered is False


# ---------------------------------------------------------------------------
# Overnight + forced flat
# ---------------------------------------------------------------------------
def test_no_overnight_futures_flags_cross_day_trade() -> None:
    entry = datetime(2024, 1, 15, 23, 50, tzinfo=timezone.utc)
    bad = ClosedTradeRecord(
        setup_id="s", instrument="MES", direction="long", quantity=1,
        entry_ts=entry, entry_price=4500.0,
        exit_ts=entry + timedelta(hours=2), exit_price=4501.0,
        exit_reason="tp", gross_pnl=5, commission=1.5, slippage=0,
        net_pnl=3.5, bars_held=120,
    )
    flag = no_overnight_futures(
        [bad], market_type="futures", timezone="UTC", session_close=time(16, 0),
    )
    assert flag.triggered is True


def test_forced_flat_compliance_flags_post_flat_entries() -> None:
    entry_ok = datetime(2024, 1, 15, 18, 0, tzinfo=timezone.utc)        # 13:00 NY
    entry_bad = datetime(2024, 1, 15, 21, 0, tzinfo=timezone.utc)       # 16:00 NY
    trades = [
        _trade(entry=entry_ok, hold_seconds=60, net_pnl=10),
        _trade(entry=entry_bad, hold_seconds=60, net_pnl=10),
    ]
    flag = forced_flat_compliance(
        trades,
        market_type="futures",
        timezone="America/New_York",
        flat_time=time(15, 55),
    )
    assert flag.triggered is True
    assert flag.value == 1.0
