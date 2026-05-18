"""Per-trade risk cap is enforced before sizing.

The position sizer floors qty at 1 even when the geometric stop is so wide
that a single contract risks more than ``risk_per_trade``. Without this
risk-engine rule, a strategy with an oversized stop would silently bypass
the per-trade dollar cap. We reject those setups up front instead.
"""

from __future__ import annotations

from datetime import datetime, time, timezone

from backtesting.portfolio import Portfolio
from config.instruments import get_instrument
from features.feature_builder import FEATURE_COLUMNS
from risk.risk_engine import RiskConfig, evaluate
from strategies.base import Setup


def _config(**overrides) -> RiskConfig:
    base = dict(
        timezone="UTC",
        trading_window_start=time(9, 30),
        trading_window_end=time(15, 55),
        force_flat_time=time(15, 55),
        max_trades_per_day=8,
        max_daily_loss=500.0,
        max_daily_profit=1500.0,
        max_open_positions=1,
        cooldown_after_loss_minutes=0,
        cooldown_after_large_win_minutes=0,
        large_win_threshold=200.0,
        market_type="futures",
        risk_per_trade=100.0,
    )
    base.update(overrides)
    return RiskConfig(**base)


def _setup(*, entry: float, stop: float, direction: str = "long") -> Setup:
    return Setup(
        instrument="MES",
        timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        strategy_name="test",
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=entry + (entry - stop) if direction == "long" else entry - (stop - entry),
        atr_at_entry=1.0,
        features={c: 0.0 for c in FEATURE_COLUMNS},
        bar_index=0,
    )


def test_blocks_when_single_contract_risk_exceeds_per_trade_cap() -> None:
    cfg = _config(risk_per_trade=50.0)
    p = Portfolio(instrument_spec=get_instrument("MES"))
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

    # MES point_value=$5; stop 20pt away → $100/contract risk; cap=$50.
    decision = evaluate(_setup(entry=4500, stop=4480), p, cfg, now)
    assert decision.allowed is False
    assert decision.rule == "risk_per_trade_exceeded"


def test_allows_when_risk_within_cap() -> None:
    cfg = _config(risk_per_trade=100.0)
    p = Portfolio(instrument_spec=get_instrument("MES"))
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)

    # 2pt stop * $5/pt = $10 risk per contract; well under $100.
    decision = evaluate(_setup(entry=4500, stop=4498), p, cfg, now)
    assert decision.allowed is True


def test_disabled_when_risk_per_trade_zero() -> None:
    """A zero (or negative) ``risk_per_trade`` disables the rule for tests
    that don't want to thread the cap through every fixture."""
    cfg = _config(risk_per_trade=0.0)
    p = Portfolio(instrument_spec=get_instrument("MES"))
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    decision = evaluate(_setup(entry=4500, stop=4400), p, cfg, now)
    assert decision.allowed is True


def test_unknown_instrument_skips_rule() -> None:
    """If the symbol isn't in the registry we cannot evaluate the cap;
    we still allow the trade so test fixtures with synthetic instruments
    don't get spuriously blocked. The check still fires for known symbols."""
    cfg = _config(risk_per_trade=1.0)  # tiny cap on purpose
    p = Portfolio(instrument_spec=get_instrument("MES"))
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    s = _setup(entry=4500, stop=4490)
    s = Setup(  # rebuild with synthetic instrument
        instrument="UNKNOWN_INST",
        timestamp=s.timestamp,
        strategy_name=s.strategy_name,
        direction=s.direction,
        entry_price=s.entry_price,
        stop_price=s.stop_price,
        target_price=s.target_price,
        atr_at_entry=s.atr_at_entry,
        features=s.features,
        bar_index=s.bar_index,
    )
    decision = evaluate(s, p, cfg, now)
    assert decision.allowed is True
