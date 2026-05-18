"""Risk engine rule matrix.

Each rule gets at least one block case and one allow case so future
refactors can't silently flip a rule to a no-op.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

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
        cooldown_after_loss_minutes=5,
        cooldown_after_large_win_minutes=15,
        large_win_threshold=200.0,
        market_type="futures",
    )
    base.update(overrides)
    return RiskConfig(**base)


def _portfolio() -> Portfolio:
    return Portfolio(instrument_spec=get_instrument("MES"), starting_equity=0.0)


def _setup(direction: str = "long") -> Setup:
    if direction == "long":
        return Setup(
            instrument="MES",
            timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
            strategy_name="test",
            direction="long",
            entry_price=100.0, stop_price=99.0, target_price=102.0,
            atr_at_entry=1.0,
            features={c: 0.0 for c in FEATURE_COLUMNS}, bar_index=0,
        )
    return Setup(
        instrument="MES",
        timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        strategy_name="test",
        direction="short",
        entry_price=100.0, stop_price=101.0, target_price=98.0,
        atr_at_entry=1.0,
        features={c: 0.0 for c in FEATURE_COLUMNS}, bar_index=0,
    )


# ---------------------------------------------------------------------------
# Allow case
# ---------------------------------------------------------------------------
def test_clean_allow_case() -> None:
    cfg = _config()
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    d = evaluate(_setup(), p, cfg, now)
    assert d.allowed is True


# ---------------------------------------------------------------------------
# Each rule blocks
# ---------------------------------------------------------------------------
def test_kill_switch_blocks() -> None:
    cfg = _config()
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    d = evaluate(_setup(), p, cfg, now, kill_switch_tripped=True)
    assert d.allowed is False
    assert d.rule == "kill_switch"


def test_high_risk_news_blocks() -> None:
    cfg = _config()
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    d = evaluate(_setup(), p, cfg, now, high_risk_news_window=True)
    assert d.allowed is False
    assert d.rule == "high_risk_news"


def test_outside_trading_window_blocks() -> None:
    cfg = _config()
    p = _portfolio()
    pre_open = datetime(2024, 1, 15, 6, 0, tzinfo=timezone.utc)
    d = evaluate(_setup(), p, cfg, pre_open)
    assert d.allowed is False
    assert d.rule == "trading_window"


def test_after_force_flat_blocks() -> None:
    cfg = _config(force_flat_time=time(14, 0))
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    d = evaluate(_setup(), p, cfg, now)
    assert d.allowed is False
    assert d.rule == "forced_flat"


def test_no_hedging_blocks_opposite_direction() -> None:
    cfg = _config()
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="open", instrument="MES", direction="long", quantity=1,
        ts=now, entry_price=100, stop_price=99, target_price=101,
        commission=1.5, slippage=0.0, bar_index=0,
    )
    d = evaluate(_setup("short"), p, cfg, now + timedelta(minutes=1))
    assert d.allowed is False
    assert d.rule in {"max_open_positions", "no_hedging"}


def test_max_open_positions_blocks() -> None:
    cfg = _config(max_open_positions=1)
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.open(
        setup_id="open", instrument="MES", direction="long", quantity=1,
        ts=now, entry_price=100, stop_price=99, target_price=101,
        commission=1.5, slippage=0.0, bar_index=0,
    )
    d = evaluate(_setup("long"), p, cfg, now + timedelta(minutes=1))
    assert d.allowed is False
    assert d.rule == "max_open_positions"


def test_max_trades_per_day_blocks() -> None:
    cfg = _config(max_trades_per_day=2)
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.day.day = now.date()
    p.day.trades = 2
    d = evaluate(_setup(), p, cfg, now)
    assert d.allowed is False
    assert d.rule == "max_trades_per_day"


def test_max_daily_loss_blocks() -> None:
    cfg = _config(max_daily_loss=100)
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.day.day = now.date()
    p.day.net_pnl = -101
    d = evaluate(_setup(), p, cfg, now)
    assert d.allowed is False
    assert d.rule == "max_daily_loss"


def test_max_daily_profit_blocks() -> None:
    cfg = _config(max_daily_profit=200)
    p = _portfolio()
    now = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.day.day = now.date()
    p.day.net_pnl = 201
    d = evaluate(_setup(), p, cfg, now)
    assert d.allowed is False
    assert d.rule == "max_daily_profit"


def test_cooldown_after_loss_blocks() -> None:
    cfg = _config(cooldown_after_loss_minutes=5)
    p = _portfolio()
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.day.day = base.date()
    p.day.last_trade_ts = base
    p.day.last_trade_net_pnl = -50
    d = evaluate(_setup(), p, cfg, base + timedelta(minutes=2))
    assert d.allowed is False
    assert d.rule == "cooldown_after_loss"


def test_cooldown_after_loss_expires() -> None:
    cfg = _config(cooldown_after_loss_minutes=5)
    p = _portfolio()
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.day.day = base.date()
    p.day.last_trade_ts = base
    p.day.last_trade_net_pnl = -50
    d = evaluate(_setup(), p, cfg, base + timedelta(minutes=10))
    assert d.allowed is True


def test_cooldown_after_large_win_blocks() -> None:
    cfg = _config(cooldown_after_large_win_minutes=15, large_win_threshold=200)
    p = _portfolio()
    base = datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc)
    p.day.day = base.date()
    p.day.last_trade_ts = base
    p.day.last_trade_net_pnl = 300
    d = evaluate(_setup(), p, cfg, base + timedelta(minutes=5))
    assert d.allowed is False
    assert d.rule == "cooldown_after_large_win"
