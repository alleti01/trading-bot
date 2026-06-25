"""Intraday loop safety: daily risk gate, force-flat, dedupe, reconciliation."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from config.settings import reload_settings
from storage.db import init_db
from workflows.intraday_loop import IntradayLoop

_NOW = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)  # weekday, mid-session


def _settings(monkeypatch, tmp_path, **overrides):
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKFLOW_MEMORY_DIR", str(memory))
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("AUTONOMOUS_TRADING_ENABLED", "true")
    monkeypatch.setenv("ENABLED_SYMBOLS", "SPY,QQQ")
    monkeypatch.setenv("WORKFLOW_REFRESH_DATA_EACH_SCAN", "false")
    monkeypatch.setenv("MAX_DAILY_LOSS", "500")
    monkeypatch.setenv("MAX_DAILY_PROFIT", "1500")
    monkeypatch.setenv("MAX_TRADES_PER_DAY", "8")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "4")
    monkeypatch.setenv("MAX_ACTIVE_SYMBOLS", "4")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def _broker(*, day_pnl: float = 0.0, positions=None, orders=None):
    b = MagicMock()
    b.get_account.return_value = SimpleNamespace(realized_pnl=day_pnl)
    b.reconcile.return_value = {
        "positions": positions or [],
        "orders": orders or [],
    }
    b.close_position.return_value = SimpleNamespace(success=True)
    return b


# ---------------------------------------------------------------------------
# Daily risk gate (unit)
# ---------------------------------------------------------------------------
def test_daily_loss_blocks_and_flattens(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    loop._reset_daily_state(_NOW)
    broker = _broker(day_pnl=-600.0)  # below -MAX_DAILY_LOSS (500)
    block, flatten, reason = loop._daily_risk_block(broker, _NOW)
    assert block and flatten
    assert reason.startswith("daily_loss_limit")


def test_daily_profit_blocks_without_flatten(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    loop._reset_daily_state(_NOW)
    broker = _broker(day_pnl=2000.0)  # above MAX_DAILY_PROFIT (1500)
    block, flatten, reason = loop._daily_risk_block(broker, _NOW)
    assert block and not flatten
    assert reason.startswith("daily_profit_target")


def test_max_trades_per_day_blocks(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, MAX_TRADES_PER_DAY="2")
    loop = IntradayLoop(settings, dry_run=False)
    loop._reset_daily_state(_NOW)
    loop._trades_today = 2
    broker = _broker(day_pnl=0.0)
    block, flatten, reason = loop._daily_risk_block(broker, _NOW)
    assert block and not flatten
    assert reason == "max_trades_per_day"


def test_force_flat_eod(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    loop._reset_daily_state(_NOW)
    broker = _broker(day_pnl=0.0)
    with patch("scheduler.market_hours.is_force_flat_due", return_value=True):
        block, flatten, reason = loop._daily_risk_block(broker, _NOW)
    assert block and flatten
    assert reason == "force_flat_eod"


def test_no_block_when_within_limits(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    loop._reset_daily_state(_NOW)
    broker = _broker(day_pnl=-100.0)
    with patch("scheduler.market_hours.is_force_flat_due", return_value=False):
        block, flatten, reason = loop._daily_risk_block(broker, _NOW)
    assert not block and not flatten


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def test_held_or_pending_collects_symbols(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    recon = {
        "positions": [{"symbol": "SPY"}],
        "orders": [{"symbol": "qqq"}],
    }
    assert loop._held_or_pending(recon) == {"SPY", "QQQ"}


def test_flatten_all_closes_positions(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    broker = _broker(positions=[{"symbol": "SPY"}, {"symbol": "QQQ"}])
    closed = loop._flatten_all(broker, broker.reconcile.return_value, reason="test")
    assert closed == 2
    assert broker.close_position.call_count == 2


def test_session_reset_clears_counters(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    loop._reset_daily_state(_NOW)
    loop._trades_today = 5
    loop._halted_today = True
    # Same day → no reset.
    loop._reset_daily_state(_NOW)
    assert loop._trades_today == 5
    # Next day → reset.
    loop._reset_daily_state(datetime(2026, 6, 18, 14, 30, tzinfo=timezone.utc))
    assert loop._trades_today == 0
    assert loop._halted_today is False


# ---------------------------------------------------------------------------
# Integration: scan_once enforces dedupe + daily loss
# ---------------------------------------------------------------------------
def test_scan_skips_already_held_symbol(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    from tests._signal_stub import make_signal

    loop = IntradayLoop(settings, dry_run=False)
    broker = _broker(day_pnl=0.0, positions=[{"symbol": "SPY"}])
    with patch("workflows.order_execution.build_broker", return_value=broker), patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=lambda self, sym: make_signal(sym, price=100.0),
    ):
        summary = loop.scan_once(now=_NOW)
    reasons = {(r["symbol"], r.get("reason")) for r in summary["results"]}
    assert ("SPY", "already_held_or_pending") in reasons


def test_scan_halts_on_daily_loss(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    loop = IntradayLoop(settings, dry_run=False)
    broker = _broker(day_pnl=-700.0, positions=[{"symbol": "SPY"}])
    with patch("workflows.order_execution.build_broker", return_value=broker):
        summary = loop.scan_once(now=_NOW)
    assert summary.get("risk_block", "").startswith("daily_loss")
    assert loop._halted_today is True
    broker.close_position.assert_called()  # flattened
