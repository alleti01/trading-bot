"""Workflow ↔ integrations broker wiring tests."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from config.settings import reload_settings
from integrations import MockBroker
from integrations.broker_base import OrderResult
from storage.db import init_db
from workflows.workflow_runner import WorkflowRunner


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path, **overrides: str):
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKFLOW_MEMORY_DIR", str(memory))
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("AUTONOMOUS_TRADING_ENABLED", "true")
    monkeypatch.setenv("ENABLED_SYMBOLS", "MES")
    monkeypatch.setenv("WORKFLOW_WEEKDAYS_ONLY", "false")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_workflow_paper_builds_order_broker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    ctx = runner._build_context(
        now=datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc),
        force=True,
    )
    assert ctx.order_broker is not None
    assert isinstance(ctx.order_broker, MockBroker)


def test_workflow_dry_run_has_no_order_broker(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    ctx = runner._build_context(
        now=datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc),
        force=True,
    )
    assert ctx.order_broker is None


def test_market_open_uses_integration_broker_for_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    now = datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)
    runner.run("premarket", now=now)

    mock_broker = MagicMock(spec=MockBroker)
    mock_broker.reconcile.return_value = {"open_positions": 0, "open_orders": 0}
    mock_broker.validate_order.return_value = MagicMock(valid=True, quote=None)
    # The workflow now places a single native bracket order (entry +
    # attached protective stop/target) instead of separate legs.
    mock_broker.place_bracket_order.return_value = OrderResult(
        success=True,
        simulated=True,
        order_id="br1",
        symbol="MES",
        side="buy",
        quantity=1.0,
        order_type="limit",
        status="working",
        stop_price=4990.0,
    )

    from tests._signal_stub import stub_market_open_signal

    with patch(
        "workflows.workflow_runner.build_broker", return_value=mock_broker
    ), stub_market_open_signal(price=5000.0):
        result = runner.run("market-open", now=now)

    assert result.success
    assert result.payload.get("actions_taken", 0) >= 1
    mock_broker.reconcile.assert_called()
    mock_broker.place_bracket_order.assert_called()
