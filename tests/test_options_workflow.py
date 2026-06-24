"""Options ↔ workflow wiring tests (market-open routing + midday manage)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import reload_settings
from storage.db import init_db
from workflows.workflow_runner import WorkflowRunner

_NOW = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)  # weekday


def _settings(monkeypatch, tmp_path, **overrides):
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKFLOW_MEMORY_DIR", str(memory))
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("AUTONOMOUS_TRADING_ENABLED", "true")
    monkeypatch.setenv("WORKFLOW_WEEKDAYS_ONLY", "false")
    monkeypatch.setenv("INSTRUMENT", "SPY")
    monkeypatch.setenv("MARKET_TYPE", "equity")
    monkeypatch.setenv("ENABLED_SYMBOLS", "SPY")
    monkeypatch.setenv("MAX_ACTIVE_SYMBOLS", "1")
    monkeypatch.setenv("OPTIONS_ENABLED", "true")
    monkeypatch.setenv("OPTIONS_ENABLED_UNDERLYINGS", "SPY,QQQ")
    monkeypatch.setenv("OPTIONS_MAX_PREMIUM_PER_TRADE", "100000")
    monkeypatch.setenv("OPTIONS_STATE_PATH", str(tmp_path / "options.json"))
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_market_open_routes_to_options_when_enabled(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    from tests._signal_stub import stub_market_open_signal

    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    runner.run("premarket", now=_NOW)
    with stub_market_open_signal(price=550.0):
        result = runner.run("market-open", now=_NOW)
    assert result.success
    decisions = result.payload.get("decisions", [])
    # At least one decision should reference the options strategy path.
    assert any(
        "Options" in (d.get("reason") or "") for d in decisions
    ), decisions


def test_market_open_equity_when_options_disabled(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, OPTIONS_ENABLED="false")
    init_db()
    from tests._signal_stub import stub_market_open_signal

    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    runner.run("premarket", now=_NOW)
    with stub_market_open_signal(price=550.0):
        result = runner.run("market-open", now=_NOW)
    assert result.success
    decisions = result.payload.get("decisions", [])
    assert not any("Options" in (d.get("reason") or "") for d in decisions)


def test_options_not_routed_for_unlisted_underlying(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch,
        tmp_path,
        OPTIONS_ENABLED_UNDERLYINGS="QQQ",  # SPY not listed
    )
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    runner.run("premarket", now=_NOW)
    result = runner.run("market-open", now=_NOW)
    assert result.success
    decisions = result.payload.get("decisions", [])
    # SPY should fall through to the equity/broker path, not options.
    assert not any("Options" in (d.get("reason") or "") for d in decisions)


def test_dry_run_does_not_route_options(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    runner.run("premarket", now=_NOW)
    result = runner.run("market-open", now=_NOW)
    assert result.success
    decisions = result.payload.get("decisions", [])
    # Dry run short-circuits to hold before options routing.
    assert all(d.get("decision") != "enter" for d in decisions)


def test_midday_manages_options(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    runner.run("premarket", now=_NOW)
    runner.run("market-open", now=_NOW)
    result = runner.run("midday", now=_NOW)
    assert result.success
    assert "options_actions" in result.payload
