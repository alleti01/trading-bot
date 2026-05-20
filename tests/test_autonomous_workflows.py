"""Autonomous workflow layer tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import reload_settings
from notifications.discord import SendResult
from notifications.notification_service import NotificationService
from storage.db import init_db
from tests.test_notification_service import _FakeDiscord
from workflows.gates import WorkflowGates
from workflows.market_open import MarketOpenWorkflow
from workflows.memory import ensure_memory_files, has_research_for_date
from workflows.premarket import PremarketWorkflow
from workflows.workflow_runner import WorkflowRunner


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: str):
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKFLOW_MEMORY_DIR", str(memory))
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "DRY_RUN")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("AUTONOMOUS_TRADING_ENABLED", "false")
    monkeypatch.setenv("ENABLE_LLM_AGENTS", "false")
    monkeypatch.setenv("PERPLEXITY_ENABLED", "false")
    monkeypatch.setenv("WORKFLOW_WEEKDAYS_ONLY", "false")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_dry_run_premarket_writes_research_without_orders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    now = datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc)
    result = runner.run("premarket", now=now)
    assert result.success
    assert result.dry_run
    paths = ensure_memory_files(settings)
    assert has_research_for_date(paths.research_log, "2026-05-19")


def test_missing_memory_files_created_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    paths = ensure_memory_files(settings)
    for p in (
        paths.strategy,
        paths.research_log,
        paths.trade_log,
        paths.weekly_review,
    ):
        assert p.exists()
        assert p.read_text(encoding="utf-8").strip()


def test_market_open_runs_premarket_inline_when_research_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    result = runner.run("market-open", now=now)
    assert result.success
    paths = ensure_memory_files(settings)
    assert has_research_for_date(paths.research_log, "2026-05-19")


def test_market_open_refuses_without_research_when_inline_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)

    with patch.object(PremarketWorkflow, "run") as mock_pre:
        mock_pre.return_value = MagicMock(success=False)
        result = runner.run("market-open", now=now)
    assert not result.success
    assert any("research" in e.lower() for e in result.errors)


def test_autonomous_execution_only_when_enabled_and_paper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(
        monkeypatch,
        tmp_path,
        AUTONOMOUS_TRADING_ENABLED="true",
        WORKFLOW_EXECUTION_MODE="PAPER",
    )
    init_db()
    gates = WorkflowGates(settings)
    assert gates.autonomous_execution_allowed("PAPER")
    assert not gates.autonomous_execution_allowed("DRY_RUN")

    runner_dry = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    assert runner_dry.dry_run is True

    settings2 = _settings(
        monkeypatch,
        tmp_path,
        AUTONOMOUS_TRADING_ENABLED="false",
        WORKFLOW_EXECUTION_MODE="PAPER",
    )
    assert not WorkflowGates(settings2).autonomous_execution_allowed("PAPER")


def test_live_workflow_mode_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_EXECUTION_MODE="LIVE")
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    result = runner.run("premarket", now=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc))
    assert result.skipped or not result.success
    assert "LIVE" in (result.skip_reason or "") or any(
        "LIVE" in e for e in result.errors
    )


def test_discord_failure_does_not_crash_workflow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    fake = _FakeDiscord(SendResult(delivered=False), raises=RuntimeError("discord down"))
    notifier = NotificationService(discord=fake)
    runner = WorkflowRunner(settings, notifier=notifier, dry_run=True)
    result = runner.run(
        "daily-summary",
        now=datetime(2026, 5, 19, 20, 0, tzinfo=timezone.utc),
    )
    assert result.success


def test_dry_run_market_open_simulates_without_paper_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path, AUTONOMOUS_TRADING_ENABLED="true")
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    now = datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)
    runner.run("premarket", now=now)
    with patch("workflows.market_open.submit_paper_order") as mock_submit:
        result = runner.run("market-open", now=now)
        mock_submit.assert_not_called()
    assert result.success
    decisions = result.payload.get("decisions", [])
    assert decisions
    assert all(d.get("decision") != "enter" for d in decisions)


def test_paper_autonomous_can_submit_when_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(
        monkeypatch,
        tmp_path,
        AUTONOMOUS_TRADING_ENABLED="true",
        WORKFLOW_EXECUTION_MODE="PAPER",
        MAX_OPEN_POSITIONS="1",
        MAX_ACTIVE_SYMBOLS="1",
    )
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=False)
    now = datetime(2026, 5, 19, 14, 30, tzinfo=timezone.utc)
    runner.run("premarket", now=now)
    result = runner.run("market-open", now=now)
    assert result.success
    assert result.payload.get("actions_taken", 0) >= 1
    assert "mock" in str(result.payload.get("broker_provider", ""))


def test_run_day_sequence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = WorkflowRunner.from_settings(settings, cli_dry_run=True)
    now = datetime(2026, 5, 19, 16, 0, tzinfo=timezone.utc)
    result = runner.run("run-day", now=now)
    assert result.success
    assert len(result.payload.get("results", [])) == 4
