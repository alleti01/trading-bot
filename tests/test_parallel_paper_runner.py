"""Parallel paper runner tests — isolation, safety, reporting."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from config.settings import reload_settings
from evaluation.evaluation_context import EvaluationContext
from evaluation.parallel_paper_runner import ParallelPaperRunner
from integrations import LiveExecutionRefused, MockBroker
from notifications.notification_service import NotificationService
from reports.paper_evaluation_report import write_combined_summary, write_track_report
from storage.db import init_db


def _settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **overrides: str):
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKFLOW_MEMORY_DIR", str(memory))
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("ENABLE_PARALLEL_PAPER", "true")
    monkeypatch.setenv("PARALLEL_BROKERS", "futures_sim,alpaca")
    monkeypatch.setenv("ALPACA_ENABLED_SYMBOLS", "SPY,QQQ")
    monkeypatch.setenv("FUTURES_SIM_ENABLED_SYMBOLS", "MES,MNQ")
    monkeypatch.setenv("ALPACA_EVALUATION_ID", "alpaca_test")
    monkeypatch.setenv("FUTURES_SIM_EVALUATION_ID", "futures_test")
    monkeypatch.setenv("ALPACA_STATE_PATH", str(tmp_path / "alpaca_state.json"))
    monkeypatch.setenv("FUTURES_SIM_STATE_PATH", str(tmp_path / "futures_state.json"))
    monkeypatch.setenv("AUTONOMOUS_TRADING_ENABLED", "false")
    monkeypatch.setenv("WORKFLOW_WEEKDAYS_ONLY", "false")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_parallel_creates_two_isolated_contexts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    tracks = runner.build_tracks()
    assert len(tracks) == 2
    providers = {t.broker_provider for t in tracks}
    assert providers == {"alpaca", "futures_sim"}
    alpaca_ctx = runner.contexts["alpaca"]
    futures_ctx = runner.contexts["futures_sim"]
    assert alpaca_ctx.evaluation_id != futures_ctx.evaluation_id
    assert alpaca_ctx.state_path != futures_ctx.state_path
    assert alpaca_ctx.report_path != futures_ctx.report_path


def test_alpaca_and_futures_sim_do_not_share_state_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()
    a = runner.contexts["alpaca"]
    f = runner.contexts["futures_sim"]
    a.save_state()
    f.save_state()
    assert a.state_path.exists()
    assert f.state_path.exists()
    a_data = json.loads(a.state_path.read_text())
    f_data = json.loads(f.state_path.read_text())
    assert a_data["evaluation_id"] == "alpaca_test"
    assert f_data["evaluation_id"] == "futures_test"
    assert a_data["broker_provider"] == "alpaca"
    assert f_data["broker_provider"] == "futures_sim"


def test_alpaca_blocks_futures_symbols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()
    broker = runner.build_broker_for(runner.contexts["alpaca"])
    result = broker.validate_order(symbol="MES", qty=1, side="buy")
    assert not result.valid
    reason = (result.reason or "").lower()
    assert "not enabled" in reason or "futures" in reason


def test_futures_sim_accepts_futures_symbols(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()
    broker = runner.build_broker_for(runner.contexts["futures_sim"])
    assert isinstance(broker, MockBroker)
    result = broker.validate_order(symbol="MES", qty=1, side="buy")
    assert result.valid


def test_discord_messages_include_broker_and_eval_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()
    ctx = runner.contexts["alpaca"]
    tags = ctx.discord_tags()
    assert tags["broker_provider"] == "alpaca"
    assert tags["evaluation_id"] == "alpaca_test"


def test_live_mode_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_EXECUTION_MODE="LIVE")
    runner = ParallelPaperRunner(settings, dry_run=False)
    with pytest.raises(LiveExecutionRefused):
        runner.build_tracks()


def test_failure_in_one_broker_does_not_crash_the_other(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()

    def _exploding_broker(*a, **kw):
        raise RuntimeError("alpaca network down")

    original_build = runner.build_broker_for

    def patched(ctx):
        if ctx.broker_provider == "alpaca":
            raise RuntimeError("alpaca network down")
        return original_build(ctx)

    runner.build_broker_for = patched  # type: ignore[assignment]
    results = runner.run_all()
    assert "futures_sim" in results
    futures_result = results["futures_sim"]
    assert futures_result.get("success") is True or "error" not in futures_result
    assert "alpaca" in results
    assert results["alpaca"].get("success") is False


def test_reports_written_separately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()
    for provider, ctx in runner.contexts.items():
        path = write_track_report(ctx, payload={"workflows": [], "trades": 0})
        assert path.exists()
        assert ctx.evaluation_id in path.parent.name


def test_combined_summary_generated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    runner = ParallelPaperRunner(settings, dry_run=True)
    runner.build_tracks()
    results = {p: {"success": True, "trades": 0} for p in runner.contexts}
    evaluations_dir = tmp_path / "evaluations"
    path = write_combined_summary(
        runner.contexts, results=results, evaluations_dir=evaluations_dir
    )
    assert path.exists()
    text = path.read_text()
    assert "Alpaca tests broker/API plumbing" in text
    assert "futures_sim tests futures strategy/model behavior" in text
    assert "Do not compare PnL directly" in text


def test_run_all_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    runner = ParallelPaperRunner(settings, dry_run=True)
    results = runner.run_all()
    assert "futures_sim" in results
    assert "alpaca" in results
