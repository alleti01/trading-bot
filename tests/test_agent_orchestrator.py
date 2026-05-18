"""End-to-end orchestrator: EOD run with mock LLM persists 5 agent_outputs."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from agents.llm_client import MockLLMClient
from agents.orchestrator import AgentOrchestrator
from notifications.notification_service import NotificationService
from reports.daily_report import write_daily_report
from storage.db import init_db, session_scope
from storage.tables import AgentOutput, ClosedTrade, RiskBlock


def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "DATABASE_URL": "sqlite:///:memory:",
        "ENABLE_LLM_AGENTS": "true",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def _seed_one_trade(base: datetime) -> None:
    with session_scope() as session:
        session.add(
            ClosedTrade(
                paper_trade_id=None,
                setup_id="orchestrator-test-1",
                instrument="MES",
                direction="long",
                quantity=1.0,
                entry_ts=base,
                entry_price=4500.0,
                exit_ts=base + timedelta(minutes=4),
                exit_price=4504.0,
                exit_reason="tp",
                pnl=20.0,
                commission=0.5,
                slippage=0.0,
            )
        )
        session.add(
            RiskBlock(
                setup_id="blocked-1",
                ts=base,
                rule="max_trades_per_day",
                reason="seeded",
            )
        )


def _captured_notifier() -> NotificationService:
    return NotificationService(discord=None)  # log-only


# ---------------------------------------------------------------------------
def test_orchestrator_with_none_llm_is_no_op(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    orch = AgentOrchestrator(s, llm=None, notifier=_captured_notifier())
    now = datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    result = orch.run_end_of_day(now=now)
    assert result.n_total() == 0
    assert orch.high_risk_news_active() is False


def test_eod_persists_five_agent_outputs_with_mock_llm(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    base = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    _seed_one_trade(base)
    artifacts = write_daily_report(s, now=base)

    orch = AgentOrchestrator(s, llm=MockLLMClient(), notifier=_captured_notifier())
    result = orch.run_end_of_day(now=base, daily_md_path=artifacts.md_path)

    assert result.n_total() == 5
    assert result.n_valid() == 5

    with session_scope() as session:
        rows = session.execute(select(AgentOutput)).scalars().all()
    names = sorted(r.agent_name for r in rows)
    assert names == ["model_review", "news", "report", "risk_explainer", "trade_journal"]
    assert all(r.schema_valid for r in rows)


def test_eod_appends_commentary_to_daily_md(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    base = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    _seed_one_trade(base)
    artifacts = write_daily_report(s, now=base)

    md_before = artifacts.md_path.read_text()
    orch = AgentOrchestrator(s, llm=MockLLMClient(), notifier=_captured_notifier())
    result = orch.run_end_of_day(now=base, daily_md_path=artifacts.md_path)
    assert result.appended_md_path == artifacts.md_path

    md_after = artifacts.md_path.read_text()
    assert "## Agent commentary" in md_after
    assert len(md_after) > len(md_before)


def test_one_failing_agent_does_not_stop_the_others(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    base = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    _seed_one_trade(base)
    artifacts = write_daily_report(s, now=base)

    mock = MockLLMClient(raise_for_agents={"news"})
    orch = AgentOrchestrator(s, llm=mock, notifier=_captured_notifier())
    result = orch.run_end_of_day(now=base, daily_md_path=artifacts.md_path)

    assert result.n_total() == 5
    assert result.results["news"].schema_valid is False
    # The other four should still have run.
    valid = {n for n, r in result.results.items() if r.schema_valid}
    assert valid == {"risk_explainer", "trade_journal", "report", "model_review"}


def test_persisted_invalid_outputs_are_kept_for_audit(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    base = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    artifacts = write_daily_report(s, now=base)

    mock = MockLLMClient(raise_for_agents={"news", "report"})
    orch = AgentOrchestrator(s, llm=mock, notifier=_captured_notifier())
    orch.run_end_of_day(now=base, daily_md_path=artifacts.md_path)

    with session_scope() as session:
        rows = session.execute(select(AgentOutput)).scalars().all()
    by_name = {r.agent_name: r for r in rows}
    assert by_name["news"].schema_valid is False
    assert by_name["report"].schema_valid is False
    assert by_name["news"].error and "llm_error" in by_name["news"].error


def test_pre_session_news_updates_high_risk_flag(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    high_risk_payload = json.dumps(
        {
            "high_risk_window": True,
            "severity": "high",
            "events": ["FOMC"],
            "summary": "FOMC at 14:00",
            "recommendation": "Stand aside.",
        }
    )
    mock = MockLLMClient(responses_by_agent={"news": high_risk_payload})
    orch = AgentOrchestrator(s, llm=mock, notifier=_captured_notifier())
    assert orch.high_risk_news_active() is False
    out = orch.run_pre_session_news(
        now=datetime(2026, 5, 18, 13, 25, tzinfo=timezone.utc)
    )
    assert out is not None
    assert out.high_risk_window is True
    assert orch.high_risk_news_active() is True


def test_pre_session_news_failure_keeps_previous_flag(tmp_path: Path) -> None:
    """A transient LLM error must not silently *unblock* trading."""
    s = _settings(tmp_path)
    # Step 1: set the flag via a successful run.
    high_risk_payload = json.dumps(
        {
            "high_risk_window": True,
            "severity": "high",
            "events": [],
            "summary": "x",
            "recommendation": "y",
        }
    )
    mock = MockLLMClient(responses_by_agent={"news": high_risk_payload})
    orch = AgentOrchestrator(s, llm=mock, notifier=_captured_notifier())
    orch.run_pre_session_news(now=datetime(2026, 5, 18, 13, 25, tzinfo=timezone.utc))
    assert orch.high_risk_news_active() is True

    # Step 2: simulate a failure on the next pre-session run; flag must persist.
    orch.llm = MockLLMClient(raise_for_agents={"news"})
    out = orch.run_pre_session_news(
        now=datetime(2026, 5, 18, 13, 25, tzinfo=timezone.utc)
    )
    assert out is None
    assert orch.high_risk_news_active() is True
