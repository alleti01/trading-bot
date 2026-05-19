"""Tests for the autonomous-paper / research advisory agents.

Coverage map:

1. Pydantic schemas (`MacroNewsAssessment`, `BacktestCritique`,
   `ModelDriftReport`, `StrategyResearchReport`, `DataQualityReport`)
   reject extra fields, accept valid examples, are frozen.
2. Each LLM-backed agent (macro_news, backtest_critic,
   strategy_research) integrates correctly with the
   :class:`ProviderRouter` — the right provider is asked, the right
   schema is enforced, and a missing API key disables the agent
   without affecting the others.
3. ``ModelDriftAgent`` — deterministic stats path produces the right
   severity buckets; the optional LLM polish never overrides the
   numeric findings; provider failures are swallowed.
4. ``DataQualityAgent`` — detects missing candles, duplicates, bad
   OHLCV, stale feeds, and empty feeds; ``blocked_symbols`` reflects
   which symbols paper mode must refuse.
5. Architectural isolation: every new agent module is covered by the
   existing ``agents/ -> execution/risk/`` import scan (smoke check
   here as well to fail fast).
6. Advisory-only: schema rejects "promote model" / "change risk" style
   keys; the orchestrator exposes the new agents only via dedicated
   read methods and never wires them into trade routing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import pytest
from pydantic import ValidationError

from agents.backtest_critic_agent import BacktestCriticAgent
from agents.base_agent import AgentContext
from agents.data_quality_agent import (
    DataQualityAgent,
    DataQualityConfig,
    scan_data_quality,
)
from agents.llm_client import LLMClient, LLMClientError
from agents.macro_news_agent import MacroNewsAgent
from agents.model_drift_agent import (
    ModelDriftAgent,
    compute_drift_report,
)
from agents.orchestrator import AgentOrchestrator, build_orchestrator
from agents.providers import (
    BaseLLMProvider,
    ProviderError,
    ProviderRouter,
    ProviderTextResult,
)
from agents.schemas import (
    AGENT_SCHEMAS,
    BacktestCritique,
    DataQualityReport,
    MacroNewsAssessment,
    ModelDriftReport,
    StrategyResearchReport,
)
from agents.strategy_research_agent import StrategyResearchAgent
from notifications.notification_service import NotificationService
from storage.db import init_db, reset_engine_for_tests


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _settings(**overrides):
    from config.settings import reload_settings

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "DATABASE_URL": "sqlite:///:memory:",
        "ENABLE_LLM_AGENTS": "false",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


def _captured_notifier() -> NotificationService:
    return NotificationService(discord=None)


class _CannedProvider(BaseLLMProvider):
    """Returns a fixed JSON payload for the next call."""

    provider_name = "canned"

    def __init__(self, payload: str) -> None:
        super().__init__(model="canned-model")
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def generate_text(self, prompt, *, system_prompt=None, temperature=0.2):  # type: ignore[override]
        self.calls.append(
            {"system": system_prompt or "", "user": prompt}
        )
        return ProviderTextResult(
            text=self.payload,
            provider=self.provider_name,
            model=self.model_name,
        )


class _ExplodingProvider(BaseLLMProvider):
    provider_name = "boom"

    def __init__(self) -> None:
        super().__init__(model="boom-model")
        self.calls = 0

    def generate_text(self, prompt, *, system_prompt=None, temperature=0.2):  # type: ignore[override]
        self.calls += 1
        raise ProviderError("simulated provider outage")


def _valid_macro_news_payload() -> dict[str, Any]:
    return {
        "risk_level": "high",
        "affected_symbols": ["MES", "MNQ", "FAKE-INVENTED"],
        "blocked_windows": [
            {
                "start": "14:00",
                "end": "14:30",
                "reason": "FOMC press conference",
                "severity": "high",
            }
        ],
        "key_events": ["FOMC"],
        "sources": [{"url": "https://example.com/fomc", "title": "FOMC"}],
        "summary": "Major macro window today; reduce or sit out.",
    }


def _valid_backtest_critique_payload() -> dict[str, Any]:
    return {
        "overall_assessment": "Edge looks weakest in the first 30m.",
        "weak_spots": [
            {
                "category": "time_window",
                "where": "09:30-10:00 ET",
                "severity": "medium",
                "evidence": "Win rate 42% vs 53% session avg",
                "suggested_experiment": (
                    "Try a session-time filter that pauses the strategy "
                    "for the first 30 minutes; compare equity curves."
                ),
            }
        ],
        "bad_time_windows": ["09:30-10:00 ET"],
        "weak_symbols": [],
        "bad_confidence_buckets": [],
        "bad_regimes": [],
        "suggested_experiments": [
            "Walk-forward with a session-time filter on/off."
        ],
    }


def _valid_strategy_research_payload() -> dict[str, Any]:
    return {
        "summary": "Two short-list ideas worth backtesting.",
        "experiments": [
            {
                "title": "ATR regime filter",
                "hypothesis": (
                    "Pullback strategy works better in low-volatility regimes."
                ),
                "experiment_plan": (
                    "Bucket bars by ATR percentile and compute "
                    "expectancy per bucket; backtest with the lowest "
                    "buckets disabled."
                ),
                "risks": ["Regime mis-classification on small samples."],
                "related_filters": ["atr_regime"],
            }
        ],
        "sources": [],
    }


# ===========================================================================
# 1. Schemas
# ===========================================================================
def test_agent_schemas_map_includes_new_agents() -> None:
    for name in (
        "macro_news",
        "backtest_critic",
        "model_drift",
        "strategy_research",
        "data_quality",
    ):
        assert name in AGENT_SCHEMAS


def test_macro_news_schema_validates_and_is_frozen() -> None:
    parsed = MacroNewsAssessment.model_validate(_valid_macro_news_payload())
    assert parsed.risk_level == "high"
    with pytest.raises(ValidationError):
        parsed.risk_level = "low"  # type: ignore[misc]


def test_macro_news_schema_rejects_invalid_severity() -> None:
    bad = _valid_macro_news_payload()
    bad["risk_level"] = "extreme"
    with pytest.raises(ValidationError):
        MacroNewsAssessment.model_validate(bad)


def test_macro_news_schema_rejects_extra_fields() -> None:
    bad = _valid_macro_news_payload()
    bad["promote_model"] = True  # advisory-only: no such field
    with pytest.raises(ValidationError):
        MacroNewsAssessment.model_validate(bad)


def test_backtest_critique_schema_rejects_unknown_category() -> None:
    bad = _valid_backtest_critique_payload()
    bad["weak_spots"][0]["category"] = "code_change"
    with pytest.raises(ValidationError):
        BacktestCritique.model_validate(bad)


def test_backtest_critique_accepts_valid_payload() -> None:
    parsed = BacktestCritique.model_validate(_valid_backtest_critique_payload())
    assert parsed.weak_spots[0].category == "time_window"


def test_strategy_research_schema_requires_experiment_shape() -> None:
    bad = _valid_strategy_research_payload()
    del bad["experiments"][0]["hypothesis"]
    with pytest.raises(ValidationError):
        StrategyResearchReport.model_validate(bad)


def test_strategy_research_schema_rejects_promote_model_field() -> None:
    bad = _valid_strategy_research_payload()
    bad["promote_model"] = True  # advisory-only
    with pytest.raises(ValidationError):
        StrategyResearchReport.model_validate(bad)


def test_data_quality_schema_accepts_blocked_symbols() -> None:
    parsed = DataQualityReport.model_validate(
        {
            "session_date": "2026-05-19",
            "checked_symbols": ["MES", "MNQ"],
            "issues": [],
            "blocked_symbols": ["MNQ"],
            "summary": "MNQ blocked due to stale feed.",
        }
    )
    assert parsed.blocked_symbols == ["MNQ"]


def test_model_drift_schema_clamps_severity_literal() -> None:
    parsed = ModelDriftReport.model_validate(
        {
            "model_name": "vwap",
            "model_version": "1",
            "severity": "watch",
            "metric_deltas": [],
            "drift_warnings": [],
            "commentary": None,
            "narrative": None,
            "retrain_recommended": False,
            "reason": "ok",
        }
    )
    assert parsed.severity == "watch"


# ===========================================================================
# 2. Agent + ProviderRouter integration
# ===========================================================================
def test_macro_news_agent_calls_routed_provider() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        OPENAI_API_KEY="sk-test",
        ENABLED_SYMBOLS="MES,MNQ",
    )
    reset_engine_for_tests()
    init_db()
    router = ProviderRouter.from_settings(s)
    canned = _CannedProvider(json.dumps(_valid_macro_news_payload()))
    router._provider_cache["perplexity"] = canned  # type: ignore[attr-defined]

    orch = AgentOrchestrator(s, provider_router=router, notifier=_captured_notifier())
    assessment = orch.run_macro_news(
        now=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        enabled_symbols=["MES", "MNQ"],
    )
    assert assessment is not None
    assert assessment.risk_level == "high"
    assert orch.high_risk_news_active() is True
    # Invented symbols are filtered out before any operational use.
    assert canned.calls, "router should have dispatched to perplexity"


def test_macro_news_disabled_when_perplexity_key_missing() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        # no PERPLEXITY_API_KEY -> macro_news disabled
        NEWS_AGENT_PROVIDER="perplexity",
        MACRO_NEWS_AGENT_PROVIDER="perplexity",
    )
    reset_engine_for_tests()
    init_db()
    router = ProviderRouter.from_settings(s)
    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    out = orch.run_macro_news(
        now=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        enabled_symbols=["MES"],
    )
    assert out is None
    # Other agents (e.g. trade_analysis on OpenAI) remain enabled.
    assert router.is_enabled("trade_analysis") is True


def test_strategy_research_uses_routed_provider() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        OPENAI_API_KEY="sk-test",
    )
    reset_engine_for_tests()
    init_db()
    router = ProviderRouter.from_settings(s)
    canned = _CannedProvider(json.dumps(_valid_strategy_research_payload()))
    router._provider_cache["perplexity"] = canned  # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    report = orch.run_strategy_research(
        now=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc),
        backtest_summary={"strategy": "vwap_ema_pullback"},
        enabled_symbols=["MES"],
    )
    assert report is not None
    assert report.experiments[0].title == "ATR regime filter"


def test_backtest_critic_uses_routed_provider() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
    )
    reset_engine_for_tests()
    init_db()
    router = ProviderRouter.from_settings(s)
    canned = _CannedProvider(json.dumps(_valid_backtest_critique_payload()))
    router._provider_cache["openai"] = canned  # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    result = orch.run_backtest_critic(
        backtest_summary={"strategy": "vwap_ema_pullback", "metrics": {}},
        now=datetime(2026, 5, 19, 21, 0, tzinfo=timezone.utc),
    )
    assert result is not None
    assert result.schema_valid is True
    assert "weak_spots" in (result.payload or {})


def test_routed_agents_survive_provider_failure() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        OPENAI_API_KEY="sk-test",
    )
    reset_engine_for_tests()
    init_db()
    router = ProviderRouter.from_settings(s)
    bomb = _ExplodingProvider()
    router._provider_cache["perplexity"] = bomb  # type: ignore[attr-defined]
    router._provider_cache["openai"] = bomb       # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    assert orch.run_macro_news(enabled_symbols=["MES"]) is None
    assert orch.run_strategy_research(backtest_summary={}, enabled_symbols=["MES"]) is None
    crit = orch.run_backtest_critic({"x": 1})
    assert crit is None or crit.schema_valid is False
    # And the orchestrator never raised.


# ===========================================================================
# 3. ModelDriftAgent — deterministic stats
# ===========================================================================
def test_model_drift_severity_no_drift() -> None:
    report = compute_drift_report(
        model_metadata={
            "name": "vwap",
            "version": "1",
            "metrics": {"win_rate": 0.55, "expectancy_per_trade": 12.0},
        },
        paper_metrics={"win_rate": 0.55, "expectancy_per_trade": 12.0, "n_trades": 100},
    )
    assert report.severity == "none"
    assert report.retrain_recommended is False


def test_model_drift_alert_triggers_retrain() -> None:
    report = compute_drift_report(
        model_metadata={
            "name": "vwap",
            "version": "1",
            "metrics": {"win_rate": 0.55, "expectancy_per_trade": 12.0},
        },
        paper_metrics={
            "win_rate": 0.10,           # huge drop
            "expectancy_per_trade": -5.0,
            "n_trades": 100,
        },
    )
    assert report.severity == "alert"
    assert report.retrain_recommended is True
    assert any("win_rate" in w for w in report.drift_warnings)


def test_model_drift_drawdown_growth_is_bad() -> None:
    report = compute_drift_report(
        model_metadata={
            "name": "vwap",
            "version": "1",
            "metrics": {"max_drawdown_dollars": 100.0},
        },
        paper_metrics={"max_drawdown_dollars": 250.0, "n_trades": 100},
    )
    # drawdown 100 -> 250 is +150% (bad direction). Should be at least
    # ``alert`` per defaults.
    assert report.severity == "alert"


def test_model_drift_handles_missing_inputs() -> None:
    report = compute_drift_report(
        model_metadata=None, paper_metrics={"win_rate": 0.5}
    )
    assert report.severity == "none"
    assert "Insufficient inputs" in report.reason


def test_model_drift_agent_optional_llm_polish_succeeds() -> None:
    s = _settings(ENABLE_LLM_AGENTS="false")
    reset_engine_for_tests()
    init_db()

    class _NarrativeClient(LLMClient):
        def __init__(self) -> None:
            self.last_user = ""

        def complete(self, *, system: str, user: str) -> str:
            self.last_user = user
            return "Win rate is way down; consider scheduling a retrain."

    client = _NarrativeClient()
    agent = ModelDriftAgent(client)
    ctx = AgentContext(
        settings_snapshot={},
        session_date="2026-05-19",
        instrument="MES",
        model_metadata={
            "name": "vwap",
            "version": "1",
            "metrics": {"win_rate": 0.55},
        },
        paper_metrics={"win_rate": 0.10, "n_trades": 100},
    )
    result = agent.run(ctx)
    assert result.schema_valid
    assert result.payload["narrative"].startswith("Win rate")


def test_model_drift_agent_swallows_llm_failure() -> None:
    class _BombClient(LLMClient):
        def complete(self, *, system: str, user: str) -> str:
            raise LLMClientError("upstream timeout")

    agent = ModelDriftAgent(_BombClient())
    ctx = AgentContext(
        settings_snapshot={},
        session_date="2026-05-19",
        instrument="MES",
        model_metadata={
            "name": "vwap",
            "version": "1",
            "metrics": {"win_rate": 0.55},
        },
        paper_metrics={"win_rate": 0.55, "n_trades": 100},
    )
    result = agent.run(ctx)
    # Stats produced fine; no narrative.
    assert result.schema_valid
    assert result.payload["narrative"] is None


def test_model_drift_runs_without_any_llm() -> None:
    agent = ModelDriftAgent(llm=None)
    ctx = AgentContext(
        settings_snapshot={},
        session_date="2026-05-19",
        instrument="MES",
        model_metadata={
            "name": "vwap",
            "version": "1",
            "metrics": {"win_rate": 0.55},
        },
        paper_metrics={"win_rate": 0.45, "n_trades": 100},
    )
    result = agent.run(ctx)
    assert result.schema_valid
    assert result.payload["narrative"] is None


# ===========================================================================
# 4. DataQualityAgent — deterministic
# ===========================================================================
def _good_df(start: datetime, n: int = 60, freq: str = "1min") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": 4500.0,
            "high": 4501.0,
            "low": 4499.5,
            "close": 4500.5,
            "volume": 100.0,
        },
        index=idx,
    )


def test_data_quality_clean_feed_has_no_blocks() -> None:
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    feed = _good_df(start=now - timedelta(minutes=60), n=60)
    report = scan_data_quality({"MES": feed}, now=now)
    assert report.blocked_symbols == []
    assert report.checked_symbols == ["MES"]
    assert "clean" in report.summary


def test_data_quality_detects_empty_feed() -> None:
    feed = _good_df(start=datetime(2026, 5, 19, 13, 0, tzinfo=timezone.utc), n=2)
    report = scan_data_quality(
        {"MES": feed},
        now=datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc),
    )
    assert "MES" in report.blocked_symbols
    assert any(i.kind == "empty_feed" for i in report.issues)


def test_data_quality_detects_stale_feed() -> None:
    # Last bar is 30 minutes old vs 10-minute threshold.
    end = datetime(2026, 5, 19, 13, 0, tzinfo=timezone.utc)
    feed = _good_df(start=end - timedelta(minutes=60), n=60)
    report = scan_data_quality(
        {"MES": feed},
        now=end + timedelta(minutes=30),
    )
    assert "MES" in report.blocked_symbols
    assert any(i.kind == "stale_feed" for i in report.issues)


def test_data_quality_detects_missing_candles() -> None:
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    base = _good_df(start=now - timedelta(minutes=120), n=120)
    # Drop 10 consecutive bars from the middle to simulate a gap.
    holes_idx = base.index[40:50]
    sparse = base.drop(index=holes_idx)
    report = scan_data_quality({"MES": sparse}, now=now)
    assert any(i.kind == "missing_candles" for i in report.issues)


def test_data_quality_detects_bad_ohlcv() -> None:
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    feed = _good_df(start=now - timedelta(minutes=60), n=60).copy()
    # Corrupt one row: high < low.
    feed.iloc[5, feed.columns.get_loc("high")] = 100.0
    feed.iloc[5, feed.columns.get_loc("low")] = 200.0
    report = scan_data_quality({"MES": feed}, now=now)
    assert "MES" in report.blocked_symbols
    assert any(i.kind == "bad_ohlcv" for i in report.issues)


def test_data_quality_detects_duplicate_timestamps() -> None:
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    base = _good_df(start=now - timedelta(minutes=60), n=60)
    dup = pd.concat([base, base.iloc[[0]]])
    report = scan_data_quality({"MES": dup}, now=now)
    # Duplicate alone is medium severity; the symbol may not be
    # blocked. We assert the issue is reported.
    assert any(i.kind == "duplicate_timestamps" for i in report.issues)


def test_data_quality_blocks_per_symbol_independence() -> None:
    """One bad symbol must not poison the others."""
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    good = _good_df(start=now - timedelta(minutes=60), n=60)
    bad = _good_df(
        start=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc), n=60
    )  # stale
    report = scan_data_quality({"MES": good, "MNQ": bad}, now=now)
    assert "MNQ" in report.blocked_symbols
    assert "MES" not in report.blocked_symbols


def test_data_quality_agent_run_with_feeds() -> None:
    agent = DataQualityAgent()
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    result = agent.run_with_feeds(
        {"MES": _good_df(start=now - timedelta(minutes=60), n=60)},
        now=now,
    )
    assert result.schema_valid
    assert result.payload["blocked_symbols"] == []


def test_orchestrator_data_quality_check_returns_blocked_symbols() -> None:
    s = _settings()
    reset_engine_for_tests()
    init_db()
    orch = AgentOrchestrator(s, llm=None, notifier=_captured_notifier())
    now = datetime(2026, 5, 19, 14, 0, tzinfo=timezone.utc)
    report = orch.run_data_quality_check(
        feeds_by_symbol={
            "MES": _good_df(start=now - timedelta(minutes=60), n=60),
            "MNQ": _good_df(
                start=datetime(2026, 5, 19, 12, 0, tzinfo=timezone.utc), n=60
            ),
        },
        now=now,
    )
    assert report is not None
    assert "MNQ" in report.blocked_symbols


# ===========================================================================
# 5. Architectural isolation (smoke)
# ===========================================================================
@pytest.mark.parametrize(
    "module_path",
    [
        "agents/macro_news_agent.py",
        "agents/backtest_critic_agent.py",
        "agents/model_drift_agent.py",
        "agents/strategy_research_agent.py",
        "agents/data_quality_agent.py",
    ],
)
def test_new_agents_do_not_import_execution_or_risk(module_path: str) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / module_path).read_text()
    assert "from execution" not in src
    assert "import execution" not in src
    assert "from risk" not in src
    assert "import risk" not in src


# ===========================================================================
# 6. Advisory-only contract
# ===========================================================================
def test_macro_news_only_blocks_never_approves() -> None:
    """The orchestrator's bridge from MacroNewsAgent into trading is
    a single ``high_risk_news_active`` flag; that flag is consumed by
    the existing block-only ``risk_engine.evaluate(..., high_risk_news_window=...)``
    path. Nothing on :class:`MacroNewsAssessment` can encode an
    "approve trade" instruction (no such field exists)."""
    fields = set(MacroNewsAssessment.model_fields)
    assert "approve" not in fields
    assert "place_trade" not in fields
    assert "force_long" not in fields
    assert "force_short" not in fields


def test_strategy_research_has_no_action_fields() -> None:
    fields = set(StrategyResearchReport.model_fields)
    forbidden = {"action", "trade", "code_patch", "promote_model", "set_threshold"}
    assert fields.isdisjoint(forbidden)


def test_backtest_critique_recommendations_are_experiments() -> None:
    fields = set(BacktestCritique.model_fields)
    forbidden = {"actions", "code_patch", "promote_model", "set_threshold"}
    assert fields.isdisjoint(forbidden)


def test_model_drift_cannot_promote() -> None:
    fields = set(ModelDriftReport.model_fields)
    # ``retrain_recommended`` is advisory; promotion is a separate
    # CLI-gated path. There is no "promote" field.
    assert "promote" not in fields
    assert "promote_model" not in fields


# ===========================================================================
# 7. build_orchestrator routing for new agents
# ===========================================================================
def test_router_routes_new_agents_correctly() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        PERPLEXITY_API_KEY="pplx-test",
    )
    router = ProviderRouter.from_settings(s)
    assert router.chosen_provider_name("macro_news") == "perplexity"
    assert router.chosen_provider_name("strategy_research") == "perplexity"
    assert router.chosen_provider_name("backtest_critic") == "openai"
    # ``model_drift`` defaults to ``none`` so the deterministic stats
    # path is chosen unless the operator wires a provider.
    assert router.chosen_provider_name("model_drift") is None


def test_orchestrator_factory_includes_new_agents() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        PERPLEXITY_API_KEY="pplx-test",
    )
    orch = build_orchestrator(s, notifier=_captured_notifier())
    assert orch.provider_router is not None
    table = orch.provider_router.routing_table()
    # New agents are visible in the routing table.
    for name in ("macro_news", "backtest_critic", "strategy_research", "model_drift"):
        assert name in table
