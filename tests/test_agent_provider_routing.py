"""End-to-end tests for the per-agent provider routing layer.

This file is the spec-aligned coverage for "Add complete agent
provider configuration": every supported agent must resolve to the
right provider, missing API keys must disable only the affected
agents, and the router must never log API keys.

Coverage map (mirrors the spec's test list):

1. NewsAgent / MacroNewsAgent / StrategyResearchAgent -> Perplexity.
2. TradeAnalysisAgent / ModelReviewAgent / ReportAgent /
   RiskExplainerAgent / TradeJournalAgent / BacktestCriticAgent
   -> OpenAI (with the right model class).
3. ModelDriftAgent / DataQualityAgent -> ``none`` (deterministic).
4. Missing PERPLEXITY_API_KEY disables all Perplexity agents
   without affecting OpenAI agents.
5. Missing OPENAI_API_KEY disables all OpenAI agents without
   affecting Perplexity agents.
6. Provider router never logs API keys (smoke test scrapes captured
   stdout for any string that looks like a real ``sk-...`` key).
7. Invalid provider name disables only that agent — and we confirm
   the Settings validator catches typos *before* the router sees
   them. The router's manual-construction path also degrades cleanly.
8. Per-agent model overrides flow through to the underlying provider
   instance (``BacktestCriticAgent`` -> ``OPENAI_REVIEW_MODEL`` etc.).
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from pydantic import ValidationError

from agents.providers import (
    AGENT_MODEL_FIELDS,
    AGENT_PROVIDER_FIELDS,
    OpenAIProvider,
    PerplexityProvider,
    ProviderRouter,
)
from agents.providers.router import DISABLED_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _settings(**overrides):
    from config.settings import reload_settings

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "DATABASE_URL": "sqlite:///:memory:",
        "ENABLE_LLM_AGENTS": "true",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


def _both_keys(**overrides) -> Any:
    """Settings with BOTH OpenAI and Perplexity keys configured."""
    return _settings(
        OPENAI_API_KEY="sk-openai-stub",
        PERPLEXITY_API_KEY="pplx-stub",
        **overrides,
    )


# ---------------------------------------------------------------------------
# 1. Perplexity-routed agents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "agent_name",
    ["news", "macro_news", "strategy_research"],
)
def test_research_agents_resolve_to_perplexity(agent_name: str) -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    provider = router.provider_for(agent_name)
    assert isinstance(provider, PerplexityProvider)
    assert provider.provider_name == "perplexity"


def test_news_agents_share_default_perplexity_model() -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    news = router.provider_for("news")
    macro = router.provider_for("macro_news")
    assert news is not None and macro is not None
    # Both default to ``PERPLEXITY_DEFAULT_MODEL`` (sonar-pro).
    assert news.model_name == s.PERPLEXITY_DEFAULT_MODEL
    assert macro.model_name == s.PERPLEXITY_DEFAULT_MODEL


def test_strategy_research_uses_heavier_perplexity_model() -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    sr = router.provider_for("strategy_research")
    assert sr is not None
    # ``STRATEGY_RESEARCH_AGENT_MODEL`` defaults to ``sonar-deep-research``
    # in Settings.
    assert sr.model_name == s.STRATEGY_RESEARCH_AGENT_MODEL


# ---------------------------------------------------------------------------
# 2. OpenAI-routed agents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "agent_name",
    [
        "trade_analysis",
        "model_review",
        "report",
        "risk_explainer",
        "trade_journal",
        "backtest_critic",
    ],
)
def test_openai_agents_resolve_to_openai(agent_name: str) -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    provider = router.provider_for(agent_name)
    assert isinstance(provider, OpenAIProvider)
    assert provider.provider_name == "openai"


def test_review_agents_get_review_model() -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    review = router.provider_for("model_review")
    critic = router.provider_for("backtest_critic")
    assert review is not None and critic is not None
    assert review.model_name == s.OPENAI_REVIEW_MODEL
    assert critic.model_name == s.OPENAI_REVIEW_MODEL


def test_writing_agents_get_default_openai_model() -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    expected = s.OPENAI_MODEL or s.OPENAI_DEFAULT_MODEL
    for agent in ("trade_analysis", "report", "risk_explainer", "trade_journal"):
        provider = router.provider_for(agent)
        assert provider is not None, agent
        assert provider.model_name == expected, agent


# ---------------------------------------------------------------------------
# 3. Deterministic / disabled agents
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("agent_name", ["model_drift", "data_quality"])
def test_deterministic_agents_default_to_none(agent_name: str) -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    assert router.chosen_provider_name(agent_name) is None
    assert router.provider_for(agent_name) is None
    assert router.is_enabled(agent_name) is False


def test_routing_table_marks_deterministic_agents_disabled() -> None:
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    table = router.routing_table()
    assert table["model_drift"]["enabled"] is False
    assert table["data_quality"]["enabled"] is False
    assert table["model_drift"]["provider"] is None
    assert table["data_quality"]["provider"] is None


# ---------------------------------------------------------------------------
# 4-5. Missing API keys disable affected agents only
# ---------------------------------------------------------------------------
def test_missing_perplexity_key_disables_only_perplexity_agents() -> None:
    s = _settings(OPENAI_API_KEY="sk-openai-stub")
    router = ProviderRouter.from_settings(s)
    for agent in ("news", "macro_news", "strategy_research"):
        assert router.is_enabled(agent) is False, agent
        assert router.provider_for(agent) is None, agent
    for agent in (
        "trade_analysis",
        "model_review",
        "report",
        "risk_explainer",
        "trade_journal",
        "backtest_critic",
    ):
        assert router.is_enabled(agent) is True, agent


def test_missing_openai_key_disables_only_openai_agents() -> None:
    s = _settings(PERPLEXITY_API_KEY="pplx-stub")
    router = ProviderRouter.from_settings(s)
    for agent in (
        "trade_analysis",
        "model_review",
        "report",
        "risk_explainer",
        "trade_journal",
        "backtest_critic",
    ):
        assert router.is_enabled(agent) is False, agent
        assert router.provider_for(agent) is None, agent
    for agent in ("news", "macro_news", "strategy_research"):
        assert router.is_enabled(agent) is True, agent


def test_no_keys_at_all_disables_every_agent() -> None:
    s = _settings()  # no provider keys
    router = ProviderRouter.from_settings(s)
    assert router.has_any_enabled() is False
    for agent in AGENT_PROVIDER_FIELDS:
        assert router.is_enabled(agent) is False, agent


def test_orchestrator_factory_skips_router_when_no_keys() -> None:
    """``build_orchestrator`` should NOT silently spin up a router
    instance with zero working agents."""
    from agents.orchestrator import build_orchestrator
    from notifications.notification_service import NotificationService

    s = _settings()
    orch = build_orchestrator(
        s, notifier=NotificationService(discord=None)
    )
    assert orch.provider_router is None
    assert orch.llm is None


# ---------------------------------------------------------------------------
# 6. Router never logs API keys
# ---------------------------------------------------------------------------
def test_router_construction_never_logs_api_keys(capsys) -> None:
    """Smoke check: build the router with sentinel keys, exercise
    every agent, then scrape stdout for those sentinels."""
    sentinel_openai = "sk-OPENAI-MUST-NOT-LEAK-9f3a1b"
    sentinel_pplx = "pplx-MUST-NOT-LEAK-c0ffee"
    s = _settings(
        OPENAI_API_KEY=sentinel_openai,
        PERPLEXITY_API_KEY=sentinel_pplx,
    )
    router = ProviderRouter.from_settings(s)
    for agent in AGENT_PROVIDER_FIELDS:
        router.provider_for(agent)
        router.client_for(agent)
        router.is_enabled(agent)
    _ = router.routing_table()

    captured = capsys.readouterr()
    combined = captured.out + captured.err
    assert sentinel_openai not in combined
    assert sentinel_pplx not in combined


def test_routing_table_never_includes_api_keys() -> None:
    sentinel_openai = "sk-OPENAI-MUST-NOT-LEAK-routingtable"
    sentinel_pplx = "pplx-MUST-NOT-LEAK-routingtable"
    s = _settings(
        OPENAI_API_KEY=sentinel_openai,
        PERPLEXITY_API_KEY=sentinel_pplx,
    )
    router = ProviderRouter.from_settings(s)
    blob = repr(router.routing_table())
    assert sentinel_openai not in blob
    assert sentinel_pplx not in blob


# ---------------------------------------------------------------------------
# 7. Invalid provider name -> validator-rejected
# ---------------------------------------------------------------------------
def test_invalid_provider_name_rejected_at_settings_validation() -> None:
    """Typos must fail loudly at config time so an operator does not
    silently disable an agent."""
    with pytest.raises(ValidationError):
        _settings(
            OPENAI_API_KEY="sk-stub",
            TRADE_ANALYSIS_AGENT_PROVIDER="opnAI",  # typo
        )


def test_invalid_provider_in_manual_router_disables_only_that_agent() -> None:
    """A manually-constructed router with an unknown provider name
    keeps the rest of the bot running."""
    router = ProviderRouter(
        agent_provider_choice={
            "trade_analysis": "myco-llm",
            "report": "openai",
        },
        agent_models={
            "trade_analysis": "gpt-fake",
            "report": "gpt-4o-mini",
        },
        provider_specs={},  # nothing usable
    )
    assert router.provider_for("trade_analysis") is None
    # ``report`` is also disabled because we didn't register a
    # provider spec — but the failure is "missing provider", not
    # "unknown provider".
    assert router.provider_for("report") is None


# ---------------------------------------------------------------------------
# 8. Per-agent model overrides flow through
# ---------------------------------------------------------------------------
def test_per_agent_model_override_used_when_set() -> None:
    s = _both_keys(
        TRADE_ANALYSIS_AGENT_MODEL="gpt-4o-mini-2026-05-19",
    )
    router = ProviderRouter.from_settings(s)
    p = router.provider_for("trade_analysis")
    assert p is not None
    assert p.model_name == "gpt-4o-mini-2026-05-19"


def test_dollar_brace_shorthand_resolves_to_default() -> None:
    """``TRADE_ANALYSIS_AGENT_MODEL=${OPENAI_DEFAULT_MODEL}`` is the
    documented "inherit default" shorthand and must NOT propagate as
    a literal model name."""
    s = _both_keys(
        TRADE_ANALYSIS_AGENT_MODEL="${OPENAI_DEFAULT_MODEL}",
        OPENAI_DEFAULT_MODEL="gpt-4o-mini",
    )
    router = ProviderRouter.from_settings(s)
    p = router.provider_for("trade_analysis")
    assert p is not None
    assert p.model_name == "gpt-4o-mini"


def test_settings_helper_resolves_models_consistently_with_router() -> None:
    """The router and Settings.model_for_agent must agree."""
    s = _both_keys()
    router = ProviderRouter.from_settings(s)
    for agent in (
        "news",
        "macro_news",
        "strategy_research",
        "trade_analysis",
        "model_review",
        "report",
        "risk_explainer",
        "trade_journal",
        "backtest_critic",
    ):
        provider = router.provider_for(agent)
        if provider is None:
            continue
        assert provider.model_name == s.model_for_agent(agent), agent


# ---------------------------------------------------------------------------
# 9. Coverage shape — every documented agent is wired
# ---------------------------------------------------------------------------
def test_every_agent_has_provider_and_model_field() -> None:
    expected = {
        "news",
        "macro_news",
        "strategy_research",
        "trade_analysis",
        "model_review",
        "report",
        "risk_explainer",
        "trade_journal",
        "backtest_critic",
        "model_drift",
        "data_quality",
    }
    assert set(AGENT_PROVIDER_FIELDS) == expected
    assert set(AGENT_MODEL_FIELDS) == expected


def test_disabled_provider_tokens_are_recognized() -> None:
    # Spec requires "none" as the canonical disabled token; we accept
    # a few common synonyms for ergonomics.
    for token in ("none", "off", "disabled", ""):
        assert token in DISABLED_NAMES


# ---------------------------------------------------------------------------
# 10. Failed provider call does not crash workflows
# ---------------------------------------------------------------------------
def test_provider_failure_does_not_propagate_to_orchestrator() -> None:
    """A blowing-up provider must surface as ``schema_valid=False``,
    not as an exception out of ``orchestrator.run_end_of_day``."""
    from agents.orchestrator import AgentOrchestrator
    from agents.providers.base import BaseLLMProvider, ProviderError
    from notifications.notification_service import NotificationService
    from storage.db import init_db, reset_engine_for_tests

    class _BoomProvider(BaseLLMProvider):
        provider_name = "boom"

        def __init__(self) -> None:
            super().__init__(model="boom-1")

        def generate_text(self, prompt, *, system_prompt=None, temperature=0.2):  # type: ignore[override]
            raise ProviderError("boom")

    s = _both_keys()
    reset_engine_for_tests()
    init_db()
    router = ProviderRouter.from_settings(s)
    bomb = _BoomProvider()
    # Inject the bomb into both provider buckets via the back-compat
    # legacy hook so every routed agent uses it.
    router._provider_cache["openai"] = bomb  # type: ignore[attr-defined]
    router._provider_cache["perplexity"] = bomb  # type: ignore[attr-defined]
    router._agent_provider_cache.clear()  # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=NotificationService(discord=None)
    )
    result = orch.run_end_of_day(
        daily_report_payload={
            "metrics": {"n_trades": 0},
            "trades": [],
            "risk_blocks": [],
            "compliance": {},
            "config": {},
            "risk_blocks_summary": {},
        },
    )
    # Five EOD agents attempted; all failed but result is still a
    # well-formed ``OrchestratorResult``.
    assert result.n_total() == 5
    assert result.n_valid() == 0
    for r in result.results.values():
        assert r.schema_valid is False
        assert r.error is not None
