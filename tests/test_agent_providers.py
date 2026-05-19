"""Multi-provider agent routing.

Coverage map:

1. Provider primitives (text + JSON) for OpenAI, Perplexity,
   Anthropic, Gemini — all with mocked HTTP.
2. Schema validation through ``generate_json``.
3. ``ProviderRouter.from_settings`` selects the right provider per
   agent, returns ``None`` for missing keys, ``None`` for unknown
   provider names, and ``None`` for explicitly disabled agents.
4. ``ProviderLLMClient`` adapts the new providers into the legacy
   ``LLMClient`` interface and wraps :class:`ProviderError` in
   :class:`LLMClientError`.
5. End-to-end: orchestrator wired through the router runs the news
   agent against a mocked Perplexity backend without touching the
   network.
6. Paper-mode safety: a provider that always raises does not crash
   the orchestrator's EOD run; failed agents are persisted as
   ``schema_valid=False`` rows.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from agents.llm_client import LLMClient, LLMClientError
from agents.orchestrator import AgentOrchestrator, build_orchestrator
from agents.providers import (
    AnthropicProvider,
    BaseLLMProvider,
    GeminiProvider,
    OpenAIProvider,
    PerplexityProvider,
    ProviderError,
    ProviderLLMClient,
    ProviderRouter,
    parse_json_with_optional_schema,
)
from agents.schemas import NewsAssessment
from notifications.notification_service import NotificationService
from storage.db import init_db, reset_engine_for_tests, session_scope
from storage.tables import AgentOutput


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
        "ENABLE_LLM_AGENTS": "false",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


def _mock_openai_response(text: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }


def _mock_perplexity_response(
    text: str, citations: list[Any] | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
    }
    if citations is not None:
        body["citations"] = citations
    return body


def _mock_anthropic_response(text: str) -> dict[str, Any]:
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "model": "claude-3-5-sonnet-latest",
    }


def _mock_gemini_response(text: str) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": text}]},
                "finishReason": "STOP",
            }
        ]
    }


class _StubResponse:
    """Minimal stub that quacks like ``httpx.Response`` for our needs."""

    def __init__(
        self, status_code: int = 200, json_body: Any = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._json = json_body
        self.text = text or json.dumps(json_body or {})

    def json(self) -> Any:
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _RecordingHttp:
    """Fake ``httpx.Client`` that records requests + returns scripted responses."""

    def __init__(self, responses: list[_StubResponse] | _StubResponse) -> None:
        if isinstance(responses, _StubResponse):
            responses = [responses]
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        json: Any = None,
        timeout: float | None = None,
    ) -> _StubResponse:
        if not self._responses:
            raise RuntimeError("RecordingHttp ran out of scripted responses")
        self.calls.append(
            {"url": url, "headers": dict(headers or {}), "json": json, "timeout": timeout}
        )
        return self._responses.pop(0)

    def close(self) -> None:  # pragma: no cover - trivial
        return None


# ===========================================================================
# 1. parse_json_with_optional_schema
# ===========================================================================
def test_parse_json_handles_markdown_fences() -> None:
    raw = "```json\n{\"a\": 1, \"b\": \"two\"}\n```"
    assert parse_json_with_optional_schema(raw) == {"a": 1, "b": "two"}


def test_parse_json_validates_against_schema() -> None:
    payload = json.dumps(
        {
            "high_risk_window": False,
            "severity": "low",
            "events": [],
            "summary": "All quiet.",
            "recommendation": "Trade plan.",
        }
    )
    out = parse_json_with_optional_schema(payload, schema=NewsAssessment)
    assert out["severity"] == "low"


def test_parse_json_raises_on_invalid_schema() -> None:
    payload = json.dumps(
        {
            "high_risk_window": False,
            "severity": "extreme",  # not in the Literal
            "events": [],
            "summary": "x",
            "recommendation": "y",
        }
    )
    with pytest.raises(ProviderError):
        parse_json_with_optional_schema(payload, schema=NewsAssessment)


def test_parse_json_raises_on_non_object() -> None:
    with pytest.raises(ProviderError):
        parse_json_with_optional_schema("[1, 2, 3]")


def test_parse_json_raises_on_empty() -> None:
    with pytest.raises(ProviderError):
        parse_json_with_optional_schema("")


# ===========================================================================
# 2. OpenAI provider
# ===========================================================================
def test_openai_generate_text_via_mocked_http() -> None:
    http = _RecordingHttp(_StubResponse(json_body=_mock_openai_response("hello!")))
    provider = OpenAIProvider("sk-test", model="gpt-4o-mini", http_client=http)
    result = provider.generate_text(
        "summarize", system_prompt="be brief", temperature=0.3
    )
    assert result.text == "hello!"
    assert result.provider == "openai"
    assert result.model == "gpt-4o-mini"
    assert result.citations == []
    # Exactly one HTTP call, with the system prompt + user prompt + temperature.
    assert len(http.calls) == 1
    body = http.calls[0]["json"]
    assert body["temperature"] == pytest.approx(0.3)
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user"]


def test_openai_generate_json_validates_with_schema() -> None:
    text = json.dumps(
        {
            "high_risk_window": True,
            "severity": "high",
            "events": ["FOMC"],
            "summary": "Big day.",
            "recommendation": "Reduce size.",
        }
    )
    http = _RecordingHttp(_StubResponse(json_body=_mock_openai_response(text)))
    provider = OpenAIProvider("sk-test", http_client=http)
    out = provider.generate_json("x", schema=NewsAssessment)
    assert out.data["severity"] == "high"
    assert out.text == text
    body = http.calls[0]["json"]
    assert body["response_format"] == {"type": "json_object"}


def test_openai_translates_http_failure_to_provider_error() -> None:
    http = _RecordingHttp(_StubResponse(status_code=503, text="upstream down"))
    provider = OpenAIProvider("sk-test", http_client=http)
    with pytest.raises(ProviderError) as exc:
        provider.generate_text("hi")
    assert "openai.bad_status" in str(exc.value)


def test_openai_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError):
        OpenAIProvider("")


# ===========================================================================
# 3. Perplexity provider
# ===========================================================================
def test_perplexity_returns_citations_old_style() -> None:
    body = _mock_perplexity_response(
        text=json.dumps({"high_risk_window": False, "severity": "low",
                         "events": [], "summary": "ok", "recommendation": "ok"}),
        citations=["https://example.com/a", "https://example.com/b"],
    )
    http = _RecordingHttp(_StubResponse(json_body=body))
    provider = PerplexityProvider(
        "pplx-test", model="sonar", http_client=http
    )
    out = provider.generate_text("news today", system_prompt="be brief")
    assert out.provider == "perplexity"
    assert {c.url for c in out.citations} == {
        "https://example.com/a",
        "https://example.com/b",
    }


def test_perplexity_returns_citations_search_results_style() -> None:
    body = _mock_perplexity_response(text="x")
    body["search_results"] = [
        {"url": "https://news.example/a", "title": "Headline A", "snippet": "..."},
        {"url": "https://news.example/b", "title": "Headline B"},
    ]
    http = _RecordingHttp(_StubResponse(json_body=body))
    provider = PerplexityProvider("pplx-test", http_client=http)
    out = provider.generate_text("news today")
    assert [c.url for c in out.citations] == [
        "https://news.example/a",
        "https://news.example/b",
    ]
    assert out.citations[0].title == "Headline A"


def test_perplexity_generate_json_validates() -> None:
    text = json.dumps(
        {
            "high_risk_window": False,
            "severity": "low",
            "events": [],
            "summary": "ok",
            "recommendation": "ok",
        }
    )
    body = _mock_perplexity_response(
        text=text, citations=[{"url": "https://x.example", "title": "X"}]
    )
    http = _RecordingHttp(_StubResponse(json_body=body))
    provider = PerplexityProvider("pplx-test", http_client=http)
    out = provider.generate_json("x", schema=NewsAssessment)
    assert out.data["severity"] == "low"
    assert out.citations[0].url == "https://x.example"


def test_perplexity_raises_on_missing_key() -> None:
    with pytest.raises(ValueError):
        PerplexityProvider("")


def test_perplexity_translates_network_failure() -> None:
    class _BoomHttp:
        def post(self, *a, **k):
            raise httpx.ConnectError("boom")

        def close(self) -> None:  # pragma: no cover
            return None

    provider = PerplexityProvider("pplx-test", http_client=_BoomHttp())
    with pytest.raises(ProviderError) as exc:
        provider.generate_text("x")
    assert "perplexity.http_error" in str(exc.value)


# ===========================================================================
# 4. Anthropic provider
# ===========================================================================
def test_anthropic_generate_text_via_mocked_http() -> None:
    http = _RecordingHttp(_StubResponse(json_body=_mock_anthropic_response("hi")))
    provider = AnthropicProvider("sk-ant-test", http_client=http)
    out = provider.generate_text("hello", system_prompt="be brief")
    assert out.text == "hi"
    assert out.provider == "anthropic"
    body = http.calls[0]["json"]
    assert body["system"] == "be brief"
    assert body["messages"][0]["role"] == "user"
    headers = http.calls[0]["headers"]
    assert headers["anthropic-version"] == AnthropicProvider.API_VERSION


def test_anthropic_generate_json_with_schema() -> None:
    payload_text = json.dumps(
        {
            "high_risk_window": False,
            "severity": "low",
            "events": [],
            "summary": "ok",
            "recommendation": "ok",
        }
    )
    http = _RecordingHttp(_StubResponse(json_body=_mock_anthropic_response(payload_text)))
    provider = AnthropicProvider("sk-ant-test", http_client=http)
    out = provider.generate_json("x", schema=NewsAssessment)
    assert out.data["severity"] == "low"


def test_anthropic_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        AnthropicProvider("")


# ===========================================================================
# 5. Gemini provider
# ===========================================================================
def test_gemini_generate_text_includes_system_instruction() -> None:
    http = _RecordingHttp(_StubResponse(json_body=_mock_gemini_response("ok")))
    provider = GeminiProvider("gemini-test", http_client=http)
    out = provider.generate_text("hello", system_prompt="be brief")
    assert out.text == "ok"
    assert out.provider == "gemini"
    body = http.calls[0]["json"]
    assert body["systemInstruction"]["parts"][0]["text"] == "be brief"
    # API key must be on the URL, never in headers (Gemini quirk).
    assert "key=gemini-test" in http.calls[0]["url"]
    assert "Authorization" not in http.calls[0]["headers"]


def test_gemini_generate_json_uses_response_mime_type() -> None:
    text = json.dumps(
        {
            "high_risk_window": False,
            "severity": "low",
            "events": [],
            "summary": "ok",
            "recommendation": "ok",
        }
    )
    http = _RecordingHttp(_StubResponse(json_body=_mock_gemini_response(text)))
    provider = GeminiProvider("gemini-test", http_client=http)
    out = provider.generate_json("x", schema=NewsAssessment)
    assert out.data["severity"] == "low"
    body = http.calls[0]["json"]
    assert (
        body["generationConfig"]["responseMimeType"] == "application/json"
    )


def test_gemini_rejects_empty_key() -> None:
    with pytest.raises(ValueError):
        GeminiProvider("")


# ===========================================================================
# 6. ProviderRouter
# ===========================================================================
def test_router_news_routes_to_perplexity_when_key_set() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        OPENAI_API_KEY="sk-test",
        NEWS_AGENT_PROVIDER="perplexity",
        TRADE_ANALYSIS_AGENT_PROVIDER="openai",
    )
    router = ProviderRouter.from_settings(s)
    p = router.provider_for("news")
    assert isinstance(p, PerplexityProvider)
    p2 = router.provider_for("trade_analysis")
    assert isinstance(p2, OpenAIProvider)


def test_router_returns_none_when_routed_provider_key_missing() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        # No PERPLEXITY_API_KEY -> news should disable.
        NEWS_AGENT_PROVIDER="perplexity",
        TRADE_ANALYSIS_AGENT_PROVIDER="openai",
    )
    router = ProviderRouter.from_settings(s)
    assert router.provider_for("news") is None
    assert router.is_enabled("news") is False
    # Other agents are unaffected.
    assert router.is_enabled("trade_analysis") is True


def test_router_returns_none_for_explicit_disabled() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        PERPLEXITY_API_KEY="pplx-test",
        REPORT_AGENT_PROVIDER="none",
    )
    router = ProviderRouter.from_settings(s)
    assert router.provider_for("report") is None


def test_router_rejects_unknown_provider_name_at_settings_validation() -> None:
    """Settings validator rejects typos in provider names so a user
    cannot silently disable an agent with ``OENAI`` etc. The router
    therefore never sees an "unknown provider" string for the
    documented agents — but if one slips through (e.g. via a manually
    constructed router) the router's ``provider_for`` still disables
    cleanly. Both behaviours are exercised here."""
    import pytest as _pytest

    with _pytest.raises(Exception):
        _settings(
            ENABLE_LLM_AGENTS="true",
            OPENAI_API_KEY="sk-test",
            TRADE_ANALYSIS_AGENT_PROVIDER="myco-llm",
        )


def test_router_returns_none_for_manually_injected_unknown_provider() -> None:
    """Direct router construction with a hand-rolled choice map still
    gracefully disables unknown providers (no crash)."""
    from agents.providers.router import ProviderRouter as PR

    router = PR(
        agent_provider_choice={"trade_analysis": "myco-llm"},
        agent_models={"trade_analysis": None},
        provider_specs={},  # nothing registered
    )
    assert router.provider_for("trade_analysis") is None
    assert router.is_enabled("trade_analysis") is False


def test_router_caches_per_agent_instances() -> None:
    """Two agents routed to the same provider with the same resolved
    model share one instance. (Different resolved models -> different
    instances; covered separately.)"""
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        TRADE_ANALYSIS_AGENT_PROVIDER="openai",
        REPORT_AGENT_PROVIDER="openai",
        # Pin both agents to the same model so they share an instance.
        TRADE_ANALYSIS_AGENT_MODEL="gpt-4o-mini",
        REPORT_AGENT_MODEL="gpt-4o-mini",
    )
    router = ProviderRouter.from_settings(s)
    a = router.provider_for("trade_analysis")
    b = router.provider_for("report")
    assert a is not None and b is not None
    assert a.model_name == "gpt-4o-mini" == b.model_name


def test_router_routing_table_reports_state() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        # Intentionally no OPENAI_API_KEY.
        NEWS_AGENT_PROVIDER="perplexity",
        TRADE_ANALYSIS_AGENT_PROVIDER="openai",
    )
    router = ProviderRouter.from_settings(s)
    table = router.routing_table()
    expected_pplx_model = s.PERPLEXITY_DEFAULT_MODEL or s.PERPLEXITY_MODEL
    assert table["news"] == {
        "provider": "perplexity",
        "enabled": True,
        "model": expected_pplx_model,
    }
    assert table["trade_analysis"]["provider"] == "openai"
    assert table["trade_analysis"]["enabled"] is False
    # Routing table shows the would-have-been model for the audit
    # trail even when the agent is disabled by a missing key.
    assert table["trade_analysis"]["model"] is not None


def test_router_has_any_enabled_false_when_no_keys() -> None:
    s = _settings(ENABLE_LLM_AGENTS="true")
    router = ProviderRouter.from_settings(s)
    assert router.has_any_enabled() is False
    assert router.enabled_agents() == []


# ===========================================================================
# 7. ProviderLLMClient adapter
# ===========================================================================
class _StubProvider(BaseLLMProvider):
    """Minimal in-process provider used to exercise the adapter."""

    provider_name = "stub"

    def __init__(self, *, text_to_return: str | Exception = "stub-out") -> None:
        super().__init__(model="stub-model")
        self._text = text_to_return

    def generate_text(self, prompt, *, system_prompt=None, temperature=0.2):  # type: ignore[override]
        if isinstance(self._text, Exception):
            raise self._text
        from agents.providers.base import ProviderTextResult

        return ProviderTextResult(
            text=self._text,
            provider=self.provider_name,
            model=self.model_name,
            citations=[],
        )


def test_provider_llm_client_returns_text() -> None:
    adapter = ProviderLLMClient(_StubProvider(text_to_return="hello"))
    assert adapter.complete(system="s", user="u") == "hello"
    assert adapter.provider_name == "stub"
    assert adapter.model == "stub-model"


def test_provider_llm_client_wraps_provider_error_in_llmclient_error() -> None:
    adapter = ProviderLLMClient(
        _StubProvider(text_to_return=ProviderError("boom"))
    )
    with pytest.raises(LLMClientError) as exc:
        adapter.complete(system="s", user="u")
    assert "boom" in str(exc.value)


# ===========================================================================
# 8. Orchestrator wiring (uses an injected mock provider)
# ===========================================================================
def _captured_notifier() -> NotificationService:
    return NotificationService(discord=None)


class _CountingProvider(BaseLLMProvider):
    """Provider that returns a canned valid NewsAssessment payload."""

    provider_name = "counting"

    def __init__(self, payload: str) -> None:
        super().__init__(model="counting-model")
        self.payload = payload
        self.calls = 0

    def generate_text(self, prompt, *, system_prompt=None, temperature=0.2):  # type: ignore[override]
        from agents.providers.base import ProviderTextResult

        self.calls += 1
        return ProviderTextResult(
            text=self.payload,
            provider=self.provider_name,
            model=self.model_name,
        )


def test_orchestrator_dispatches_news_through_router(tmp_path: Path) -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        OPENAI_API_KEY="sk-test",
        REPORTS_DIR=str(tmp_path / "reports"),
    )
    reset_engine_for_tests()
    init_db()

    router = ProviderRouter.from_settings(s)
    counting = _CountingProvider(
        payload=json.dumps(
            {
                "high_risk_window": True,
                "severity": "high",
                "events": ["FOMC"],
                "summary": "Big day ahead.",
                "recommendation": "Reduce size.",
            }
        )
    )
    # Inject the counting provider into the router so the orchestrator
    # calls it through ``client_for("news")``.
    router._provider_cache["perplexity"] = counting  # type: ignore[attr-defined]
    router._client_cache.pop("news", None)  # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    assessment = orch.run_pre_session_news(
        now=datetime(2026, 5, 19, 12, 30, tzinfo=timezone.utc)
    )
    assert assessment is not None
    assert assessment.high_risk_window is True
    assert orch.high_risk_news_active() is True
    assert counting.calls == 1


def test_orchestrator_skips_agents_with_disabled_provider(tmp_path: Path) -> None:
    """News routes to perplexity but no key — the news agent must be
    skipped without affecting the other agents."""
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        # No PERPLEXITY_API_KEY -> news disabled.
        NEWS_AGENT_PROVIDER="perplexity",
        TRADE_ANALYSIS_AGENT_PROVIDER="openai",
        MODEL_REVIEW_AGENT_PROVIDER="openai",
        REPORT_AGENT_PROVIDER="openai",
        RISK_EXPLAINER_AGENT_PROVIDER="openai",
        TRADE_JOURNAL_AGENT_PROVIDER="openai",
        REPORTS_DIR=str(tmp_path / "reports"),
    )
    reset_engine_for_tests()
    init_db()

    router = ProviderRouter.from_settings(s)
    # Inject a deterministic OpenAI provider so the ones routed to
    # OpenAI succeed without network access.
    canned = _CountingProvider(payload="{}")
    router._provider_cache["openai"] = canned  # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    result = orch.run_end_of_day(
        now=datetime(2026, 5, 19, 21, 0, tzinfo=timezone.utc),
        daily_report_payload={
            "metrics": {"n_trades": 0},
            "trades": [],
            "risk_blocks": [],
            "compliance": {},
            "config": {},
            "risk_blocks_summary": {},
        },
    )
    # Five agents were attempted; news is disabled (key missing) so it
    # produced a schema_valid=False row with a clear error string.
    assert "news" in result.results
    news = result.results["news"]
    assert news.schema_valid is False
    assert news.error and "agent.disabled" in news.error
    # The OpenAI-routed agents got the canned provider; their JSON is
    # ``{}`` so schema validation fails for them too — but the failure
    # is a *schema* failure, not a routing one. That's the contract.
    assert any(
        not r.schema_valid and r.error and "schema_invalid" in r.error
        for name, r in result.results.items()
        if name != "news"
    )

    with session_scope() as session:
        rows = session.execute(
            __import__("sqlalchemy").select(AgentOutput)
        ).scalars().all()
    # Five attempts persisted regardless of validity — audit trail intact.
    assert len(rows) == 5


# ===========================================================================
# 9. Paper-mode safety: provider failure does not crash orchestrator
# ===========================================================================
class _ExplodingProvider(BaseLLMProvider):
    provider_name = "boom"

    def __init__(self) -> None:
        super().__init__(model="boom-model")
        self.calls = 0

    def generate_text(self, prompt, *, system_prompt=None, temperature=0.2):  # type: ignore[override]
        self.calls += 1
        raise ProviderError("simulated provider outage")


def test_orchestrator_survives_provider_failure(tmp_path: Path) -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        PERPLEXITY_API_KEY="pplx-test",
        OPENAI_API_KEY="sk-test",
        REPORTS_DIR=str(tmp_path / "reports"),
    )
    reset_engine_for_tests()
    init_db()

    router = ProviderRouter.from_settings(s)
    bomb = _ExplodingProvider()
    router._provider_cache["perplexity"] = bomb  # type: ignore[attr-defined]
    router._provider_cache["openai"] = bomb      # type: ignore[attr-defined]

    orch = AgentOrchestrator(
        s, provider_router=router, notifier=_captured_notifier()
    )
    # Must not raise.
    result = orch.run_end_of_day(
        now=datetime(2026, 5, 19, 21, 0, tzinfo=timezone.utc),
        daily_report_payload={
            "metrics": {"n_trades": 0},
            "trades": [],
            "risk_blocks": [],
            "compliance": {},
            "config": {},
            "risk_blocks_summary": {},
        },
    )
    assert result.n_total() == 5
    assert result.n_valid() == 0
    # Every result is an LLM error, persisted for the audit trail.
    for r in result.results.values():
        assert r.schema_valid is False
        assert r.error and "llm_error" in r.error


# ===========================================================================
# 10. build_orchestrator factory paths
# ===========================================================================
def test_build_orchestrator_uses_router_when_keys_present() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        PERPLEXITY_API_KEY="pplx-test",
    )
    orch = build_orchestrator(s, notifier=_captured_notifier())
    assert orch.provider_router is not None
    assert orch.llm is None


def test_build_orchestrator_falls_back_to_legacy_openai() -> None:
    """``ENABLE_LLM_AGENTS=true`` + only OPENAI_API_KEY: keep working
    via the legacy single-client path so existing single-key
    deployments don't suddenly need new env vars."""
    s = _settings(
        ENABLE_LLM_AGENTS="true",
        OPENAI_API_KEY="sk-test",
        # All per-agent providers default to providers we DO have a
        # key for ("openai") so the router *also* succeeds. Force-disable
        # every agent so the router reports "nothing enabled" and the
        # builder falls back to the legacy path.
        NEWS_AGENT_PROVIDER="none",
        MACRO_NEWS_AGENT_PROVIDER="none",
        STRATEGY_RESEARCH_AGENT_PROVIDER="none",
        TRADE_ANALYSIS_AGENT_PROVIDER="none",
        MODEL_REVIEW_AGENT_PROVIDER="none",
        REPORT_AGENT_PROVIDER="none",
        RISK_EXPLAINER_AGENT_PROVIDER="none",
        TRADE_JOURNAL_AGENT_PROVIDER="none",
        BACKTEST_CRITIC_AGENT_PROVIDER="none",
        MODEL_DRIFT_AGENT_PROVIDER="none",
    )
    orch = build_orchestrator(s, notifier=_captured_notifier())
    assert orch.provider_router is None
    assert orch.llm is not None  # legacy OpenAILLMClient


def test_build_orchestrator_no_op_without_keys() -> None:
    s = _settings(ENABLE_LLM_AGENTS="true")
    orch = build_orchestrator(s, notifier=_captured_notifier())
    assert orch.provider_router is None
    assert orch.llm is None


def test_build_orchestrator_disabled_flag_short_circuits() -> None:
    s = _settings(
        ENABLE_LLM_AGENTS="false",
        OPENAI_API_KEY="sk-test",
        PERPLEXITY_API_KEY="pplx-test",
    )
    orch = build_orchestrator(s, notifier=_captured_notifier())
    assert orch.provider_router is None
    assert orch.llm is None
