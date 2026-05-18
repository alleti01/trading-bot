"""LLMClient implementations: Mock dispatch + builder gating."""

from __future__ import annotations

import json
import os

import pytest

from agents.llm_client import (
    LLMClientError,
    MockLLMClient,
    OpenAILLMClient,
    build_llm_client,
)
from agents.schemas import NewsAssessment


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


# ---------------------------------------------------------------------------
# MockLLMClient
# ---------------------------------------------------------------------------
def test_mock_returns_default_news_payload_parses_into_schema() -> None:
    mock = MockLLMClient()
    raw = mock.complete(system="agent=news\nx", user="agent=news\nbody")
    parsed = NewsAssessment.model_validate_json(raw)
    assert parsed.severity in ("low", "medium", "high")


def test_mock_dispatches_via_user_marker() -> None:
    mock = MockLLMClient()
    risk_raw = mock.complete(system="generic", user="agent=risk_explainer\nbody")
    payload = json.loads(risk_raw)
    assert "blocks" in payload
    assert "overall_assessment" in payload


def test_mock_falls_back_to_empty_object_for_unknown_agent() -> None:
    mock = MockLLMClient()
    raw = mock.complete(system="agent=unknown", user="body")
    assert raw == "{}"


def test_mock_can_simulate_failure_for_specific_agent() -> None:
    mock = MockLLMClient(raise_for_agents={"news"})
    with pytest.raises(LLMClientError):
        mock.complete(system="agent=news", user="agent=news\nbody")


def test_mock_records_calls() -> None:
    mock = MockLLMClient()
    mock.complete(system="s1", user="agent=news\nu1")
    mock.complete(system="s2", user="agent=report\nu2")
    assert len(mock.calls) == 2


def test_mock_responses_can_be_overridden() -> None:
    custom = json.dumps(
        {
            "high_risk_window": True,
            "severity": "medium",
            "events": ["custom-event"],
            "summary": "custom",
            "recommendation": "custom",
        }
    )
    mock = MockLLMClient(responses_by_agent={"news": custom})
    out = mock.complete(system="x", user="agent=news\ny")
    assert json.loads(out)["events"] == ["custom-event"]


# ---------------------------------------------------------------------------
# build_llm_client gating
# ---------------------------------------------------------------------------
def test_builder_returns_none_when_agents_disabled() -> None:
    s = _settings(ENABLE_LLM_AGENTS="false", OPENAI_API_KEY="anything")
    assert build_llm_client(s) is None


def test_builder_returns_none_when_no_api_key() -> None:
    s = _settings(ENABLE_LLM_AGENTS="true")
    # OPENAI_API_KEY left unset.
    assert build_llm_client(s) is None


def test_builder_returns_openai_client_when_configured() -> None:
    s = _settings(ENABLE_LLM_AGENTS="true", OPENAI_API_KEY="sk-test")
    client = build_llm_client(s)
    assert isinstance(client, OpenAILLMClient)
    assert client.model == s.LLM_MODEL


# ---------------------------------------------------------------------------
# OpenAILLMClient: just check construction guards (no network call here).
# ---------------------------------------------------------------------------
def test_openai_client_rejects_empty_api_key() -> None:
    with pytest.raises(ValueError):
        OpenAILLMClient("")
