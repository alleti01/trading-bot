"""Schema-mismatch and bad-JSON paths must never raise out of an agent."""

from __future__ import annotations

import os

from agents.base_agent import AgentContext
from agents.llm_client import LLMClient, LLMClientError
from agents.news_agent import NewsAgent


def _settings(**overrides):
    from config.settings import reload_settings

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


class _FixedClient(LLMClient):
    """Returns a hard-coded response — used to drive parse / validation failures."""

    def __init__(self, response: str = "", *, raise_with: Exception | None = None) -> None:
        self.response = response
        self.raise_with = raise_with

    def complete(self, *, system: str, user: str) -> str:
        if self.raise_with is not None:
            raise self.raise_with
        return self.response


def _ctx() -> AgentContext:
    return AgentContext(
        settings_snapshot={
            "INSTRUMENT": "MES",
            "MARKET_TYPE": "futures",
            "trading_window_start": "09:30:00",
            "trading_window_end": "15:55:00",
        },
        session_date="2026-05-18",
        instrument="MES",
    )


def test_invalid_json_returns_schema_invalid_without_raising() -> None:
    _settings()
    agent = NewsAgent(_FixedClient("not json at all"))
    result = agent.run(_ctx())
    assert result.schema_valid is False
    assert result.payload is None
    assert result.raw_text == "not json at all"
    assert (result.error or "").startswith(("schema_invalid", "json_invalid"))


def test_missing_required_field_returns_schema_invalid() -> None:
    _settings()
    bad = '{"high_risk_window": true, "severity": "low", "events": []}'
    agent = NewsAgent(_FixedClient(bad))
    result = agent.run(_ctx())
    assert result.schema_valid is False
    assert "schema_invalid" in (result.error or "")


def test_extra_field_rejected_by_strict_schema() -> None:
    _settings()
    bad = (
        '{"high_risk_window": false, "severity": "low", "events": [], '
        '"summary": "x", "recommendation": "y", "extra": 1}'
    )
    agent = NewsAgent(_FixedClient(bad))
    result = agent.run(_ctx())
    assert result.schema_valid is False


def test_markdown_fence_around_json_is_stripped_and_parsed() -> None:
    _settings()
    fenced = (
        "```json\n"
        '{"high_risk_window": false, "severity": "low", '
        '"events": [], "summary": "x", "recommendation": "y"}\n'
        "```"
    )
    agent = NewsAgent(_FixedClient(fenced))
    result = agent.run(_ctx())
    assert result.schema_valid is True
    assert result.payload is not None
    assert result.payload["severity"] == "low"


def test_llm_client_error_is_caught() -> None:
    _settings()
    agent = NewsAgent(_FixedClient(raise_with=LLMClientError("boom")))
    result = agent.run(_ctx())
    assert result.schema_valid is False
    assert "llm_error" in (result.error or "")


def test_unexpected_exception_in_complete_is_caught() -> None:
    _settings()
    agent = NewsAgent(_FixedClient(raise_with=RuntimeError("kaboom")))
    result = agent.run(_ctx())
    assert result.schema_valid is False
    assert "llm_unexpected" in (result.error or "")
