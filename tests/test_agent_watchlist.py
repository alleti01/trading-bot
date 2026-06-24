"""Tests for the advisory agent watchlist (propose_watchlist_symbols)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.orchestrator import _extract_ticker_list


# ---------------------------------------------------------------------------
# Ticker parsing
# ---------------------------------------------------------------------------
def test_extract_strict_json() -> None:
    assert _extract_ticker_list('{"symbols": ["AAPL", "nvda"]}') == ["AAPL", "NVDA"]


def test_extract_bare_array() -> None:
    assert _extract_ticker_list('["SPY", "QQQ"]') == ["SPY", "QQQ"]


def test_extract_code_fenced_json() -> None:
    raw = "```json\n{\"symbols\": [\"MSFT\"]}\n```"
    assert _extract_ticker_list(raw) == ["MSFT"]


def test_extract_regex_fallback() -> None:
    out = _extract_ticker_list("I like AAPL and NVDA today")
    assert "AAPL" in out and "NVDA" in out


def test_extract_empty() -> None:
    assert _extract_ticker_list("") == []
    assert _extract_ticker_list("no tickers here lowercase") == []


# ---------------------------------------------------------------------------
# propose_watchlist_symbols (mocked client)
# ---------------------------------------------------------------------------
def _orchestrator_with_client(monkeypatch, response: str, *, raises: bool = False):
    from agents.orchestrator import AgentOrchestrator
    from agents.llm_client import LLMClientError
    from config.settings import reload_settings
    from notifications.notification_service import NotificationService

    settings = reload_settings()
    orch = AgentOrchestrator(
        settings, notifier=NotificationService(discord=None)
    )

    client = MagicMock()
    if raises:
        client.complete.side_effect = LLMClientError("boom")
    else:
        client.complete.return_value = response
    monkeypatch.setattr(orch, "_client_for", lambda name: client)
    return orch


def test_propose_watchlist_filters_to_allowlist(monkeypatch) -> None:
    orch = _orchestrator_with_client(
        monkeypatch, '{"symbols": ["AAPL", "FAKE", "NVDA"]}'
    )
    out = orch.propose_watchlist_symbols(allowlist=["AAPL", "NVDA", "MSFT"])
    assert out == ["AAPL", "NVDA"]
    assert "FAKE" not in out


def test_propose_watchlist_respects_max(monkeypatch) -> None:
    orch = _orchestrator_with_client(
        monkeypatch, '{"symbols": ["AAPL", "NVDA", "MSFT", "AMD"]}'
    )
    out = orch.propose_watchlist_symbols(
        allowlist=["AAPL", "NVDA", "MSFT", "AMD"], max_symbols=2
    )
    assert len(out) == 2


def test_propose_watchlist_empty_allowlist(monkeypatch) -> None:
    orch = _orchestrator_with_client(monkeypatch, '{"symbols": ["AAPL"]}')
    assert orch.propose_watchlist_symbols(allowlist=[]) == []


def test_propose_watchlist_llm_failure_returns_empty(monkeypatch) -> None:
    orch = _orchestrator_with_client(monkeypatch, "", raises=True)
    assert orch.propose_watchlist_symbols(allowlist=["AAPL"]) == []


def test_propose_watchlist_disabled_client(monkeypatch) -> None:
    from agents.orchestrator import AgentOrchestrator
    from config.settings import reload_settings
    from notifications.notification_service import NotificationService

    settings = reload_settings()
    orch = AgentOrchestrator(settings, notifier=NotificationService(discord=None))
    monkeypatch.setattr(orch, "_client_for", lambda name: None)
    assert orch.propose_watchlist_symbols(allowlist=["AAPL"]) == []


# ---------------------------------------------------------------------------
# Intraday loop caches the watchlist per session
# ---------------------------------------------------------------------------
def test_intraday_caches_watchlist(monkeypatch, tmp_path) -> None:
    from datetime import datetime, timezone

    from config.settings import reload_settings
    from workflows.intraday_loop import IntradayLoop

    monkeypatch.setenv("WORKFLOW_DYNAMIC_UNIVERSE", "true")
    monkeypatch.setenv("WORKFLOW_AGENT_WATCHLIST", "true")
    monkeypatch.setenv("ENABLED_SYMBOLS", "SPY")
    settings = reload_settings()

    calls = {"n": 0}

    def _fake_propose(_settings, *, orchestrator=None):
        calls["n"] += 1
        return ["AAPL", "NVDA"]

    monkeypatch.setattr("workflows.intraday_loop.propose_agent_watchlist", _fake_propose)
    loop = IntradayLoop(settings, dry_run=True, orchestrator=object())
    now = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    loop._universe(now=now)
    loop._universe(now=now)
    loop._universe(now=now)
    # Same session day → proposed once, then cached.
    assert calls["n"] == 1
