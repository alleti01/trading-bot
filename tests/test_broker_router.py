"""Broker router selection + provider-specific safety rails."""

from __future__ import annotations

import pytest

from config.settings import Settings, reload_settings
from integrations import (
    AlpacaConfigurationError,
    AlpacaPaperClient,
    BrokerRouter,
    InvalidBrokerProviderError,
    LiveExecutionRefused,
    MockBroker,
    TradovateConfigurationError,
    TradovateDemoClient,
    build_broker,
)


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("ENABLED_SYMBOLS", "AAPL")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    monkeypatch.setenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    monkeypatch.setenv("TRADOVATE_DEMO", "true")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------
def test_router_selects_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    assert isinstance(broker, MockBroker)


def test_router_selects_alpaca(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="alpaca",
        ALPACA_API_KEY="key",
        ALPACA_SECRET_KEY="secret",
    )
    broker = build_broker(settings)
    assert isinstance(broker, AlpacaPaperClient)
    assert broker.provider_name == "alpaca"


def test_router_selects_tradovate(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="tradovate",
        ENABLED_SYMBOLS="MES",
        TRADOVATE_USERNAME="u",
        TRADOVATE_PASSWORD="p",
        TRADOVATE_APP_ID="app",
    )
    broker = build_broker(settings)
    assert isinstance(broker, TradovateDemoClient)
    assert broker.provider_name == "tradovate"


# ---------------------------------------------------------------------------
# Missing credentials
# ---------------------------------------------------------------------------
def test_missing_alpaca_keys_blocks_alpaca(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="alpaca",
    )
    with pytest.raises(AlpacaConfigurationError):
        build_broker(settings)


def test_missing_tradovate_keys_blocks_tradovate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="tradovate",
        ENABLED_SYMBOLS="MES",
    )
    with pytest.raises(TradovateConfigurationError):
        build_broker(settings)


# ---------------------------------------------------------------------------
# Per-provider safety refusals
# ---------------------------------------------------------------------------
def test_alpaca_refuses_live_url(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="alpaca",
        ALPACA_API_KEY="k",
        ALPACA_SECRET_KEY="s",
        ALPACA_BASE_URL="https://api.alpaca.markets",
    )
    with pytest.raises(AlpacaConfigurationError):
        build_broker(settings)


def test_alpaca_refuses_paper_false(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="alpaca",
        ALPACA_API_KEY="k",
        ALPACA_SECRET_KEY="s",
        ALPACA_PAPER="false",
    )
    with pytest.raises(AlpacaConfigurationError):
        build_broker(settings)


def test_tradovate_refuses_live_url(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        BROKER_PROVIDER="tradovate",
        TRADOVATE_USERNAME="u",
        TRADOVATE_PASSWORD="p",
        TRADOVATE_APP_ID="app",
        TRADOVATE_BASE_URL="https://live.tradovateapi.com/v1",
    )
    with pytest.raises(TradovateConfigurationError):
        build_broker(settings)


# ---------------------------------------------------------------------------
# Live mode + invalid provider
# ---------------------------------------------------------------------------
def test_live_mode_refuses_for_all_brokers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for provider in ("mock", "alpaca", "tradovate"):
        settings = _settings(
            monkeypatch,
            WORKFLOW_EXECUTION_MODE="LIVE",
            BROKER_PROVIDER=provider,
            ALPACA_API_KEY="k",
            ALPACA_SECRET_KEY="s",
            TRADOVATE_USERNAME="u",
            TRADOVATE_PASSWORD="p",
            TRADOVATE_APP_ID="app",
        )
        with pytest.raises(LiveExecutionRefused):
            build_broker(settings)


def test_invalid_provider_blocked_at_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BROKER_PROVIDER", "robinhood")
    with pytest.raises(Exception):
        reload_settings()


def test_router_rejects_manually_injected_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    settings.BROKER_PROVIDER = "robinhood"  # type: ignore[assignment]
    with pytest.raises(InvalidBrokerProviderError):
        BrokerRouter(settings).for_execution_mode()


# ---------------------------------------------------------------------------
# DRY_RUN forces mock regardless of provider
# ---------------------------------------------------------------------------
def test_dry_run_forces_mock_for_alpaca(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        WORKFLOW_EXECUTION_MODE="DRY_RUN",
        BROKER_PROVIDER="alpaca",
    )
    broker = build_broker(settings)
    assert isinstance(broker, MockBroker)


def test_dry_run_forces_mock_for_tradovate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        WORKFLOW_EXECUTION_MODE="DRY_RUN",
        BROKER_PROVIDER="tradovate",
    )
    broker = build_broker(settings)
    assert isinstance(broker, MockBroker)
