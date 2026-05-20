"""Broker router + adapter safety tests (DRY_RUN/PAPER/LIVE rails)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from config.settings import reload_settings
from integrations import (
    BrokerError,
    BrokerRouter,
    LiveExecutionRefused,
    MockBroker,
    SUPPORTED_SYMBOLS,
    TradovateConfigurationError,
    build_broker,
)
from integrations.broker_base import OrderResult


def _settings(monkeypatch: pytest.MonkeyPatch, **overrides: str):
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "DRY_RUN")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("ENABLED_SYMBOLS", "MES,MNQ")
    monkeypatch.setenv("TRADOVATE_DEMO", "true")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_dry_run_always_returns_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, BROKER_PROVIDER="tradovate")
    broker = build_broker(settings)
    assert isinstance(broker, MockBroker)


def test_live_execution_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, WORKFLOW_EXECUTION_MODE="LIVE")
    with pytest.raises(LiveExecutionRefused):
        build_broker(settings)


def test_paper_with_mock_provider_returns_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, WORKFLOW_EXECUTION_MODE="PAPER")
    broker = build_broker(settings)
    assert isinstance(broker, MockBroker)


def test_paper_tradovate_requires_demo(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(
        monkeypatch,
        WORKFLOW_EXECUTION_MODE="PAPER",
        BROKER_PROVIDER="tradovate",
        TRADOVATE_DEMO="false",
        TRADOVATE_USERNAME="u",
        TRADOVATE_PASSWORD="p",
        TRADOVATE_APP_ID="app",
    )
    with pytest.raises(TradovateConfigurationError):
        build_broker(settings)


def test_paper_tradovate_requires_demo_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        WORKFLOW_EXECUTION_MODE="PAPER",
        BROKER_PROVIDER="tradovate",
        TRADOVATE_BASE_URL="https://live.tradovateapi.com/v1",
        TRADOVATE_USERNAME="u",
        TRADOVATE_PASSWORD="p",
        TRADOVATE_APP_ID="app",
    )
    with pytest.raises(TradovateConfigurationError):
        build_broker(settings)


def test_paper_tradovate_requires_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(
        monkeypatch,
        WORKFLOW_EXECUTION_MODE="PAPER",
        BROKER_PROVIDER="tradovate",
    )
    with pytest.raises(TradovateConfigurationError):
        build_broker(settings)


def test_router_is_live_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch, WORKFLOW_EXECUTION_MODE="LIVE")
    assert BrokerRouter(settings).is_live_locked() is True


def test_validate_order_blocks_unsupported_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    result = broker.validate_order(symbol="AAPL", qty=1, side="buy")
    assert not result.valid
    assert "unsupported" in (result.reason or "").lower()


def test_validate_order_blocks_disabled_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch, ENABLED_SYMBOLS="MES")
    broker = build_broker(settings)
    result = broker.validate_order(symbol="MNQ", qty=1, side="buy")
    assert not result.valid
    assert "not enabled" in (result.reason or "")


def test_validate_order_blocks_missing_quote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    broker = MockBroker(enabled_symbols=["MES"])
    broker._quotes.pop("MES")  # noqa: SLF001 — testing missing-quote path
    result = broker.validate_order(symbol="MES", qty=1, side="buy")
    assert not result.valid
    assert "no quote" in (result.reason or "").lower()


def test_limit_order_payload_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    result = broker.place_limit_order(
        symbol="MES", qty=1, side="buy", limit_price=4990.0
    )
    assert isinstance(result, OrderResult)
    assert result.success
    assert result.order_type == "limit"
    assert result.limit_price == 4990.0
    assert result.simulated is True
    payload = result.to_payload()
    assert payload["order_type"] == "limit"
    assert payload["symbol"] == "MES"


def test_stop_order_payload_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    result = broker.place_stop_order(
        symbol="MES", qty=1, stop_price=4970.0, side="sell"
    )
    assert result.success
    assert result.order_type == "stop"
    assert result.stop_price == 4970.0


def test_market_order_returns_filled_in_mock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    result = broker.place_market_order(symbol="MES", qty=1, side="buy")
    assert result.success
    assert result.status == "filled"
    assert result.fill_price is not None


def test_supported_symbol_set_matches_spec() -> None:
    expected = {"MES", "MNQ", "ES", "NQ", "MGC", "MCL", "MYM", "M2K"}
    assert set(SUPPORTED_SYMBOLS) == expected


def test_broker_failure_blocks_workflow_trading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``reconcile`` fails the workflow path must surface a BrokerError."""
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    bad = MagicMock(side_effect=RuntimeError("network down"))
    broker.get_positions = bad  # type: ignore[assignment]
    with pytest.raises(BrokerError):
        broker.reconcile()


def test_discord_failure_does_not_crash_broker_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adapter does not import the notifier; a Discord outage cannot crash it."""
    settings = _settings(monkeypatch)
    broker = build_broker(settings)
    result = broker.place_limit_order(
        symbol="MES", qty=1, side="buy", limit_price=4990.0
    )
    assert result.success


def test_tradingview_not_required_for_tradovate_paper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Building a broker must not require any TradingView config."""
    monkeypatch.delenv("TRADINGVIEW_WEBHOOK_SECRET", raising=False)
    settings = _settings(monkeypatch, WORKFLOW_EXECUTION_MODE="PAPER")
    broker = build_broker(settings)
    assert broker.provider_name == "mock"
