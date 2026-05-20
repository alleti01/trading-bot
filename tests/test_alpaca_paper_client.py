"""Alpaca paper client safety + payload tests (no real network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from integrations.alpaca_paper_client import (
    AlpacaConfigurationError,
    AlpacaFuturesNotSupported,
    AlpacaPaperClient,
)
from integrations.broker_base import BrokerError


_PAPER_URL = "https://paper-api.alpaca.markets"


def _client(http: MagicMock, **overrides) -> AlpacaPaperClient:
    base = dict(
        api_key="key",
        secret_key="secret",
        base_url=_PAPER_URL,
        paper=True,
        enabled_symbols=["AAPL", "MSFT"],
        http_client=http,
    )
    base.update(overrides)
    return AlpacaPaperClient(**base)


def _stub_response(status_code: int, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_refuses_when_paper_false() -> None:
    with pytest.raises(AlpacaConfigurationError):
        AlpacaPaperClient(
            api_key="k",
            secret_key="s",
            base_url=_PAPER_URL,
            paper=False,
        )


def test_refuses_non_paper_url() -> None:
    with pytest.raises(AlpacaConfigurationError):
        AlpacaPaperClient(
            api_key="k",
            secret_key="s",
            base_url="https://api.alpaca.markets",
            paper=True,
        )


def test_refuses_missing_credentials() -> None:
    with pytest.raises(AlpacaConfigurationError):
        AlpacaPaperClient(
            api_key=None,
            secret_key="s",
            base_url=_PAPER_URL,
            paper=True,
        )


@pytest.mark.parametrize(
    "symbol", ["MES", "MNQ", "ES", "NQ", "MGC", "MCL", "MYM", "M2K"]
)
def test_blocks_futures_symbols(symbol: str) -> None:
    http = MagicMock()
    client = _client(http, enabled_symbols=[symbol])
    with pytest.raises(AlpacaFuturesNotSupported):
        client.place_market_order(symbol=symbol, qty=1, side="buy")


def test_validate_order_blocks_futures() -> None:
    http = MagicMock()
    client = _client(http, enabled_symbols=["MES"])
    result = client.validate_order(symbol="MES", qty=1, side="buy")
    assert not result.valid
    assert "futures" in (result.reason or "").lower()


def test_validate_order_blocks_disabled_symbol() -> None:
    http = MagicMock()
    client = _client(http, enabled_symbols=["AAPL"])
    result = client.validate_order(symbol="MSFT", qty=1, side="buy")
    assert not result.valid
    assert "not enabled" in (result.reason or "")


def test_validate_order_blocks_missing_account() -> None:
    http = MagicMock()
    http.get.side_effect = httpx.ConnectError("boom")
    client = _client(http)
    result = client.validate_order(symbol="AAPL", qty=1, side="buy")
    assert not result.valid
    assert "account_state_failed" in (result.reason or "")


def test_limit_order_payload_uses_paper_endpoint() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(
        200, {"id": "abc-1", "status": "accepted"}
    )
    client = _client(http)
    result = client.place_limit_order(
        symbol="AAPL", qty=10, side="buy", limit_price=180.5
    )
    assert result.success
    assert result.order_type == "limit"
    assert result.limit_price == 180.5
    args, kwargs = http.post.call_args
    assert args[0].startswith(_PAPER_URL)
    body = kwargs["json"]
    assert body["symbol"] == "AAPL"
    assert body["type"] == "limit"
    assert body["limit_price"] == 180.5
    assert kwargs["headers"]["APCA-API-KEY-ID"] == "key"
    assert kwargs["headers"]["APCA-API-SECRET-KEY"] == "secret"


def test_stop_order_payload_includes_stop_price() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(
        200, {"id": "stop-1", "status": "accepted"}
    )
    client = _client(http)
    result = client.place_stop_order(
        symbol="AAPL", qty=2, stop_price=175.0, side="sell"
    )
    assert result.success
    body = http.post.call_args.kwargs["json"]
    assert body["stop_price"] == 175.0
    assert body["type"] == "stop"


def test_market_order_blocked_when_quote_missing() -> None:
    http = MagicMock()
    http.get.side_effect = [
        _stub_response(200, {"id": "acct", "cash": "0"}),
        _stub_response(404, {"error": "no quote"}),
    ]
    client = _client(http)
    result = client.validate_order(symbol="AAPL", qty=1, side="buy")
    assert not result.valid
    assert "no_quote" in (result.reason or "")


def test_close_position_returns_structured_result() -> None:
    http = MagicMock()
    http.delete.return_value = _stub_response(
        200, {"id": "close-1", "qty": "5"}
    )
    client = _client(http)
    result = client.close_position(symbol="AAPL")
    assert result.success
    assert result.status == "filled"
    assert result.symbol == "AAPL"


def test_cancel_order_handles_404_gracefully() -> None:
    http = MagicMock()
    http.delete.return_value = _stub_response(404, {})
    client = _client(http)
    result = client.cancel_order(order_id="nope")
    assert result.status == "cancelled"


def test_broker_failure_surfaces_after_network_error() -> None:
    http = MagicMock()
    http.get.side_effect = httpx.ConnectError("boom")
    client = _client(http)
    with pytest.raises(BrokerError):
        client.get_positions()


def test_trailing_stop_requires_percent_kind() -> None:
    http = MagicMock()
    client = _client(http)
    with pytest.raises(BrokerError):
        client.place_trailing_stop(
            symbol="AAPL",
            qty=1,
            trail=5,
            side="sell",
            trail_kind="ticks",
        )
