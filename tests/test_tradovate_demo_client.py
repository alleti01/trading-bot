"""Tradovate demo client safety + payload tests (no real network calls)."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from integrations.broker_base import BrokerError, redact_secrets
from integrations.tradovate_demo_client import (
    TradovateConfigurationError,
    TradovateDemoClient,
)


_DEMO_URL = "https://demo.tradovateapi.com/v1"


def _client(http: MagicMock, **overrides) -> TradovateDemoClient:
    base = dict(
        base_url=_DEMO_URL,
        username="demo-user",
        password="demo-pass",
        app_id="app",
        app_version="1.0.0",
        client_id="cid",
        client_secret="csecret",
        demo=True,
        enabled_symbols=["MES", "MNQ"],
        http_client=http,
    )
    base.update(overrides)
    return TradovateDemoClient(**base)


def _stub_response(status_code: int, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_refuses_when_demo_false() -> None:
    with pytest.raises(TradovateConfigurationError):
        TradovateDemoClient(
            base_url=_DEMO_URL,
            username="u",
            password="p",
            app_id="app",
            app_version="1.0.0",
            client_id="c",
            client_secret="s",
            demo=False,
        )


def test_refuses_non_demo_url() -> None:
    with pytest.raises(TradovateConfigurationError):
        TradovateDemoClient(
            base_url="https://live.tradovateapi.com/v1",
            username="u",
            password="p",
            app_id="app",
            app_version="1.0.0",
            client_id="c",
            client_secret="s",
            demo=True,
        )


def test_refuses_missing_credentials() -> None:
    with pytest.raises(TradovateConfigurationError):
        TradovateDemoClient(
            base_url=_DEMO_URL,
            username=None,
            password="p",
            app_id="app",
            app_version="1.0.0",
            client_id="c",
            client_secret="s",
            demo=True,
        )


def test_auth_error_propagates_as_broker_error() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(401, {"errorText": "bad creds"})
    client = _client(http)
    with pytest.raises(BrokerError):
        client.get_account()


def test_limit_order_payload_uses_demo_endpoint() -> None:
    http = MagicMock()
    http.post.side_effect = [
        _stub_response(200, {"accessToken": "tok", "expirationTime": 0}),
        _stub_response(200, {"orderId": "abc-123", "ordStatus": "Working"}),
    ]
    http.get.return_value = _stub_response(
        200, [{"id": 99, "name": "demo-account"}]
    )
    client = _client(http)
    result = client.place_limit_order(
        symbol="MES", qty=1, side="buy", limit_price=5000.0
    )
    assert result.success
    assert result.order_type == "limit"
    assert result.limit_price == 5000.0
    args, kwargs = http.post.call_args_list[1]
    assert _DEMO_URL in args[0]
    body = kwargs["json"]
    assert body["accountId"] == "99"
    assert body["contractName"] == "MES"
    assert body["orderType"] == "Limit"
    assert body["timeInForce"] == "DAY"
    assert kwargs["headers"]["Authorization"] == "Bearer tok"


def test_stop_order_payload_includes_stop_price() -> None:
    http = MagicMock()
    http.post.side_effect = [
        _stub_response(200, {"accessToken": "tok"}),
        _stub_response(200, {"orderId": "stop-1", "ordStatus": "Working"}),
    ]
    http.get.return_value = _stub_response(200, [{"id": 1}])
    client = _client(http)
    result = client.place_stop_order(
        symbol="MNQ", qty=2, stop_price=17400.0, side="sell"
    )
    assert result.success
    body = http.post.call_args_list[1].kwargs["json"]
    assert body["stopPrice"] == 17400.0
    assert body["orderType"] == "Stop"


def test_market_order_blocked_when_validation_fails() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(200, {"accessToken": "tok"})
    http.get.side_effect = [
        _stub_response(200, [{"id": 1}]),
        _stub_response(404, {"errorText": "no quote"}),
    ]
    client = _client(http)
    result = client.validate_order(symbol="MES", qty=1, side="buy")
    assert not result.valid
    assert "no_quote" in (result.reason or "")


def test_validation_blocks_disabled_symbol() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(200, {"accessToken": "tok"})
    client = _client(http, enabled_symbols=["MES"])
    result = client.validate_order(symbol="MNQ", qty=1, side="buy")
    assert not result.valid
    assert "not enabled" in (result.reason or "")


def test_redact_secrets_masks_credentials() -> None:
    raw = {
        "name": "demo",
        "password": "secret",
        "client_secret": "abc",
        "nested": {"access_token": "xyz", "ok": True},
    }
    masked = redact_secrets(raw)
    assert masked["password"] == "***"
    assert masked["client_secret"] == "***"
    assert masked["nested"]["access_token"] == "***"
    assert masked["nested"]["ok"] is True
    assert masked["name"] == "demo"


def test_broker_failure_surfaces_after_network_error() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(200, {"accessToken": "tok"})
    http.get.side_effect = httpx.ConnectError("boom")
    client = _client(http)
    with pytest.raises(BrokerError):
        client.get_positions()
