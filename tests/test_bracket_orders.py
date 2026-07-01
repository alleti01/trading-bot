"""Bracket order tests for Alpaca + mock brokers."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from integrations.alpaca_paper_client import AlpacaPaperClient
from integrations.broker_base import OrderResult
from integrations.mock_broker import MockBroker

_PAPER_URL = "https://paper-api.alpaca.markets"


def _stub_response(status_code: int, payload):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    return resp


def test_mock_bracket_order_fills_and_carries_legs() -> None:
    broker = MockBroker(enabled_symbols=["SPY"])
    result = broker.place_bracket_order(
        symbol="SPY",
        qty=1,
        side="buy",
        entry_price=550.0,
        stop_price=545.0,
        target_price=560.0,
    )
    assert isinstance(result, OrderResult)
    assert result.success
    assert result.order_type == "limit"
    assert result.raw["order_class"] == "bracket"
    assert result.raw["stop_loss"] == 545.0
    assert result.raw["take_profit"] == 560.0


def test_alpaca_bracket_order_payload_has_legs() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(
        200, {"id": "br-1", "status": "accepted"}
    )
    client = AlpacaPaperClient(
        api_key="k",
        secret_key="s",
        base_url=_PAPER_URL,
        paper=True,
        enabled_symbols=["SPY"],
        http_client=http,
    )
    result = client.place_bracket_order(
        symbol="SPY",
        qty=1,
        side="buy",
        entry_price=550.0,
        stop_price=545.0,
        target_price=560.0,
    )
    assert result.success
    assert result.stop_price == 545.0
    body = http.post.call_args.kwargs["json"]
    assert body["order_class"] == "bracket"
    assert body["type"] == "limit"
    assert body["limit_price"] == 550.0
    assert body["take_profit"]["limit_price"] == 560.0
    assert body["stop_loss"]["stop_price"] == 545.0


def test_alpaca_bracket_sends_integer_qty_and_penny_prices() -> None:
    # Whole-share qty must serialize as an int (67, not 67.0) so Alpaca
    # doesn't route it into the fractional path (which 422s with brackets),
    # and prices must be penny-rounded.
    http = MagicMock()
    http.post.return_value = _stub_response(200, {"id": "br-x", "status": "accepted"})
    client = AlpacaPaperClient(
        api_key="k", secret_key="s", base_url=_PAPER_URL, paper=True,
        enabled_symbols=["MSFT"], http_client=http,
    )
    client.place_bracket_order(
        symbol="MSFT", qty=67.0, side="sell",
        entry_price=372.51, stop_price=373.0688, target_price=371.3925,
    )
    body = http.post.call_args.kwargs["json"]
    assert body["qty"] == 67
    assert isinstance(body["qty"], int)
    assert body["limit_price"] == 372.51
    assert body["stop_loss"]["stop_price"] == 373.07
    assert body["take_profit"]["limit_price"] == 371.39


def test_alpaca_surfaces_error_message_on_rejection() -> None:
    # A 422 with a JSON message must propagate the reason, not a bare status.
    from integrations.broker_base import BrokerError

    http = MagicMock()
    http.post.return_value = _stub_response(
        422, {"code": 40010001, "message": "potential wash trade detected"}
    )
    client = AlpacaPaperClient(
        api_key="k", secret_key="s", base_url=_PAPER_URL, paper=True,
        enabled_symbols=["MSFT"], http_client=http,
    )
    with pytest.raises(BrokerError, match="potential wash trade detected"):
        client.place_bracket_order(
            symbol="MSFT", qty=1, side="sell",
            entry_price=372.51, stop_price=373.07, target_price=371.39,
        )


def test_alpaca_bracket_synthesizes_target_when_missing() -> None:
    http = MagicMock()
    http.post.return_value = _stub_response(200, {"id": "br-2", "status": "accepted"})
    client = AlpacaPaperClient(
        api_key="k",
        secret_key="s",
        base_url=_PAPER_URL,
        paper=True,
        enabled_symbols=["SPY"],
        http_client=http,
    )
    result = client.place_bracket_order(
        symbol="SPY",
        qty=1,
        side="buy",
        entry_price=100.0,
        stop_price=95.0,
    )
    assert result.success
    body = http.post.call_args.kwargs["json"]
    # risk = 5, target = entry + 2*risk = 110 for a long.
    assert body["take_profit"]["limit_price"] == 110.0


def test_alpaca_bracket_rejects_futures() -> None:
    from integrations.alpaca_paper_client import AlpacaFuturesNotSupported

    http = MagicMock()
    client = AlpacaPaperClient(
        api_key="k",
        secret_key="s",
        base_url=_PAPER_URL,
        paper=True,
        enabled_symbols=["MES"],
        http_client=http,
    )
    with pytest.raises(AlpacaFuturesNotSupported):
        client.place_bracket_order(
            symbol="MES", qty=1, side="buy", entry_price=5000.0, stop_price=4990.0
        )


def test_base_broker_default_bracket_is_limit_entry() -> None:
    # Tradovate / other adapters inherit the default: a plain limit entry.
    broker = MockBroker(enabled_symbols=["SPY"])
    # MockBroker overrides bracket, so test the default via a tiny stub.
    from integrations.broker_base import BaseBroker

    calls = {}

    class _StubBroker(BaseBroker):
        provider_name = "stub"

        def get_account(self):  # noqa: ANN201
            raise NotImplementedError

        def get_positions(self):  # noqa: ANN201
            return []

        def get_open_orders(self):  # noqa: ANN201
            return []

        def get_latest_quote(self, symbol):  # noqa: ANN001, ANN201
            raise NotImplementedError

        def validate_order(self, *, symbol, qty, side):  # noqa: ANN001, ANN201
            raise NotImplementedError

        def place_market_order(self, **kw):  # noqa: ANN003, ANN201
            raise NotImplementedError

        def place_limit_order(self, *, symbol, qty, side, limit_price, time_in_force="day"):  # noqa: ANN001, ANN201
            calls["limit"] = (symbol, qty, side, limit_price)
            return OrderResult(
                success=True, simulated=True, order_id="x", symbol=symbol,
                side=side, quantity=qty, order_type="limit", status="working",
                limit_price=limit_price,
            )

        def place_stop_order(self, **kw):  # noqa: ANN003, ANN201
            raise NotImplementedError

        def place_trailing_stop(self, **kw):  # noqa: ANN003, ANN201
            raise NotImplementedError

        def close_position(self, *, symbol):  # noqa: ANN001, ANN201
            raise NotImplementedError

        def cancel_order(self, *, order_id):  # noqa: ANN001, ANN201
            raise NotImplementedError

    stub = _StubBroker()
    result = stub.place_bracket_order(
        symbol="SPY", qty=1, side="buy", entry_price=550.0, stop_price=545.0
    )
    assert result.success
    assert calls["limit"] == ("SPY", 1, "buy", 550.0)
