"""Options execution adapter (Alpaca paper + mock). Single-leg + multi-leg.

DRY_RUN / mock returns simulated JSON without any network call. PAPER
uses Alpaca's paper options endpoints only. LIVE is refused upstream by
the broker router — this module never has a live path.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from app.logging_config import get_logger
from integrations.broker_base import BrokerError, redact_secrets
from options.contract import OptionAction, OptionContract

OrderClass = Literal["simple", "spread", "iron_condor", "straddle"]


@dataclass
class OptionLeg:
    contract: OptionContract
    action: OptionAction
    ratio: int = 1


@dataclass
class OptionOrderResult:
    success: bool
    simulated: bool
    order_id: str
    underlying: str
    order_class: OrderClass
    legs: list[dict[str, Any]] = field(default_factory=list)
    status: str = "accepted"
    limit_price: Optional[float] = None
    reason: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "simulated": self.simulated,
            "order_id": self.order_id,
            "underlying": self.underlying,
            "order_class": self.order_class,
            "legs": self.legs,
            "status": self.status,
            "limit_price": self.limit_price,
            "reason": self.reason,
        }


class BaseOptionsExecutor:
    provider_name = "abstract"

    def place_single_leg(
        self, *, contract: OptionContract, action: OptionAction, qty: int,
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        raise NotImplementedError

    def place_multi_leg(
        self, *, underlying: str, legs: list[OptionLeg], qty: int,
        order_class: OrderClass, limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        raise NotImplementedError

    def close_contract(self, *, contract: OptionContract, qty: int) -> OptionOrderResult:
        raise NotImplementedError


class MockOptionsExecutor(BaseOptionsExecutor):
    """In-memory options executor for DRY_RUN + tests (no network)."""

    provider_name = "mock"

    def __init__(self) -> None:
        self.log = get_logger("options.execution.mock")
        self.orders: dict[str, OptionOrderResult] = {}

    def place_single_leg(
        self, *, contract: OptionContract, action: OptionAction, qty: int,
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        oid = uuid.uuid4().hex
        result = OptionOrderResult(
            success=True,
            simulated=True,
            order_id=oid,
            underlying=contract.underlying,
            order_class="simple",
            legs=[{
                "occ_symbol": contract.occ_symbol,
                "action": action,
                "ratio": 1,
                "strike": contract.strike,
                "option_type": contract.option_type,
            }],
            status="filled",
            limit_price=limit_price or contract.mid_price,
        )
        self.orders[oid] = result
        return result

    def place_multi_leg(
        self, *, underlying: str, legs: list[OptionLeg], qty: int,
        order_class: OrderClass, limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        oid = uuid.uuid4().hex
        result = OptionOrderResult(
            success=True,
            simulated=True,
            order_id=oid,
            underlying=underlying,
            order_class=order_class,
            legs=[
                {
                    "occ_symbol": leg.contract.occ_symbol,
                    "action": leg.action,
                    "ratio": leg.ratio,
                    "strike": leg.contract.strike,
                    "option_type": leg.contract.option_type,
                }
                for leg in legs
            ],
            status="filled",
            limit_price=limit_price,
        )
        self.orders[oid] = result
        return result

    def close_contract(self, *, contract: OptionContract, qty: int) -> OptionOrderResult:
        action: OptionAction = "sell_to_close"
        return self.place_single_leg(contract=contract, action=action, qty=qty)


class AlpacaOptionsExecutor(BaseOptionsExecutor):
    """Places options orders on Alpaca's PAPER endpoints only."""

    provider_name = "alpaca"
    _PAPER_HOST = "paper-api.alpaca.markets"

    def __init__(
        self,
        *,
        api_key: Optional[str],
        secret_key: Optional[str],
        base_url: str,
        paper: bool,
        timeout_seconds: float = 15.0,
        http_client: Any = None,
    ) -> None:
        self.log = get_logger("options.execution.alpaca")
        if not paper:
            raise BrokerError(
                "ALPACA_PAPER=false — refusing options trading. Paper only."
            )
        if self._PAPER_HOST not in (base_url or "").lower():
            raise BrokerError(
                f"ALPACA_BASE_URL must contain {self._PAPER_HOST} for options paper trading."
            )
        if not api_key or not secret_key:
            raise BrokerError("Missing Alpaca credentials for options trading.")
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._secret_key = secret_key
        self._timeout = timeout_seconds
        self._http = http_client

    def _client(self):  # noqa: ANN202
        if self._http is not None:
            return self._http
        import httpx

        self._http = httpx.Client(timeout=self._timeout)
        return self._http

    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }

    def _post_order(self, body: dict[str, Any]) -> dict[str, Any]:
        import httpx

        url = f"{self.base_url}/v2/orders"
        safe = redact_secrets(body)
        try:
            resp = self._client().post(url, headers=self._headers(), json=body)
        except httpx.HTTPError as e:
            self.log.error("options.order_failed", body=safe, error=str(e))
            raise BrokerError(f"Alpaca options order network error: {e}") from e
        if resp.status_code >= 400:
            self.log.error("options.order_status", body=safe, status=resp.status_code)
            raise BrokerError(
                f"Alpaca options order failed status={resp.status_code}"
            )
        data = resp.json()
        self.log.info("options.order_ok", body=safe, order_id=data.get("id"))
        return data

    @staticmethod
    def _alpaca_side(action: OptionAction) -> str:
        return "buy" if action in ("buy_to_open", "buy_to_close") else "sell"

    @staticmethod
    def _alpaca_position_intent(action: OptionAction) -> str:
        return {
            "buy_to_open": "buy_to_open",
            "sell_to_close": "sell_to_close",
            "sell_to_open": "sell_to_open",
            "buy_to_close": "buy_to_close",
        }[action]

    def place_single_leg(
        self, *, contract: OptionContract, action: OptionAction, qty: int,
        limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        body: dict[str, Any] = {
            "symbol": contract.occ_symbol,
            "qty": str(qty),
            "side": self._alpaca_side(action),
            "type": "limit" if limit_price else "market",
            "time_in_force": "day",
            "position_intent": self._alpaca_position_intent(action),
        }
        if limit_price:
            body["limit_price"] = str(limit_price)
        data = self._post_order(body)
        return OptionOrderResult(
            success="id" in data,
            simulated=True,
            order_id=str(data.get("id", uuid.uuid4().hex)),
            underlying=contract.underlying,
            order_class="simple",
            legs=[{"occ_symbol": contract.occ_symbol, "action": action, "ratio": 1}],
            status=str(data.get("status", "accepted")),
            limit_price=limit_price,
            raw=data,
        )

    def place_multi_leg(
        self, *, underlying: str, legs: list[OptionLeg], qty: int,
        order_class: OrderClass, limit_price: Optional[float] = None,
    ) -> OptionOrderResult:
        body: dict[str, Any] = {
            "order_class": "mleg",
            "qty": str(qty),
            "type": "limit" if limit_price else "market",
            "time_in_force": "day",
            "legs": [
                {
                    "symbol": leg.contract.occ_symbol,
                    "ratio_qty": str(leg.ratio),
                    "side": self._alpaca_side(leg.action),
                    "position_intent": self._alpaca_position_intent(leg.action),
                }
                for leg in legs
            ],
        }
        if limit_price:
            body["limit_price"] = str(limit_price)
        data = self._post_order(body)
        return OptionOrderResult(
            success="id" in data,
            simulated=True,
            order_id=str(data.get("id", uuid.uuid4().hex)),
            underlying=underlying,
            order_class=order_class,
            legs=[
                {"occ_symbol": leg.contract.occ_symbol, "action": leg.action, "ratio": leg.ratio}
                for leg in legs
            ],
            status=str(data.get("status", "accepted")),
            limit_price=limit_price,
            raw=data,
        )

    def close_contract(self, *, contract: OptionContract, qty: int) -> OptionOrderResult:
        return self.place_single_leg(
            contract=contract, action="sell_to_close", qty=qty
        )
