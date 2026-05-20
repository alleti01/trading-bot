"""Tradovate DEMO/PAPER broker adapter (httpx-based, simulation only).

Hard safety rails (refuse to act unless ALL are true):

1. ``WORKFLOW_EXECUTION_MODE != "LIVE"`` (LIVE is locked at the router).
2. ``TRADOVATE_DEMO=true``.
3. ``TRADOVATE_BASE_URL`` contains ``demo.tradovateapi.com``.
4. Required credentials are set.

This module never imports anything from ``execution/`` or ``risk/`` — it
is a thin client used by the workflow layer.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

import httpx

from app.logging_config import get_logger
from integrations.broker_base import (
    AccountState,
    BaseBroker,
    BrokerError,
    OpenOrder,
    OrderResult,
    OrderSide,
    PositionState,
    Quote,
    SUPPORTED_SYMBOLS,
    TimeInForce,
    ValidationResult,
    redact_secrets,
)


_DEMO_HOST = "demo.tradovateapi.com"


class TradovateConfigurationError(BrokerError):
    """Raised when Tradovate config does not satisfy the demo-only rails."""


class TradovateDemoClient(BaseBroker):
    """HTTP client for Tradovate's DEMO REST API (simulation only)."""

    provider_name = "tradovate"

    def __init__(
        self,
        *,
        base_url: str,
        username: Optional[str],
        password: Optional[str],
        app_id: Optional[str],
        app_version: str,
        client_id: Optional[str],
        client_secret: Optional[str],
        demo: bool,
        enabled_symbols: Optional[list[str]] = None,
        timeout_seconds: float = 15.0,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.log = get_logger("integrations.tradovate")
        self.enabled_symbols = {
            s.upper() for s in (enabled_symbols or list(SUPPORTED_SYMBOLS))
        }
        if not demo:
            raise TradovateConfigurationError(
                "TRADOVATE_DEMO=false — refusing to trade. "
                "Set TRADOVATE_DEMO=true to use Tradovate's demo/simulation API."
            )
        if _DEMO_HOST not in (base_url or "").lower():
            raise TradovateConfigurationError(
                f"TRADOVATE_BASE_URL '{base_url}' does not contain "
                f"'{_DEMO_HOST}'. Refusing to trade against a non-demo URL."
            )
        if not username or not password or not app_id:
            raise TradovateConfigurationError(
                "Missing Tradovate demo credentials "
                "(TRADOVATE_USERNAME / TRADOVATE_PASSWORD / TRADOVATE_APP_ID)."
            )

        self.base_url = base_url.rstrip("/")
        self._username = username
        self._password = password
        self._app_id = app_id
        self._app_version = app_version
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token: Optional[str] = None
        self._access_token_expires_ms: int = 0
        self._account_id: Optional[str] = None
        self._http = http_client or httpx.Client(timeout=timeout_seconds)

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def _ensure_access_token(self) -> str:
        if (
            self._access_token
            and self._access_token_expires_ms > int(time.time() * 1000) + 30_000
        ):
            return self._access_token
        body = {
            "name": self._username,
            "password": self._password,
            "appId": self._app_id,
            "appVersion": self._app_version,
            "cid": self._client_id,
            "sec": self._client_secret,
        }
        url = f"{self.base_url}/auth/accesstokenrequest"
        try:
            resp = self._http.post(url, json=body)
        except httpx.HTTPError as e:
            raise BrokerError(f"Tradovate auth network error: {e}") from e

        if resp.status_code != 200:
            raise BrokerError(
                f"Tradovate auth failed (status={resp.status_code})"
            )
        data = resp.json()
        token = data.get("accessToken")
        expires_at = data.get("expirationTime")
        if not token:
            raise BrokerError(
                f"Tradovate auth missing accessToken (errorText={data.get('errorText')})"
            )
        self._access_token = token
        self._access_token_expires_ms = (
            int(time.time() * 1000) + 60 * 60 * 1000
            if not expires_at
            else int(time.time() * 1000) + 60 * 60 * 1000
        )
        self.log.info(
            "tradovate.auth_ok",
            user=self._username,
            base_url=self.base_url,
        )
        return token

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._ensure_access_token()}"}

    def _get(self, path: str, *, params: Optional[dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._http.get(url, headers=self._headers(), params=params)
        except httpx.HTTPError as e:
            raise BrokerError(f"Tradovate GET {path} network error: {e}") from e
        if resp.status_code >= 400:
            raise BrokerError(
                f"Tradovate GET {path} failed status={resp.status_code}"
            )
        return resp.json()

    def _post(self, path: str, *, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        safe_body = redact_secrets(body)
        try:
            resp = self._http.post(url, headers=self._headers(), json=body)
        except httpx.HTTPError as e:
            self.log.error("tradovate.post_failed", path=path, body=safe_body, error=str(e))
            raise BrokerError(f"Tradovate POST {path} network error: {e}") from e
        if resp.status_code >= 400:
            self.log.error(
                "tradovate.post_status",
                path=path,
                body=safe_body,
                status=resp.status_code,
            )
            raise BrokerError(
                f"Tradovate POST {path} failed status={resp.status_code}"
            )
        data = resp.json()
        self.log.info(
            "tradovate.post_ok",
            path=path,
            body=safe_body,
            response_keys=sorted(data.keys()) if isinstance(data, dict) else "list",
        )
        return data

    # ------------------------------------------------------------------
    # Accounts / state
    # ------------------------------------------------------------------
    def _resolve_account_id(self) -> str:
        if self._account_id:
            return self._account_id
        accounts = self._get("/account/list")
        if not isinstance(accounts, list) or not accounts:
            raise BrokerError("Tradovate returned no demo accounts")
        primary = accounts[0]
        self._account_id = str(primary.get("id"))
        return self._account_id

    def get_account(self) -> AccountState:
        account_id = self._resolve_account_id()
        cash = 0.0
        try:
            cash_resp = self._get("/cashBalance/getCashBalanceSnapshot", params={"accountId": account_id})
            if isinstance(cash_resp, dict):
                cash = float(cash_resp.get("amount", 0.0))
        except BrokerError:
            cash = 0.0
        return AccountState(
            account_id=account_id,
            cash_balance=cash,
            buying_power=cash,
            equity=cash,
        )

    def get_positions(self) -> list[PositionState]:
        account_id = self._resolve_account_id()
        rows = self._get("/position/list", params={"accountId": account_id})
        positions: list[PositionState] = []
        if isinstance(rows, list):
            for row in rows:
                qty = float(row.get("netPos", 0.0) or 0.0)
                if qty == 0.0:
                    continue
                direction = "long" if qty > 0 else "short"
                positions.append(
                    PositionState(
                        symbol=str(row.get("contractName", "")).upper(),
                        quantity=qty,
                        average_price=float(row.get("avgPrice", 0.0) or 0.0),
                        direction=direction,
                    )
                )
        return positions

    def get_open_orders(self) -> list[OpenOrder]:
        account_id = self._resolve_account_id()
        rows = self._get("/order/list", params={"accountId": account_id})
        orders: list[OpenOrder] = []
        if isinstance(rows, list):
            for row in rows:
                if str(row.get("ordStatus")).lower() not in {"working", "accepted"}:
                    continue
                orders.append(
                    OpenOrder(
                        order_id=str(row.get("id")),
                        symbol=str(row.get("contractName", "")).upper(),
                        side="buy" if str(row.get("action")).lower() == "buy" else "sell",
                        quantity=float(row.get("orderQty", 0.0) or 0.0),
                        order_type=_map_order_type(row.get("orderType")),
                        status=str(row.get("ordStatus", "working")).lower(),  # type: ignore[arg-type]
                        limit_price=row.get("price"),
                        stop_price=row.get("stopPrice"),
                    )
                )
        return orders

    def get_latest_quote(self, symbol: str) -> Quote:
        sym = symbol.upper()
        try:
            data = self._get(f"/marketData/quote", params={"contractName": sym})
        except BrokerError as e:
            raise BrokerError(f"No quote for {sym}: {e}") from e
        if not isinstance(data, dict):
            raise BrokerError(f"Bad quote payload for {sym}")
        bid = float(data.get("bid", 0.0) or 0.0)
        ask = float(data.get("ask", 0.0) or 0.0)
        last = float(data.get("last", 0.0) or 0.0) or ((bid + ask) / 2 if bid and ask else 0.0)
        if not last:
            raise BrokerError(f"No quote for {sym}")
        return Quote(symbol=sym, bid=bid, ask=ask, last=last)

    def validate_order(
        self, *, symbol: str, qty: float, side: OrderSide
    ) -> ValidationResult:
        sym = symbol.upper()
        if qty <= 0:
            return ValidationResult(valid=False, reason="qty must be > 0")
        if sym not in SUPPORTED_SYMBOLS:
            return ValidationResult(valid=False, reason=f"unsupported symbol {sym}")
        if sym not in self.enabled_symbols:
            return ValidationResult(valid=False, reason=f"symbol {sym} not enabled")
        try:
            self._resolve_account_id()
        except BrokerError as e:
            return ValidationResult(valid=False, reason=f"account_state_failed: {e}")
        try:
            quote = self.get_latest_quote(sym)
        except BrokerError as e:
            return ValidationResult(valid=False, reason=f"no_quote: {e}")
        return ValidationResult(valid=True, quote=quote)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def _build_order_body(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: TimeInForce = "day",
    ) -> dict[str, Any]:
        return {
            "accountId": self._resolve_account_id(),
            "contractName": symbol.upper(),
            "action": "Buy" if side == "buy" else "Sell",
            "orderQty": float(qty),
            "orderType": order_type,
            "price": limit_price,
            "stopPrice": stop_price,
            "timeInForce": time_in_force.upper(),
            "isAutomated": True,
        }

    def _submit(self, body: dict[str, Any], *, default_status: str = "working") -> dict[str, Any]:
        try:
            resp = self._post("/order/placeorder", body=body)
        except BrokerError:
            raise
        if not isinstance(resp, dict):
            raise BrokerError("Tradovate returned a non-object order response")
        return resp

    def place_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        body = self._build_order_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="Market",
            time_in_force=time_in_force,
        )
        resp = self._submit(body)
        return _result_from_response(resp, body, order_type="market")

    def place_limit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        body = self._build_order_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="Limit",
            limit_price=limit_price,
            time_in_force=time_in_force,
        )
        resp = self._submit(body)
        return _result_from_response(resp, body, order_type="limit")

    def place_stop_order(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        body = self._build_order_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="Stop",
            stop_price=stop_price,
            time_in_force=time_in_force,
        )
        resp = self._submit(body)
        return _result_from_response(resp, body, order_type="stop")

    def place_trailing_stop(
        self,
        *,
        symbol: str,
        qty: float,
        trail: float,
        side: OrderSide,
        trail_kind: Literal["ticks", "percent"] = "ticks",
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        body = self._build_order_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="TrailingStop",
            time_in_force=time_in_force,
        )
        body["trail"] = trail
        body["trailKind"] = trail_kind
        resp = self._submit(body)
        result = _result_from_response(resp, body, order_type="trailing_stop")
        result.raw.update({"trail": trail, "trail_kind": trail_kind})
        return result

    def close_position(self, *, symbol: str) -> OrderResult:
        positions = self.get_positions()
        target = next((p for p in positions if p.symbol == symbol.upper()), None)
        if target is None:
            return OrderResult(
                success=False,
                simulated=True,
                order_id=uuid.uuid4().hex,
                symbol=symbol.upper(),
                side="sell",
                quantity=0.0,
                order_type="market",
                status="rejected",
                reason="no_position",
            )
        side: OrderSide = "sell" if target.direction == "long" else "buy"
        return self.place_market_order(
            symbol=symbol,
            qty=abs(target.quantity),
            side=side,
        )

    def cancel_order(self, *, order_id: str) -> OrderResult:
        body = {"orderId": order_id}
        try:
            resp = self._post("/order/cancelorder", body=body)
        except BrokerError as e:
            return OrderResult(
                success=False,
                simulated=True,
                order_id=order_id,
                symbol="",
                side="buy",
                quantity=0.0,
                order_type="market",
                status="rejected",
                reason=str(e),
            )
        return OrderResult(
            success=bool(resp.get("ok", True)),
            simulated=True,
            order_id=str(resp.get("id", order_id)),
            symbol=str(resp.get("contractName", "")),
            side="buy",
            quantity=0.0,
            order_type="market",
            status="cancelled",
            raw=resp,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _map_order_type(raw: Any) -> str:
    s = str(raw or "").lower()
    if "stop" in s and "trail" in s:
        return "trailing_stop"
    if "stop" in s:
        return "stop"
    if "limit" in s:
        return "limit"
    return "market"


def _result_from_response(
    resp: dict[str, Any],
    body: dict[str, Any],
    *,
    order_type: str,
) -> OrderResult:
    success = bool(resp.get("orderId") or resp.get("id") or resp.get("ok", True))
    status = "working"
    if str(resp.get("ordStatus", "")).lower() in {"filled", "rejected", "cancelled"}:
        status = str(resp["ordStatus"]).lower()
    side: OrderSide = "buy" if str(body.get("action")).lower() == "buy" else "sell"
    return OrderResult(
        success=success,
        simulated=True,
        order_id=str(resp.get("orderId") or resp.get("id") or uuid.uuid4().hex),
        symbol=str(body.get("contractName", "")).upper(),
        side=side,
        quantity=float(body.get("orderQty", 0.0)),
        order_type=order_type,  # type: ignore[arg-type]
        status=status,  # type: ignore[arg-type]
        time_in_force=str(body.get("timeInForce", "DAY")).lower(),  # type: ignore[arg-type]
        limit_price=body.get("price"),
        stop_price=body.get("stopPrice"),
        reason=resp.get("errorText"),
        raw=resp,
    )
