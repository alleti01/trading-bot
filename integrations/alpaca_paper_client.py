"""Alpaca PAPER trading adapter (httpx, sandbox only).

Hard safety rails (refuse to construct unless ALL pass):

1. ``ALPACA_PAPER=true``
2. ``ALPACA_BASE_URL`` contains ``paper-api.alpaca.markets``
3. ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` are set

Futures symbols (``MES, MNQ, ES, NQ, MGC, MCL, MYM, M2K``) are rejected
at every order method with a clear error message — Alpaca does not
trade futures. Use the mock broker or the Tradovate adapter for those.

This module never imports ``execution/`` or ``risk/``. It is a thin
client used by the broker router.
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
    ExitFill,
    FUTURES_SYMBOLS,
    OpenOrder,
    OrderResult,
    OrderSide,
    PositionState,
    Quote,
    TimeInForce,
    ValidationResult,
    redact_secrets,
)


_PAPER_HOST = "paper-api.alpaca.markets"
_DATA_URL = "https://data.alpaca.markets/v2"


class AlpacaConfigurationError(BrokerError):
    """Raised when Alpaca config does not satisfy the paper-only rails."""


class AlpacaFuturesNotSupported(BrokerError):
    """Raised whenever a caller asks Alpaca for a futures symbol."""


class AlpacaPaperClient(BaseBroker):
    """REST client for Alpaca's PAPER endpoints (sandbox only)."""

    provider_name = "alpaca"

    def __init__(
        self,
        *,
        api_key: Optional[str],
        secret_key: Optional[str],
        base_url: str,
        paper: bool,
        enabled_symbols: Optional[list[str]] = None,
        timeout_seconds: float = 15.0,
        http_client: Optional[httpx.Client] = None,
        data_url: str = _DATA_URL,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.log = get_logger("integrations.alpaca")
        if not paper:
            raise AlpacaConfigurationError(
                "ALPACA_PAPER=false — refusing to trade. "
                "Set ALPACA_PAPER=true to use Alpaca's paper sandbox."
            )
        if _PAPER_HOST not in (base_url or "").lower():
            raise AlpacaConfigurationError(
                f"ALPACA_BASE_URL '{base_url}' does not contain "
                f"'{_PAPER_HOST}'. Refusing to trade against a non-paper URL."
            )
        if not api_key or not secret_key:
            raise AlpacaConfigurationError(
                "Missing Alpaca paper credentials "
                "(ALPACA_API_KEY / ALPACA_SECRET_KEY)."
            )

        self.base_url = base_url.rstrip("/")
        self.data_url = data_url.rstrip("/")
        self._api_key = api_key
        self._secret_key = secret_key
        self.enabled_symbols = {
            s.upper() for s in (enabled_symbols or [])
        }
        self._http = http_client or httpx.Client(timeout=timeout_seconds)
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff = float(retry_backoff_seconds)
        self._headers_cache = {
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._secret_key,
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return dict(self._headers_cache)

    def _with_retry(self, label: str, fn):  # noqa: ANN001, ANN202
        """Retry transient network errors (TLS/connect/read timeouts, DNS).

        A single flaky handshake shouldn't fail the pre-trade reconcile and
        block the whole scan cycle. HTTP status errors (4xx/5xx) are NOT
        retried here — only ``httpx.HTTPError`` transport failures.
        """
        last: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except httpx.HTTPError as e:
                last = e
                if attempt < self._max_retries:
                    self.log.warning(
                        "alpaca.net_retry",
                        op=label,
                        attempt=attempt + 1,
                        max=self._max_retries,
                        error=str(e),
                    )
                    time.sleep(self._retry_backoff * (attempt + 1))
                    continue
                raise
        assert last is not None  # pragma: no cover
        raise last

    def _get(self, path: str, *, params: Optional[dict[str, Any]] = None, base: Optional[str] = None) -> Any:
        url = f"{base or self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._with_retry(
                f"GET {path}",
                lambda: self._http.get(url, headers=self._headers(), params=params),
            )
        except httpx.HTTPError as e:
            raise BrokerError(f"Alpaca GET {path} network error: {e}") from e
        if resp.status_code >= 400:
            raise BrokerError(
                f"Alpaca GET {path} failed status={resp.status_code}"
            )
        return resp.json()

    def _post(self, path: str, *, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        safe_body = redact_secrets(body)
        try:
            resp = self._with_retry(
                f"POST {path}",
                lambda: self._http.post(url, headers=self._headers(), json=body),
            )
        except httpx.HTTPError as e:
            self.log.error("alpaca.post_failed", path=path, body=safe_body, error=str(e))
            raise BrokerError(f"Alpaca POST {path} network error: {e}") from e
        if resp.status_code >= 400:
            # Alpaca returns a JSON body like {"code": ..., "message": "..."}
            # explaining the rejection. Surface it so the log + Discord alert
            # say *why* (e.g. wash trade, fractional+bracket, not shortable).
            detail = ""
            try:
                err = resp.json()
                detail = err.get("message") if isinstance(err, dict) else str(err)
            except Exception:  # noqa: BLE001
                detail = (resp.text or "").strip()[:300]
            self.log.error(
                "alpaca.post_status",
                path=path,
                body=safe_body,
                status=resp.status_code,
                detail=detail,
            )
            suffix = f": {detail}" if detail else ""
            raise BrokerError(
                f"Alpaca POST {path} failed status={resp.status_code}{suffix}"
            )
        data = resp.json()
        self.log.info(
            "alpaca.post_ok",
            path=path,
            body=safe_body,
            response_keys=sorted(data.keys()) if isinstance(data, dict) else "list",
        )
        return data

    def _delete(self, path: str) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        try:
            resp = self._http.delete(url, headers=self._headers())
        except httpx.HTTPError as e:
            raise BrokerError(f"Alpaca DELETE {path} network error: {e}") from e
        if resp.status_code >= 400 and resp.status_code != 404:
            raise BrokerError(
                f"Alpaca DELETE {path} failed status={resp.status_code}"
            )
        try:
            return resp.json()
        except ValueError:
            return {}

    # ------------------------------------------------------------------
    # Symbol guards
    # ------------------------------------------------------------------
    @staticmethod
    def _refuse_if_futures(symbol: str) -> None:
        sym = symbol.upper()
        if sym in FUTURES_SYMBOLS:
            raise AlpacaFuturesNotSupported(
                "Alpaca adapter does not support futures. "
                "Use local simulator or futures broker adapter."
            )

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def get_account(self) -> AccountState:
        data = self._get("/v2/account")
        return AccountState(
            account_id=str(data.get("id", "")),
            cash_balance=float(data.get("cash", 0.0) or 0.0),
            buying_power=float(data.get("buying_power", 0.0) or 0.0),
            equity=float(data.get("equity", 0.0) or 0.0),
            realized_pnl=float(data.get("equity", 0.0) or 0.0)
            - float(data.get("last_equity", 0.0) or 0.0),
            unrealized_pnl=0.0,
        )

    def get_positions(self) -> list[PositionState]:
        rows = self._get("/v2/positions")
        result: list[PositionState] = []
        if isinstance(rows, list):
            for row in rows:
                qty = float(row.get("qty", 0.0) or 0.0)
                side = str(row.get("side", "long")).lower()
                direction = "long" if side == "long" else "short"
                result.append(
                    PositionState(
                        symbol=str(row.get("symbol", "")).upper(),
                        quantity=qty,
                        average_price=float(row.get("avg_entry_price", 0.0) or 0.0),
                        direction=direction,  # type: ignore[arg-type]
                        unrealized_pnl=float(row.get("unrealized_pl", 0.0) or 0.0),
                    )
                )
        return result

    def get_open_orders(self) -> list[OpenOrder]:
        rows = self._get("/v2/orders", params={"status": "open"})
        result: list[OpenOrder] = []
        if isinstance(rows, list):
            for row in rows:
                result.append(
                    OpenOrder(
                        order_id=str(row.get("id", "")),
                        symbol=str(row.get("symbol", "")).upper(),
                        side=str(row.get("side", "buy")).lower(),  # type: ignore[arg-type]
                        quantity=float(row.get("qty", 0.0) or 0.0),
                        order_type=_map_order_type(row.get("type")),  # type: ignore[arg-type]
                        status=_map_status(row.get("status", "new")),  # type: ignore[arg-type]
                        limit_price=_maybe_float(row.get("limit_price")),
                        stop_price=_maybe_float(row.get("stop_price")),
                        time_in_force=str(row.get("time_in_force", "day")).lower(),  # type: ignore[arg-type]
                    )
                )
        return result

    def get_last_exit_fill(
        self, *, symbol: str, exit_side: OrderSide
    ) -> Optional[ExitFill]:
        """Find the most recent filled *closing* order for ``symbol``.

        When a bracket's take-profit or stop-loss fills, that leg becomes a
        closed order on the ``exit_side`` (opposite the entry). We pull the
        recent closed orders (parents + nested legs) and return the newest
        filled one on that side so the loop can compute realised P&L.
        """
        sym = symbol.upper()
        try:
            rows = self._get(
                "/v2/orders",
                params={
                    "status": "closed",
                    "symbols": sym,
                    "limit": 50,
                    "direction": "desc",
                    "nested": "true",
                },
            )
        except BrokerError as e:
            self.log.warning("alpaca.exit_fill_lookup_failed", symbol=sym, error=str(e))
            return None
        if not isinstance(rows, list):
            return None

        for parent in rows:
            if not isinstance(parent, dict):
                continue
            # A bracket parent nests its take-profit / stop-loss legs.
            candidates = [parent, *(parent.get("legs") or [])]
            for order in candidates:
                if not isinstance(order, dict):
                    continue
                if str(order.get("symbol", "")).upper() != sym:
                    continue
                if str(order.get("side", "")).lower() != exit_side:
                    continue
                if str(order.get("status", "")).lower() != "filled":
                    continue
                fill_price = _maybe_float(order.get("filled_avg_price"))
                if fill_price is None:
                    continue
                fill_qty = _maybe_float(order.get("filled_qty")) or 0.0
                return ExitFill(
                    symbol=sym,
                    price=fill_price,
                    quantity=abs(fill_qty),
                    side=exit_side,
                    exit_kind=_exit_kind_from_order(order),
                    filled_at=str(order.get("filled_at", "") or ""),
                )
        return None

    def get_latest_quote(self, symbol: str) -> Quote:
        self._refuse_if_futures(symbol)
        sym = symbol.upper()
        data = self._get(
            f"/stocks/{sym}/quotes/latest",
            base=self.data_url,
        )
        if not isinstance(data, dict):
            raise BrokerError(f"Bad quote payload for {sym}")
        quote = data.get("quote") or {}
        bid = float(quote.get("bp", 0.0) or 0.0)
        ask = float(quote.get("ap", 0.0) or 0.0)
        last = ask or bid
        if not last:
            raise BrokerError(f"No quote for {sym}")
        return Quote(
            symbol=sym,
            bid=bid,
            ask=ask,
            last=last,
            timestamp_ms=int(time.time() * 1000),
        )

    def validate_order(
        self, *, symbol: str, qty: float, side: OrderSide
    ) -> ValidationResult:
        sym = symbol.upper()
        if qty <= 0:
            return ValidationResult(valid=False, reason="qty must be > 0")
        if sym in FUTURES_SYMBOLS:
            return ValidationResult(
                valid=False,
                reason=(
                    "Alpaca adapter does not support futures. "
                    "Use local simulator or futures broker adapter."
                ),
            )
        if self.enabled_symbols and sym not in self.enabled_symbols:
            return ValidationResult(valid=False, reason=f"symbol {sym} not enabled")
        try:
            self.get_account()
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
    def _build_body(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: str,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        time_in_force: TimeInForce = "day",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": float(qty),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        if limit_price is not None:
            body["limit_price"] = float(limit_price)
        if stop_price is not None:
            body["stop_price"] = float(stop_price)
        if trail_percent is not None:
            body["trail_percent"] = float(trail_percent)
        return body

    def place_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        self._refuse_if_futures(symbol)
        body = self._build_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="market",
            time_in_force=time_in_force,
        )
        resp = self._post("/v2/orders", body=body)
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
        self._refuse_if_futures(symbol)
        body = self._build_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="limit",
            limit_price=limit_price,
            time_in_force=time_in_force,
        )
        resp = self._post("/v2/orders", body=body)
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
        self._refuse_if_futures(symbol)
        body = self._build_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="stop",
            stop_price=stop_price,
            time_in_force=time_in_force,
        )
        resp = self._post("/v2/orders", body=body)
        return _result_from_response(resp, body, order_type="stop")

    def place_trailing_stop(
        self,
        *,
        symbol: str,
        qty: float,
        trail: float,
        side: OrderSide,
        trail_kind: Literal["ticks", "percent"] = "percent",
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        self._refuse_if_futures(symbol)
        if trail_kind != "percent":
            raise BrokerError(
                "Alpaca trailing stops are percent-only. "
                "Pass trail_kind='percent'."
            )
        body = self._build_body(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="trailing_stop",
            trail_percent=trail,
            time_in_force=time_in_force,
        )
        resp = self._post("/v2/orders", body=body)
        result = _result_from_response(resp, body, order_type="trailing_stop")
        result.raw.update({"trail": trail, "trail_kind": "percent"})
        return result

    def place_bracket_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        entry_price: float,
        stop_price: float,
        target_price: Optional[float] = None,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        """Native Alpaca bracket order: entry + stop_loss (+ optional take_profit).

        Alpaca rejects a free-standing protective stop placed immediately
        after a still-unfilled entry (the 403 we saw). A ``bracket`` order
        class attaches the protective legs so the broker arms them once
        the entry fills.
        """
        self._refuse_if_futures(symbol)
        # A bracket order requires a take-profit leg; synthesize one from a
        # symmetric risk:reward if the caller didn't supply a target.
        if target_price is None:
            risk = abs(entry_price - stop_price)
            target_price = (
                entry_price + 2 * risk if side == "buy" else entry_price - 2 * risk
            )
        # Bracket orders (and short sales) do NOT support fractional shares.
        # Sending a float like 67.0 can route into Alpaca's fractional path
        # and 422. Send a whole-share integer; equity sizing already floors.
        qty_f = float(qty)
        qty_val: Any = int(qty_f) if qty_f.is_integer() else qty_f
        # Equity prices must be penny-aligned (<= 2 decimals) or Alpaca 422s.
        body: dict[str, Any] = {
            "symbol": symbol.upper(),
            "qty": qty_val,
            "side": side,
            "type": "limit",
            "limit_price": round(float(entry_price), 2),
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "take_profit": {"limit_price": round(float(target_price), 2)},
            "stop_loss": {"stop_price": round(float(stop_price), 2)},
        }
        resp = self._post("/v2/orders", body=body)
        raw = dict(resp) if isinstance(resp, dict) else {}
        raw.setdefault("order_class", "bracket")
        raw["take_profit"] = body["take_profit"]
        raw["stop_loss"] = body["stop_loss"]
        return OrderResult(
            success="id" in resp if isinstance(resp, dict) else False,
            simulated=True,
            order_id=str(resp.get("id", uuid.uuid4().hex)) if isinstance(resp, dict) else uuid.uuid4().hex,
            symbol=symbol.upper(),
            side=side,
            quantity=float(qty),
            order_type="limit",
            status=_map_status(resp.get("status", "accepted")) if isinstance(resp, dict) else "accepted",
            time_in_force=time_in_force,
            limit_price=float(entry_price),
            stop_price=float(stop_price),
            reason=resp.get("reject_reason") if isinstance(resp, dict) else None,
            raw=raw,
        )

    def close_position(self, *, symbol: str) -> OrderResult:
        self._refuse_if_futures(symbol)
        sym = symbol.upper()
        try:
            data = self._delete(f"/v2/positions/{sym}")
        except BrokerError as e:
            return OrderResult(
                success=False,
                simulated=True,
                order_id=uuid.uuid4().hex,
                symbol=sym,
                side="sell",
                quantity=0.0,
                order_type="market",
                status="rejected",
                reason=str(e),
            )
        order_id = str(data.get("id", uuid.uuid4().hex)) if isinstance(data, dict) else uuid.uuid4().hex
        return OrderResult(
            success=True,
            simulated=True,
            order_id=order_id,
            symbol=sym,
            side="sell",
            quantity=float(data.get("qty", 0.0) or 0.0) if isinstance(data, dict) else 0.0,
            order_type="market",
            status="filled",
            reason="close_position",
            raw=data if isinstance(data, dict) else {},
        )

    def cancel_order(self, *, order_id: str) -> OrderResult:
        try:
            self._delete(f"/v2/orders/{order_id}")
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
            success=True,
            simulated=True,
            order_id=order_id,
            symbol="",
            side="buy",
            quantity=0.0,
            order_type="market",
            status="cancelled",
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _maybe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_order_type(raw: Any) -> str:
    s = str(raw or "").lower()
    if "trailing" in s:
        return "trailing_stop"
    if "stop" in s:
        return "stop"
    if "limit" in s:
        return "limit"
    return "market"


def _exit_kind_from_order(order: dict[str, Any]) -> str:
    """Infer why a position closed from the closing order's type.

    A take-profit leg is a ``limit`` order; a stop-loss leg is a ``stop``
    (or ``stop_limit``); anything else (e.g. a market close on flatten)
    is a plain ``closed``.
    """
    t = str(order.get("type", "") or order.get("order_type", "")).lower()
    if "stop" in t:
        return "stop_loss"
    if "limit" in t:
        return "take_profit"
    return "closed"


def _map_status(raw: Any) -> str:
    s = str(raw or "").lower()
    if s in {"filled", "partially_filled"}:
        return "filled"
    if s in {"new", "pending_new", "accepted", "accepted_for_bidding"}:
        return "accepted"
    if s in {"canceled", "expired", "replaced"}:
        return "cancelled"
    if s in {"rejected", "suspended"}:
        return "rejected"
    return "working"


def _result_from_response(
    resp: dict[str, Any],
    body: dict[str, Any],
    *,
    order_type: str,
) -> OrderResult:
    if not isinstance(resp, dict):
        raise BrokerError("Alpaca returned a non-object order response")
    success = "id" in resp
    return OrderResult(
        success=success,
        simulated=True,
        order_id=str(resp.get("id", uuid.uuid4().hex)),
        symbol=str(body.get("symbol", "")).upper(),
        side=str(body.get("side", "buy")).lower(),  # type: ignore[arg-type]
        quantity=float(body.get("qty", 0.0) or 0.0),
        order_type=order_type,  # type: ignore[arg-type]
        status=_map_status(resp.get("status", "new")),  # type: ignore[arg-type]
        time_in_force=str(body.get("time_in_force", "day")).lower(),  # type: ignore[arg-type]
        limit_price=_maybe_float(body.get("limit_price")),
        stop_price=_maybe_float(body.get("stop_price")),
        fill_price=_maybe_float(resp.get("filled_avg_price")),
        reason=resp.get("reject_reason"),
        raw=resp,
    )
