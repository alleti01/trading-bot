"""Abstract broker adapter interface (paper/demo only)."""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

OrderSide = Literal["buy", "sell"]
TimeInForce = Literal["day", "gtc", "ioc", "fok"]
OrderType = Literal["market", "limit", "stop", "trailing_stop"]
OrderStatus = Literal["accepted", "rejected", "filled", "working", "cancelled"]


class BrokerError(RuntimeError):
    """Raised when a broker call cannot proceed safely."""


class LiveExecutionRefused(BrokerError):
    """Raised whenever a caller asks the adapter to act in LIVE mode."""


@dataclass(frozen=True)
class AccountState:
    account_id: str
    cash_balance: float
    buying_power: float
    equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class PositionState:
    symbol: str
    quantity: float
    average_price: float
    direction: Literal["long", "short", "flat"]
    unrealized_pnl: float = 0.0


@dataclass(frozen=True)
class OpenOrder:
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = "day"


@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    last: float
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass(frozen=True)
class ExitFill:
    """The fill that closed a position.

    ``exit_kind`` records *why* the position closed based on the closing
    order type: a take-profit limit, a protective stop, or a plain close.
    """

    symbol: str
    price: float
    quantity: float
    side: OrderSide
    exit_kind: Literal["take_profit", "stop_loss", "closed"] = "closed"
    filled_at: str = ""


@dataclass(frozen=True)
class OrderResult:
    """Structured response returned by every order method.

    All fields are JSON-safe so the workflow runner can log them and
    the audit trail can persist them without further coercion.
    """

    success: bool
    simulated: bool
    order_id: str
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    status: OrderStatus
    time_in_force: TimeInForce = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    fill_price: Optional[float] = None
    reason: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_payload(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "simulated": self.simulated,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "order_type": self.order_type,
            "status": self.status,
            "time_in_force": self.time_in_force,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "fill_price": self.fill_price,
            "reason": self.reason,
            "request_id": self.request_id,
        }


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reason: Optional[str] = None
    quote: Optional[Quote] = None


class BaseBroker(ABC):
    """Abstract broker adapter — paper/demo only."""

    provider_name: str = "abstract"

    @abstractmethod
    def get_account(self) -> AccountState: ...

    @abstractmethod
    def get_positions(self) -> list[PositionState]: ...

    @abstractmethod
    def get_open_orders(self) -> list[OpenOrder]: ...

    @abstractmethod
    def get_latest_quote(self, symbol: str) -> Quote: ...

    @abstractmethod
    def validate_order(
        self, *, symbol: str, qty: float, side: OrderSide
    ) -> ValidationResult: ...

    @abstractmethod
    def place_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult: ...

    @abstractmethod
    def place_limit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult: ...

    @abstractmethod
    def place_stop_order(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult: ...

    @abstractmethod
    def place_trailing_stop(
        self,
        *,
        symbol: str,
        qty: float,
        trail: float,
        side: OrderSide,
        trail_kind: Literal["ticks", "percent"] = "ticks",
        time_in_force: TimeInForce = "day",
    ) -> OrderResult: ...

    @abstractmethod
    def close_position(self, *, symbol: str) -> OrderResult: ...

    @abstractmethod
    def cancel_order(self, *, order_id: str) -> OrderResult: ...

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
        """Place an entry plus attached protective stop/target as one unit.

        Default implementation (used by adapters that do not support a
        native bracket order class) falls back to a plain limit entry —
        the caller is responsible for the protective leg. Adapters with a
        native bracket order (e.g. Alpaca) override this to attach the
        stop/target so the broker manages them after the entry fills.
        """
        return self.place_limit_order(
            symbol=symbol,
            qty=qty,
            side=side,
            limit_price=entry_price,
            time_in_force=time_in_force,
        )

    def get_last_exit_fill(
        self, *, symbol: str, exit_side: OrderSide
    ) -> Optional[ExitFill]:
        """Most recent *closing* fill for ``symbol`` on ``exit_side``.

        Used to report realised P&L after a bracket's take-profit/stop-loss
        fills server-side (so the loop never saw the close itself). Adapters
        that can't look up historical fills return ``None`` and the caller
        degrades to a close alert without a dollar figure.
        """
        return None

    def reconcile(self) -> dict[str, Any]:
        """Pull positions/orders before placing new orders."""
        try:
            positions = self.get_positions()
            orders = self.get_open_orders()
            account = self.get_account()
        except Exception as e:  # noqa: BLE001
            raise BrokerError(f"Broker reconcile failed: {e}") from e
        return {
            "account_id": account.account_id,
            "open_positions": len(positions),
            "open_orders": len(orders),
            "positions": [_position_payload(p) for p in positions],
            "orders": [_order_payload(o) for o in orders],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SUPPORTED_SYMBOLS: frozenset[str] = frozenset(
    {"MES", "MNQ", "ES", "NQ", "MGC", "MCL", "MYM", "M2K"}
)

# Symbols Alpaca cannot service (futures). Listed here so the router /
# adapter can refuse with a consistent message regardless of which path
# tried to place the order.
FUTURES_SYMBOLS: frozenset[str] = SUPPORTED_SYMBOLS


def _position_payload(p: PositionState) -> dict[str, Any]:
    return {
        "symbol": p.symbol,
        "quantity": p.quantity,
        "direction": p.direction,
        "average_price": p.average_price,
        "unrealized_pnl": p.unrealized_pnl,
    }


def _order_payload(o: OpenOrder) -> dict[str, Any]:
    return {
        "order_id": o.order_id,
        "symbol": o.symbol,
        "side": o.side,
        "quantity": o.quantity,
        "order_type": o.order_type,
        "status": o.status,
        "limit_price": o.limit_price,
        "stop_price": o.stop_price,
        "time_in_force": o.time_in_force,
    }


def redact_secrets(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip credential-shaped fields from a payload before logging."""
    blocklist = {
        "password",
        "secret",
        "client_secret",
        "access_token",
        "md_access_token",
        "authorization",
        "api_key",
    }
    cleaned: dict[str, Any] = {}
    for k, v in payload.items():
        if k.lower() in blocklist:
            cleaned[k] = "***"
            continue
        if isinstance(v, dict):
            cleaned[k] = redact_secrets(v)
        else:
            cleaned[k] = v
    return cleaned
