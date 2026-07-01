"""In-memory mock broker for DRY_RUN and tests (never touches the network)."""

from __future__ import annotations

import time
import uuid
from typing import Literal, Optional

from app.logging_config import get_logger
from integrations.broker_base import (
    AccountState,
    BaseBroker,
    ExitFill,
    OpenOrder,
    OrderResult,
    OrderSide,
    PositionState,
    Quote,
    SUPPORTED_SYMBOLS,
    TimeInForce,
    ValidationResult,
)


_DEFAULT_QUOTES: dict[str, float] = {
    "MES": 5050.00,
    "MNQ": 17500.00,
    "ES": 5050.00,
    "NQ": 17500.00,
    "MGC": 2350.00,
    "MCL": 78.50,
    "MYM": 38500.00,
    "M2K": 2050.00,
    # Equities / ETFs
    "SPY": 550.00,
    "QQQ": 480.00,
    "AAPL": 210.00,
    "MSFT": 440.00,
    "IWM": 220.00,
    "NVDA": 130.00,
    "TSLA": 250.00,
    "AMD": 160.00,
}

# Fallback price for an enabled symbol with no seeded quote, so the
# mock broker can service equities/options it wasn't explicitly told
# about without raising.
_FALLBACK_QUOTE = 100.0


class MockBroker(BaseBroker):
    """Pure in-memory adapter — used by DRY_RUN and unit tests."""

    provider_name = "mock"

    def __init__(
        self,
        *,
        enabled_symbols: Optional[list[str]] = None,
        seed_quotes: Optional[dict[str, float]] = None,
    ) -> None:
        self.log = get_logger("integrations.mock_broker")
        self.enabled_symbols = {
            s.upper() for s in (enabled_symbols or list(SUPPORTED_SYMBOLS))
        }
        self._quotes: dict[str, float] = {**_DEFAULT_QUOTES, **(seed_quotes or {})}
        self._positions: dict[str, PositionState] = {}
        self._open_orders: dict[str, OpenOrder] = {}
        self._last_exit_fills: dict[str, ExitFill] = {}
        self._account_id = "MOCK-DEMO-1"

    # ----------------------- account state ----------------------------
    def get_account(self) -> AccountState:
        return AccountState(
            account_id=self._account_id,
            cash_balance=50_000.0,
            buying_power=50_000.0,
            equity=50_000.0,
        )

    def get_positions(self) -> list[PositionState]:
        return list(self._positions.values())

    def get_open_orders(self) -> list[OpenOrder]:
        return list(self._open_orders.values())

    def get_latest_quote(self, symbol: str) -> Quote:
        sym = symbol.upper()
        last = self._quotes.get(sym)
        if last is None:
            # Mint a synthetic quote for any enabled symbol the mock
            # wasn't seeded with (equities/options the futures registry
            # doesn't know about). Unknown + not enabled → raise.
            if sym in self.enabled_symbols:
                last = _FALLBACK_QUOTE
            else:
                raise KeyError(f"No mock quote for symbol {sym!r}")
        return Quote(
            symbol=sym,
            bid=last - 0.25,
            ask=last + 0.25,
            last=last,
            timestamp_ms=int(time.time() * 1000),
        )

    def validate_order(
        self, *, symbol: str, qty: float, side: OrderSide
    ) -> ValidationResult:
        sym = symbol.upper()
        if qty <= 0:
            return ValidationResult(valid=False, reason="qty must be > 0")
        # The mock broker is asset-class-agnostic: it services any symbol
        # the operator enabled (futures, equities, options). The futures
        # SUPPORTED_SYMBOLS set is no longer a hard gate here.
        if sym not in self.enabled_symbols:
            return ValidationResult(valid=False, reason=f"symbol {sym} not enabled")
        try:
            quote = self.get_latest_quote(sym)
        except KeyError:
            return ValidationResult(valid=False, reason=f"no quote for {sym}")
        return ValidationResult(valid=True, quote=quote)

    # --------------------------- orders ------------------------------
    def place_market_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        return self._record_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="market",
            time_in_force=time_in_force,
            fill=True,
        )

    def place_limit_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        limit_price: float,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        return self._record_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="limit",
            limit_price=limit_price,
            time_in_force=time_in_force,
            fill=False,
        )

    def place_stop_order(
        self,
        *,
        symbol: str,
        qty: float,
        stop_price: float,
        side: OrderSide,
        time_in_force: TimeInForce = "day",
    ) -> OrderResult:
        return self._record_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="stop",
            stop_price=stop_price,
            time_in_force=time_in_force,
            fill=False,
        )

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
        # Simulate a filled bracket entry; carry the protective legs in raw.
        result = self._record_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="limit",
            limit_price=entry_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            fill=True,
        )
        result.raw.update(
            {
                "order_class": "bracket",
                "take_profit": target_price,
                "stop_loss": stop_price,
            }
        )
        return result

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
        result = self._record_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="trailing_stop",
            time_in_force=time_in_force,
            fill=False,
        )
        result.raw.update({"trail": trail, "trail_kind": trail_kind})
        return result

    def close_position(self, *, symbol: str) -> OrderResult:
        sym = symbol.upper()
        pos = self._positions.pop(sym, None)
        if pos is None:
            return OrderResult(
                success=False,
                simulated=True,
                order_id=uuid.uuid4().hex,
                symbol=sym,
                side="sell",
                quantity=0.0,
                order_type="market",
                status="rejected",
                reason="no_position",
            )
        side: OrderSide = "sell" if pos.direction == "long" else "buy"
        fill_price = self._quotes.get(sym)
        if fill_price is not None:
            self._last_exit_fills[sym] = ExitFill(
                symbol=sym,
                price=float(fill_price),
                quantity=abs(pos.quantity),
                side=side,
                exit_kind="closed",
            )
        return OrderResult(
            success=True,
            simulated=True,
            order_id=uuid.uuid4().hex,
            symbol=sym,
            side=side,
            quantity=abs(pos.quantity),
            order_type="market",
            status="filled",
            fill_price=fill_price,
            reason="close_position",
        )

    def get_last_exit_fill(
        self, *, symbol: str, exit_side: OrderSide
    ) -> Optional[ExitFill]:
        fill = self._last_exit_fills.get(symbol.upper())
        if fill is not None and fill.side == exit_side:
            return fill
        return None

    def cancel_order(self, *, order_id: str) -> OrderResult:
        existing = self._open_orders.pop(order_id, None)
        if existing is None:
            return OrderResult(
                success=False,
                simulated=True,
                order_id=order_id,
                symbol="",
                side="buy",
                quantity=0.0,
                order_type="market",
                status="rejected",
                reason="unknown_order_id",
            )
        return OrderResult(
            success=True,
            simulated=True,
            order_id=existing.order_id,
            symbol=existing.symbol,
            side=existing.side,
            quantity=existing.quantity,
            order_type=existing.order_type,
            status="cancelled",
        )

    # --------------------------- internals ----------------------------
    def _record_order(
        self,
        *,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: str,
        time_in_force: TimeInForce,
        fill: bool,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> OrderResult:
        sym = symbol.upper()
        order_id = uuid.uuid4().hex
        if fill:
            quote = self._quotes.get(sym)
            self._positions[sym] = PositionState(
                symbol=sym,
                quantity=qty if side == "buy" else -qty,
                average_price=quote or 0.0,
                direction="long" if side == "buy" else "short",
            )
            return OrderResult(
                success=True,
                simulated=True,
                order_id=order_id,
                symbol=sym,
                side=side,
                quantity=qty,
                order_type=order_type,  # type: ignore[arg-type]
                status="filled",
                limit_price=limit_price,
                stop_price=stop_price,
                fill_price=quote,
                time_in_force=time_in_force,
            )

        self._open_orders[order_id] = OpenOrder(
            order_id=order_id,
            symbol=sym,
            side=side,
            quantity=qty,
            order_type=order_type,  # type: ignore[arg-type]
            status="working",
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )
        return OrderResult(
            success=True,
            simulated=True,
            order_id=order_id,
            symbol=sym,
            side=side,
            quantity=qty,
            order_type=order_type,  # type: ignore[arg-type]
            status="working",
            limit_price=limit_price,
            stop_price=stop_price,
            time_in_force=time_in_force,
        )
