"""Helpers for building closed-trade notification payloads.

A trade close is the moment the user most wants to hear about: *did I make
or lose money, and how much?* The raw :class:`ClosedTradeRecord` already
carries everything needed, so this module turns it into a single, rich
payload that every close path (paper loop TP/SL/time exits, end-of-day
forced flatten, and TradingView webhook closes) can reuse. Keeping the
shape consistent lets the Discord layer render one nice "Trade Closed"
message regardless of which path fired it.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backtesting.portfolio import ClosedTradeRecord


def classify_result(net_pnl: float) -> str:
    """Map a net dollar PnL to a human verdict."""
    if net_pnl > 0:
        return "WIN"
    if net_pnl < 0:
        return "LOSS"
    return "BREAKEVEN"


def build_trade_closed_payload(
    record: "ClosedTradeRecord",
    *,
    ts: Optional[datetime] = None,
    symbol_key: str = "instrument",
) -> dict[str, Any]:
    """Build a rich, human-readable payload for a closed-trade notification.

    The payload tells the user *what* closed and *how much* they gained or
    lost: realised net PnL (dollars), the win/loss verdict, the price move
    as a percentage, entry/exit prices, size, holding period and fees.

    ``symbol_key`` lets the webhook path use ``symbol`` while the paper loop
    uses ``instrument`` — both render identically downstream.
    """
    net = float(record.net_pnl)
    gross = float(record.gross_pnl)
    entry = float(record.entry_price)
    exit_price = float(record.exit_price)

    sign = 1.0 if record.direction == "long" else -1.0
    price_return_pct = (sign * (exit_price - entry) / entry * 100.0) if entry else 0.0
    costs = float(record.commission) + float(record.slippage)
    when = ts if ts is not None else record.exit_ts

    return {
        symbol_key: record.instrument,
        "direction": record.direction,
        "result": classify_result(net),
        "exit_reason": record.exit_reason,
        "net_pnl": round(net, 2),
        "gross_pnl": round(gross, 2),
        "return_pct": round(price_return_pct, 2),
        "entry_price": round(entry, 4),
        "exit_price": round(exit_price, 4),
        "quantity": record.quantity,
        "bars_held": record.bars_held,
        "costs": round(costs, 2),
        "ts": when.isoformat() if hasattr(when, "isoformat") else str(when),
    }
