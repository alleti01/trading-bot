"""Risk-based position sizing.

Standard formula: contracts = floor(RISK_PER_TRADE / risk_per_contract),
where risk_per_contract is ``|entry - stop| * point_value``.

Capped at ``MAX_POSITION_SIZE`` and floored at 1 (the bot never tries to
"size to zero"; if the geometric stop is so far away that 1 contract
exceeds RISK_PER_TRADE, the *risk engine* should reject the setup
upstream — sizing returns 1 + a flag that the caller can act on).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from config.instruments import InstrumentSpec, get_instrument


@dataclass(frozen=True)
class SizingResult:
    quantity: int
    risk_per_contract_dollars: float
    expected_risk_dollars: float
    exceeds_risk_per_trade: bool


def size_position(
    *,
    entry_price: float,
    stop_price: float,
    instrument: str | InstrumentSpec,
    risk_per_trade: float,
    max_position_size: int,
) -> SizingResult:
    if risk_per_trade <= 0:
        raise ValueError("risk_per_trade must be positive")
    if max_position_size < 1:
        raise ValueError("max_position_size must be >= 1")

    spec = instrument if isinstance(instrument, InstrumentSpec) else get_instrument(instrument)

    distance = abs(float(entry_price) - float(stop_price))
    if distance <= 0:
        raise ValueError("entry_price and stop_price must differ")

    risk_per_contract = distance * spec.point_value
    if risk_per_contract <= 0:
        raise ValueError("risk_per_contract is non-positive (check point_value/prices)")

    raw_qty = math.floor(risk_per_trade / risk_per_contract)
    qty = max(1, min(int(raw_qty), int(max_position_size)))
    expected_risk = qty * risk_per_contract
    exceeds = expected_risk > risk_per_trade + 1e-9

    return SizingResult(
        quantity=qty,
        risk_per_contract_dollars=risk_per_contract,
        expected_risk_dollars=expected_risk,
        exceeds_risk_per_trade=exceeds,
    )


def size_equity_shares(
    *,
    entry_price: float,
    stop_price: float,
    instrument: str | InstrumentSpec,
    risk_per_trade: float,
    max_shares: int,
    max_notional: float,
) -> SizingResult:
    """Risk-based share count for equities, capped by shares + notional.

    shares = floor(risk_per_trade / per-share risk), then clamped to
    ``max_shares`` and ``floor(max_notional / entry_price)`` so a trade
    never risks more than ``risk_per_trade`` nor spends more than
    ``max_notional``. Always returns at least 1 share.
    """
    base = size_position(
        entry_price=entry_price,
        stop_price=stop_price,
        instrument=instrument,
        risk_per_trade=risk_per_trade,
        max_position_size=max(1, int(max_shares)),
    )
    qty = base.quantity
    if entry_price > 0:
        notional_cap = int(max_notional // float(entry_price))
        qty = min(qty, max(1, notional_cap))
    qty = max(1, qty)
    expected_risk = qty * base.risk_per_contract_dollars
    return SizingResult(
        quantity=qty,
        risk_per_contract_dollars=base.risk_per_contract_dollars,
        expected_risk_dollars=expected_risk,
        exceeds_risk_per_trade=expected_risk > risk_per_trade + 1e-9,
    )
