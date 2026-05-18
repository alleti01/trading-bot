"""Instrument metadata: tick size, tick value, point value, etc.

Day 1 stub. Full implementations land alongside the data loader (Day 2).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    market_type: str          # "futures" | "crypto"
    tick_size: float          # smallest price increment
    tick_value: float         # $ value per tick per contract / unit
    point_value: float        # $ value per 1.00 price move
    session_tz: str           # timezone for trading session
    session_open: str         # HH:MM
    session_close: str        # HH:MM


# Minimal seed registry — extended on Day 2.
_REGISTRY: dict[str, InstrumentSpec] = {
    "MES": InstrumentSpec(
        symbol="MES",
        market_type="futures",
        tick_size=0.25,
        tick_value=1.25,
        point_value=5.0,
        session_tz="America/New_York",
        session_open="09:30",
        session_close="16:00",
    ),
    "MNQ": InstrumentSpec(
        symbol="MNQ",
        market_type="futures",
        tick_size=0.25,
        tick_value=0.50,
        point_value=2.0,
        session_tz="America/New_York",
        session_open="09:30",
        session_close="16:00",
    ),
    "BTC": InstrumentSpec(
        symbol="BTC",
        market_type="crypto",
        tick_size=0.01,
        tick_value=0.01,
        point_value=1.0,
        session_tz="UTC",
        session_open="00:00",
        session_close="23:59",
    ),
}


def get_instrument(symbol: str) -> InstrumentSpec:
    try:
        return _REGISTRY[symbol.upper()]
    except KeyError as e:
        raise KeyError(
            f"Unknown instrument '{symbol}'. Known: {sorted(_REGISTRY)}"
        ) from e
