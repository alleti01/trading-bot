"""Options contract model + OCC symbol encoding/decoding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Optional

OptionType = Literal["call", "put"]
OptionAction = Literal[
    "buy_to_open",
    "sell_to_close",
    "sell_to_open",
    "buy_to_close",
]

# OCC symbol: ROOT(≤6) + YYMMDD + C/P + strike*1000 zero-padded to 8.
# Example: SPY260718C00550000  → SPY 2026-07-18 Call $550.00
_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])"
    r"(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class OptionContract:
    """A single option contract.

    ``occ_symbol`` is the canonical broker identifier. Greeks are
    optional — they're populated from the chain when available and used
    by the position manager / risk rules.
    """

    underlying: str
    expiry: date
    strike: float
    option_type: OptionType
    occ_symbol: str
    # Market data (optional; filled from the chain)
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    # Greeks (optional)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    implied_volatility: Optional[float] = None
    open_interest: Optional[int] = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def mid_price(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return round((self.bid + self.ask) / 2.0, 4)
        return self.last

    def days_to_expiry(self, *, now: Optional[datetime] = None) -> int:
        ref = (now or datetime.now()).date()
        return (self.expiry - ref).days

    def to_payload(self) -> dict[str, Any]:
        return {
            "underlying": self.underlying,
            "occ_symbol": self.occ_symbol,
            "expiry": self.expiry.isoformat(),
            "strike": self.strike,
            "option_type": self.option_type,
            "bid": self.bid,
            "ask": self.ask,
            "mid_price": self.mid_price,
            "delta": self.delta,
            "theta": self.theta,
            "implied_volatility": self.implied_volatility,
            "days_to_expiry": self.days_to_expiry(),
        }


def build_occ_symbol(
    underlying: str, expiry: date, strike: float, option_type: OptionType
) -> str:
    root = underlying.upper()
    cp = "C" if option_type == "call" else "P"
    strike_int = int(round(strike * 1000))
    return f"{root}{expiry:%y%m%d}{cp}{strike_int:08d}"


def parse_occ_symbol(occ_symbol: str) -> OptionContract:
    m = _OCC_RE.match(occ_symbol.strip().upper())
    if not m:
        raise ValueError(f"Invalid OCC option symbol: {occ_symbol!r}")
    root = m.group("root")
    yy = int(m.group("yy"))
    mm = int(m.group("mm"))
    dd = int(m.group("dd"))
    expiry = date(2000 + yy, mm, dd)
    option_type: OptionType = "call" if m.group("cp") == "C" else "put"
    strike = int(m.group("strike")) / 1000.0
    return OptionContract(
        underlying=root,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        occ_symbol=occ_symbol.strip().upper(),
    )
