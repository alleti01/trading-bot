"""Black-Scholes greeks (deterministic, no external deps).

Used to estimate greeks when the chain provider does not supply them,
and by the position manager / risk rules. Pure-Python normal CDF/PDF so
there is no SciPy dependency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    theta: float
    vega: float
    price: float


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_greeks(
    *,
    spot: float,
    strike: float,
    days_to_expiry: float,
    volatility: float,
    option_type: str,
    risk_free_rate: float = 0.04,
) -> Optional[Greeks]:
    """Return Black-Scholes greeks, or ``None`` for degenerate inputs.

    ``days_to_expiry`` is calendar days; ``volatility`` is annualized
    (e.g. 0.20 for 20% IV). Theta is returned per calendar day.
    """
    if spot <= 0 or strike <= 0 or volatility <= 0 or days_to_expiry <= 0:
        return None
    t = days_to_expiry / 365.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * t) / (
        volatility * sqrt_t
    )
    d2 = d1 - volatility * sqrt_t

    is_call = option_type == "call"
    if is_call:
        delta = _norm_cdf(d1)
        price = spot * _norm_cdf(d1) - strike * math.exp(-risk_free_rate * t) * _norm_cdf(d2)
        theta_annual = (
            -(spot * _norm_pdf(d1) * volatility) / (2 * sqrt_t)
            - risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(d2)
        )
    else:
        delta = _norm_cdf(d1) - 1.0
        price = strike * math.exp(-risk_free_rate * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        theta_annual = (
            -(spot * _norm_pdf(d1) * volatility) / (2 * sqrt_t)
            + risk_free_rate * strike * math.exp(-risk_free_rate * t) * _norm_cdf(-d2)
        )

    gamma = _norm_pdf(d1) / (spot * volatility * sqrt_t)
    vega = spot * _norm_pdf(d1) * sqrt_t / 100.0  # per 1 vol point
    theta = theta_annual / 365.0  # per calendar day

    return Greeks(
        delta=round(delta, 6),
        gamma=round(gamma, 6),
        theta=round(theta, 6),
        vega=round(vega, 6),
        price=round(max(price, 0.0), 4),
    )
