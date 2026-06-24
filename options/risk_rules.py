"""Deterministic options risk rules (advisory gates, never trade)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from options.contract import OptionContract


@dataclass(frozen=True)
class OptionsRiskConfig:
    max_premium_per_trade: float = 500.0
    max_open_positions: int = 5
    min_dte: int = 7
    max_dte: int = 60
    min_open_interest: int = 0
    max_iv: Optional[float] = None  # block if IV above this (e.g. 1.5 == 150%)


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    reason: str = ""


class OptionsRiskEngine:
    """Pure, deterministic gate. Does not place or modify orders."""

    def __init__(self, config: Optional[OptionsRiskConfig] = None) -> None:
        self.config = config or OptionsRiskConfig()

    def evaluate_entry(
        self,
        contract: OptionContract,
        *,
        qty: int,
        open_positions: int,
        now=None,
    ) -> RiskDecision:
        c = self.config
        if qty <= 0:
            return RiskDecision(False, "qty must be > 0")
        if open_positions >= c.max_open_positions:
            return RiskDecision(False, f"max_open_positions reached ({c.max_open_positions})")

        dte = contract.days_to_expiry(now=now)
        if dte < c.min_dte:
            return RiskDecision(False, f"DTE {dte} < min_dte {c.min_dte}")
        if dte > c.max_dte:
            return RiskDecision(False, f"DTE {dte} > max_dte {c.max_dte}")

        price = contract.mid_price
        if price is None:
            return RiskDecision(False, "no price for contract")
        # Options are priced per share; one contract == 100 shares.
        premium = price * 100.0 * qty
        if premium > c.max_premium_per_trade:
            return RiskDecision(
                False,
                f"premium ${premium:.2f} > max ${c.max_premium_per_trade:.2f}",
            )

        if c.min_open_interest and (contract.open_interest or 0) < c.min_open_interest:
            return RiskDecision(
                False,
                f"open_interest {contract.open_interest} < min {c.min_open_interest}",
            )
        if (
            c.max_iv is not None
            and contract.implied_volatility is not None
            and contract.implied_volatility > c.max_iv
        ):
            return RiskDecision(
                False,
                f"IV {contract.implied_volatility:.2f} > max {c.max_iv:.2f}",
            )
        return RiskDecision(True, "ok")

    def estimate_premium(self, contract: OptionContract, *, qty: int) -> Optional[float]:
        price = contract.mid_price
        if price is None:
            return None
        return round(price * 100.0 * qty, 2)
