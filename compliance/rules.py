"""Generic compliance checks (HFT detection, hold-time concentration).

These are *post-hoc* analyses run on the closed trade ledger. They flag
violations on the report; they do not block trades in real time. Rules
that *should* block in real time live in ``risk/risk_engine.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from backtesting.portfolio import ClosedTradeRecord


HFT_HOLD_THRESHOLD_SECONDS = 10
HFT_TRADE_FRACTION_LIMIT = 0.50
HFT_PROFIT_FRACTION_LIMIT = 0.50


@dataclass
class ComplianceFlag:
    rule: str
    triggered: bool
    detail: str
    value: float


def hold_time_seconds(trade: ClosedTradeRecord) -> float:
    return float((trade.exit_ts - trade.entry_ts).total_seconds())


def hft_trade_fraction(trades: list[ClosedTradeRecord]) -> ComplianceFlag:
    """Flag if more than 50% of trades are held under 10 seconds."""
    if not trades:
        return ComplianceFlag("hft_trade_fraction", False, "no trades", 0.0)
    fast = [t for t in trades if hold_time_seconds(t) < HFT_HOLD_THRESHOLD_SECONDS]
    frac = len(fast) / len(trades)
    triggered = frac > HFT_TRADE_FRACTION_LIMIT
    return ComplianceFlag(
        rule="hft_trade_fraction",
        triggered=triggered,
        detail=(
            f"{len(fast)}/{len(trades)} trades held < {HFT_HOLD_THRESHOLD_SECONDS}s "
            f"({frac:.1%}); limit = {HFT_TRADE_FRACTION_LIMIT:.0%}."
        ),
        value=float(frac),
    )


def hft_profit_fraction(trades: list[ClosedTradeRecord]) -> ComplianceFlag:
    """Flag if >50% of TOTAL POSITIVE pnl came from sub-10-second trades."""
    pos_total = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    if pos_total <= 0:
        return ComplianceFlag(
            "hft_profit_fraction", False, "no positive pnl to attribute", 0.0
        )
    fast_pos = sum(
        t.net_pnl
        for t in trades
        if t.net_pnl > 0 and hold_time_seconds(t) < HFT_HOLD_THRESHOLD_SECONDS
    )
    frac = fast_pos / pos_total
    triggered = frac > HFT_PROFIT_FRACTION_LIMIT
    return ComplianceFlag(
        rule="hft_profit_fraction",
        triggered=triggered,
        detail=(
            f"{fast_pos:.2f}/{pos_total:.2f} winning $ from <"
            f"{HFT_HOLD_THRESHOLD_SECONDS}s trades ({frac:.1%}); "
            f"limit = {HFT_PROFIT_FRACTION_LIMIT:.0%}."
        ),
        value=float(frac),
    )


def general_compliance_flags(trades: list[ClosedTradeRecord]) -> list[ComplianceFlag]:
    return [hft_trade_fraction(trades), hft_profit_fraction(trades)]
