"""Tradeify-style consistency + overnight + forced-flat compliance checks.

Run after a backtest. These rules are designed in the spirit of prop-firm
discipline; the bot does not assume Tradeify is its broker, but the
checks are useful for any account that wants to avoid:

- Overnight futures exposure.
- Outsized single-day profits inflating an apparent track record.
- Sloppy session discipline (entries after the configured flat time).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

from backtesting.portfolio import ClosedTradeRecord


@dataclass
class TradeifyFlag:
    rule: str
    triggered: bool
    detail: str
    value: float


def consistency_rule(
    trades: list[ClosedTradeRecord],
    *,
    limit_percent: float,
) -> TradeifyFlag:
    """``max(daily_profit) / total_profit < limit`` when total_profit > 0."""
    if not trades:
        return TradeifyFlag("consistency", False, "no trades", 0.0)

    daily: dict[str, float] = {}
    for t in trades:
        d = t.exit_ts.date().isoformat()
        daily[d] = daily.get(d, 0.0) + t.net_pnl

    total = sum(daily.values())
    if total <= 0:
        return TradeifyFlag(
            "consistency",
            False,
            "total profit non-positive; consistency rule N/A",
            0.0,
        )
    biggest = max(daily.values())
    pct = (biggest / total) * 100.0
    triggered = pct >= limit_percent
    return TradeifyFlag(
        rule="consistency",
        triggered=triggered,
        detail=(
            f"largest day = {biggest:.2f} of total {total:.2f} "
            f"({pct:.1f}% vs limit {limit_percent:.1f}%)."
        ),
        value=float(pct),
    )


def no_overnight_futures(
    trades: list[ClosedTradeRecord],
    *,
    market_type: str,
    timezone: str,
    session_close: time,
) -> TradeifyFlag:
    """Trade is non-compliant if futures exit_ts is on a different
    LOCAL date than entry_ts, OR exit_ts is after session_close on the
    entry date."""
    if market_type != "futures":
        return TradeifyFlag(
            "no_overnight_futures",
            False,
            "market_type is not futures; rule N/A",
            0.0,
        )
    if not trades:
        return TradeifyFlag("no_overnight_futures", False, "no trades", 0.0)

    tz = ZoneInfo(timezone)
    bad: list[ClosedTradeRecord] = []
    for t in trades:
        e = t.entry_ts.astimezone(tz)
        x = t.exit_ts.astimezone(tz)
        if e.date() != x.date() or x.time() > session_close:
            bad.append(t)
    triggered = len(bad) > 0
    return TradeifyFlag(
        rule="no_overnight_futures",
        triggered=triggered,
        detail=(
            f"{len(bad)} trade(s) violated overnight rule "
            f"(session_close={session_close})."
        ),
        value=float(len(bad)),
    )


def forced_flat_compliance(
    trades: list[ClosedTradeRecord],
    *,
    market_type: str,
    timezone: str,
    flat_time: time,
) -> TradeifyFlag:
    """Trade entries must occur strictly before ``flat_time`` (LOCAL)."""
    if market_type != "futures":
        return TradeifyFlag("forced_flat", False, "market_type is not futures", 0.0)
    if not trades:
        return TradeifyFlag("forced_flat", False, "no trades", 0.0)

    tz = ZoneInfo(timezone)
    bad = [t for t in trades if t.entry_ts.astimezone(tz).time() >= flat_time]
    triggered = len(bad) > 0
    return TradeifyFlag(
        rule="forced_flat",
        triggered=triggered,
        detail=f"{len(bad)} trade(s) entered at/after FORCE_FLAT_TIME={flat_time}.",
        value=float(len(bad)),
    )


def tradeify_compliance_flags(
    trades: list[ClosedTradeRecord],
    *,
    market_type: str,
    timezone: str,
    session_close: time,
    flat_time: time,
    consistency_limit_percent: float,
) -> list[TradeifyFlag]:
    return [
        consistency_rule(trades, limit_percent=consistency_limit_percent),
        no_overnight_futures(
            trades,
            market_type=market_type,
            timezone=timezone,
            session_close=session_close,
        ),
        forced_flat_compliance(
            trades,
            market_type=market_type,
            timezone=timezone,
            flat_time=flat_time,
        ),
    ]
