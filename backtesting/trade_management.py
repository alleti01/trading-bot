"""Shared trade-exit decision logic.

Both the backtest engine (``backtesting/engine.py``) and the live paper
loop (``paper/loop.py``) must apply identical exit rules — otherwise a
strategy that looks profitable in backtest can drift in paper trading.
This module is the single source of truth.

Exit precedence (matches the labeler on Day 3 and the original engine):

1. **Forced flat** (futures only). Once local time hits ``force_flat_time``
   on the current bar, close the position at the bar's open, no questions
   asked. Reason ``"forced_flat"``.
2. **Same-bar TP and SL**: stop wins (conservative tie-break, label 0).
3. **SL-only** hit: exit at stop price.
4. **TP-only** hit: exit at target price.
5. **Time-out**: ``bars_held >= max_hold_bars`` → exit at bar's close.

If none of the above triggers the function returns ``None`` and the
caller leaves the position open.

The function is **pure**: it reads bar OHLC and a few config knobs and
emits an ``ExitDecision``. ``apply_exit`` is the I/O side that calls
``FillsModel.exit`` and ``Portfolio.close``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from backtesting.fills import FillsModel
from backtesting.portfolio import ClosedTradeRecord, OpenPosition, Portfolio


ExitReason = Literal["tp", "sl", "time", "forced_flat", "end_of_data", "manual"]


@dataclass(frozen=True)
class ExitDecision:
    """Outcome of ``check_exit``: what to close at, and why."""

    reason: ExitReason
    raw_price: float


def check_exit(
    *,
    position: OpenPosition,
    bar: pd.Series,
    bar_index: int,
    bar_ts: datetime,
    max_hold_bars: int,
    force_flat_time: time,
    market_type: str,
    tz: ZoneInfo,
) -> Optional[ExitDecision]:
    """Decide whether a position must close on this bar.

    Returns ``None`` if the position should keep running.
    """
    if position.direction not in ("long", "short"):
        raise ValueError(f"Unknown direction {position.direction!r}")

    # 1) Forced flat (futures only). The risk engine already prevents new
    #    entries past flat time; this handles the already-open position.
    if market_type == "futures":
        local_now = bar_ts.astimezone(tz).time()
        if local_now >= force_flat_time:
            return ExitDecision(reason="forced_flat", raw_price=float(bar["open"]))

    high = float(bar["high"])
    low = float(bar["low"])
    is_long = position.direction == "long"
    tp_hit = (high >= position.target_price) if is_long else (low <= position.target_price)
    sl_hit = (low <= position.stop_price) if is_long else (high >= position.stop_price)

    # 2) Same-bar tie → SL first.
    if tp_hit and sl_hit:
        return ExitDecision(reason="sl", raw_price=position.stop_price)
    if sl_hit:
        return ExitDecision(reason="sl", raw_price=position.stop_price)
    if tp_hit:
        return ExitDecision(reason="tp", raw_price=position.target_price)

    # 3) Time-out. ``bars_held`` is "bars elapsed since entry"; the entry
    #    bar itself is bar 0, so time-out fires on the (max_hold_bars)th bar.
    if position.bars_held(bar_index) >= max_hold_bars:
        return ExitDecision(reason="time", raw_price=float(bar["close"]))

    return None


def apply_exit(
    *,
    portfolio: Portfolio,
    fills: FillsModel,
    decision: ExitDecision,
    ts: datetime,
    bar_index: int,
) -> ClosedTradeRecord:
    """Apply an ``ExitDecision``: compute fill, mutate portfolio, return record."""
    pos = portfolio.open_position
    if pos is None:
        raise RuntimeError("apply_exit called with no open position")

    costs = fills.exit(
        direction=pos.direction,
        raw_price=decision.raw_price,
        quantity=pos.quantity,
    )
    return portfolio.close(
        ts=ts,
        exit_price=costs.fill_price,
        exit_reason=decision.reason,
        commission=costs.commission,
        slippage=costs.slippage,
        bar_index=bar_index,
    )
