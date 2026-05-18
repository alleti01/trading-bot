"""Position book + day state + equity curve.

The Portfolio is a *state container*. It does not enforce risk rules
(that's ``risk/risk_engine.py``) and it does not decide when to enter
or exit (that's ``backtesting/engine.py``). Its job is to:

- Track at most one open position at a time (no hedging by construction).
- Record closed trades for downstream metrics + compliance.
- Roll over per-day stats so rules like ``MAX_TRADES_PER_DAY`` and
  ``MAX_DAILY_LOSS`` have something to read.
- Maintain an equity curve sampled at trade close.

PnL convention
--------------
PnL is in **dollars**, computed as
``(exit_price - entry_price) * point_value * quantity`` for longs and
the negation for shorts. Commissions are subtracted to produce ``net_pnl``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from config.instruments import InstrumentSpec


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class OpenPosition:
    setup_id: str
    instrument: str
    direction: str  # "long" | "short"
    quantity: float
    entry_ts: datetime
    entry_price: float
    stop_price: float
    target_price: float
    entry_commission: float
    entry_slippage: float
    entry_bar_index: int

    def bars_held(self, current_bar_index: int) -> int:
        return max(0, current_bar_index - self.entry_bar_index)


@dataclass
class ClosedTradeRecord:
    setup_id: str
    instrument: str
    direction: str
    quantity: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str  # "tp" | "sl" | "time" | "flat" | "manual"
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    bars_held: int


@dataclass
class DayState:
    day: date
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    net_pnl: float = 0.0
    last_trade_ts: Optional[datetime] = None
    last_trade_net_pnl: Optional[float] = None
    cooldown_until_ts: Optional[datetime] = None  # set by engine, read by risk

    def reset(self, new_day: date) -> None:
        self.day = new_day
        self.trades = 0
        self.wins = 0
        self.losses = 0
        self.gross_pnl = 0.0
        self.net_pnl = 0.0
        self.last_trade_ts = None
        self.last_trade_net_pnl = None
        self.cooldown_until_ts = None


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------
class Portfolio:
    def __init__(self, *, instrument_spec: InstrumentSpec, starting_equity: float = 0.0) -> None:
        self.spec = instrument_spec
        self.starting_equity = float(starting_equity)
        self.equity = float(starting_equity)
        self.open_position: Optional[OpenPosition] = None
        self.closed_trades: list[ClosedTradeRecord] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self.day: DayState = DayState(day=date(1970, 1, 1))

    # ---------------------------- queries ---------------------------------
    def is_flat(self) -> bool:
        return self.open_position is None

    def has_long(self) -> bool:
        return self.open_position is not None and self.open_position.direction == "long"

    def has_short(self) -> bool:
        return self.open_position is not None and self.open_position.direction == "short"

    # ---------------------------- mutations -------------------------------
    def maybe_roll_day(self, ts: datetime) -> bool:
        """If ``ts`` is on a different calendar day, reset day stats."""
        new_day = ts.date()
        if new_day != self.day.day:
            self.day.reset(new_day)
            return True
        return False

    def open(
        self,
        *,
        setup_id: str,
        instrument: str,
        direction: str,
        quantity: float,
        ts: datetime,
        entry_price: float,
        stop_price: float,
        target_price: float,
        commission: float,
        slippage: float,
        bar_index: int,
    ) -> OpenPosition:
        if self.open_position is not None:
            raise RuntimeError(
                f"Cannot open new position while {self.open_position.direction} is open."
            )
        if direction not in ("long", "short"):
            raise ValueError(f"Unknown direction {direction!r}")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        self.open_position = OpenPosition(
            setup_id=setup_id,
            instrument=instrument,
            direction=direction,
            quantity=float(quantity),
            entry_ts=ts,
            entry_price=float(entry_price),
            stop_price=float(stop_price),
            target_price=float(target_price),
            entry_commission=float(commission),
            entry_slippage=float(slippage),
            entry_bar_index=int(bar_index),
        )
        return self.open_position

    def close(
        self,
        *,
        ts: datetime,
        exit_price: float,
        exit_reason: str,
        commission: float,
        slippage: float,
        bar_index: int,
    ) -> ClosedTradeRecord:
        pos = self.open_position
        if pos is None:
            raise RuntimeError("Cannot close: no open position.")

        sign = 1.0 if pos.direction == "long" else -1.0
        gross_pnl = (
            sign * (float(exit_price) - pos.entry_price) * self.spec.point_value * pos.quantity
        )
        total_commission = pos.entry_commission + float(commission)
        total_slippage = pos.entry_slippage + float(slippage)
        net_pnl = gross_pnl - total_commission

        record = ClosedTradeRecord(
            setup_id=pos.setup_id,
            instrument=pos.instrument,
            direction=pos.direction,
            quantity=pos.quantity,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            exit_ts=ts,
            exit_price=float(exit_price),
            exit_reason=exit_reason,
            gross_pnl=gross_pnl,
            commission=total_commission,
            slippage=total_slippage,
            net_pnl=net_pnl,
            bars_held=max(0, int(bar_index) - pos.entry_bar_index),
        )

        # Update aggregates.
        self.maybe_roll_day(ts)
        self.day.trades += 1
        self.day.gross_pnl += gross_pnl
        self.day.net_pnl += net_pnl
        if net_pnl > 0:
            self.day.wins += 1
        elif net_pnl < 0:
            self.day.losses += 1
        self.day.last_trade_ts = ts
        self.day.last_trade_net_pnl = net_pnl

        self.closed_trades.append(record)
        self.equity += net_pnl
        self.equity_curve.append((ts, self.equity))

        self.open_position = None
        return record
