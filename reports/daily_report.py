"""Daily PnL + risk + compliance summary.

Day 5 ships the **minimal** EOD summary used by the scheduler/Discord:
counts of trades, risk blocks, net PnL since the start of the local
session date. The full Markdown EOD report (per-trade journal table,
compliance flags, equity curve plot) is Day 6's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.logging_config import get_logger
from scheduler.market_hours import session_date
from storage.db import session_scope
from storage.tables import ClosedTrade, RiskBlock


@dataclass(frozen=True)
class EndOfDaySummary:
    """Counts that fit into a single Discord message."""

    session_date: str
    trades: int
    wins: int
    losses: int
    breakevens: int
    gross_pnl: float
    net_pnl: float
    risk_blocks: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_date": self.session_date,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "breakevens": self.breakevens,
            "gross_pnl": round(self.gross_pnl, 2),
            "net_pnl": round(self.net_pnl, 2),
            "risk_blocks": self.risk_blocks,
        }


def _session_window(now: datetime, settings) -> tuple[datetime, datetime]:
    """Return (start_utc, end_utc) for the local session date of ``now``."""
    tz = ZoneInfo(settings.TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    sd = session_date(now, settings)
    start_local = datetime.combine(sd, time(0, 0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def generate_end_of_day_summary(settings, *, now: datetime | None = None) -> EndOfDaySummary:
    """Query ``closed_trades`` + ``risk_blocks`` for the current session date."""
    log = get_logger("reports.daily_report")
    now = now or datetime.now(tz=timezone.utc)
    start_utc, end_utc = _session_window(now, settings)

    with session_scope() as session:
        closed_rows = session.execute(
            select(ClosedTrade).where(
                ClosedTrade.exit_ts >= start_utc,
                ClosedTrade.exit_ts < end_utc,
            )
        ).scalars().all()

        risk_rows = session.execute(
            select(RiskBlock).where(
                RiskBlock.ts >= start_utc,
                RiskBlock.ts < end_utc,
            )
        ).scalars().all()

        trades = len(closed_rows)
        wins = sum(1 for r in closed_rows if r.pnl > 0)
        losses = sum(1 for r in closed_rows if r.pnl < 0)
        breakevens = trades - wins - losses
        net_pnl = float(sum(r.pnl for r in closed_rows))
        gross_pnl = float(sum(r.pnl + (r.commission or 0.0) for r in closed_rows))
        risk_blocks = len(risk_rows)

    summary = EndOfDaySummary(
        session_date=session_date(now, settings).isoformat(),
        trades=trades,
        wins=wins,
        losses=losses,
        breakevens=breakevens,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        risk_blocks=risk_blocks,
    )
    log.info("eod.summary", **summary.to_payload())
    return summary
