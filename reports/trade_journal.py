"""Trade-by-trade journal export.

Two callers in mind:

- The scheduler's end-of-day job, which writes one CSV per session day
  via :func:`write_trade_journal_for_session`.
- A future CLI/operator workflow that wants any arbitrary date range
  exported via :func:`export_trade_journal`.

The CSV columns are stable so downstream tooling (a spreadsheet, a small
Streamlit dashboard, etc.) can rely on the schema:

    setup_id, paper_trade_id, instrument, direction, quantity,
    entry_ts, entry_price, exit_ts, exit_price, hold_seconds,
    exit_reason, gross_pnl, commission, slippage, net_pnl

Times are written as ISO-8601 in UTC; downstream tools convert to local
display tz themselves.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.logging_config import get_logger
from config.settings import Settings
from scheduler.market_hours import session_date
from storage.db import session_scope
from storage.tables import ClosedTrade


CSV_COLUMNS: tuple[str, ...] = (
    "setup_id",
    "paper_trade_id",
    "instrument",
    "direction",
    "quantity",
    "entry_ts",
    "entry_price",
    "exit_ts",
    "exit_price",
    "hold_seconds",
    "exit_reason",
    "gross_pnl",
    "commission",
    "slippage",
    "net_pnl",
)


@dataclass(frozen=True)
class TradeJournalRow:
    setup_id: Optional[str]
    paper_trade_id: Optional[str]
    instrument: str
    direction: str
    quantity: float
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    hold_seconds: float
    exit_reason: str
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float

    def to_csv_row(self) -> dict[str, object]:
        return {
            "setup_id": self.setup_id or "",
            "paper_trade_id": self.paper_trade_id or "",
            "instrument": self.instrument,
            "direction": self.direction,
            "quantity": f"{self.quantity:g}",
            "entry_ts": self.entry_ts.astimezone(timezone.utc).isoformat(),
            "entry_price": f"{self.entry_price:.6f}",
            "exit_ts": self.exit_ts.astimezone(timezone.utc).isoformat(),
            "exit_price": f"{self.exit_price:.6f}",
            "hold_seconds": f"{self.hold_seconds:.0f}",
            "exit_reason": self.exit_reason,
            "gross_pnl": f"{self.gross_pnl:.4f}",
            "commission": f"{self.commission:.4f}",
            "slippage": f"{self.slippage:.6f}",
            "net_pnl": f"{self.net_pnl:.4f}",
        }


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
def _row_to_journal(row: ClosedTrade) -> TradeJournalRow:
    hold = (row.exit_ts - row.entry_ts).total_seconds()
    gross = float(row.pnl) + float(row.commission or 0.0)
    return TradeJournalRow(
        setup_id=row.setup_id,
        paper_trade_id=row.paper_trade_id,
        instrument=row.instrument,
        direction=row.direction,
        quantity=float(row.quantity),
        entry_ts=row.entry_ts,
        entry_price=float(row.entry_price),
        exit_ts=row.exit_ts,
        exit_price=float(row.exit_price),
        hold_seconds=float(hold),
        exit_reason=row.exit_reason,
        gross_pnl=gross,
        commission=float(row.commission or 0.0),
        slippage=float(row.slippage or 0.0),
        net_pnl=float(row.pnl),
    )


def query_trade_journal(
    *,
    start: datetime,
    end: datetime,
    instrument: Optional[str] = None,
) -> list[TradeJournalRow]:
    """Pull closed trades whose ``exit_ts`` falls in ``[start, end)``.

    ``start`` and ``end`` MUST be tz-aware. Naive datetimes are refused so
    we don't accidentally export an off-by-N-hours window.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end < start:
        raise ValueError("end must not be before start")

    start_utc = start.astimezone(timezone.utc)
    end_utc = end.astimezone(timezone.utc)

    stmt = select(ClosedTrade).where(
        ClosedTrade.exit_ts >= start_utc, ClosedTrade.exit_ts < end_utc
    )
    if instrument is not None:
        stmt = stmt.where(ClosedTrade.instrument == instrument)
    stmt = stmt.order_by(ClosedTrade.exit_ts.asc())

    with session_scope() as session:
        rows = session.execute(stmt).scalars().all()
    return [_row_to_journal(r) for r in rows]


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def export_trade_journal(
    rows: list[TradeJournalRow],
    out_path: Path,
) -> Path:
    """Write a list of rows to ``out_path`` as CSV. Always writes a header
    so importers can rely on the schema even when there are zero trades."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_row())
    return out_path


def _session_window_utc(now: datetime, settings: Settings) -> tuple[datetime, datetime]:
    tz = ZoneInfo(settings.TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    sd = session_date(now, settings)
    start_local = datetime.combine(sd, time(0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def write_trade_journal_for_session(
    settings: Settings,
    *,
    now: datetime | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Convenience: pull the session-day's trades and write them to CSV."""
    now = now or datetime.now(tz=timezone.utc)
    start_utc, end_utc = _session_window_utc(now, settings)
    rows = query_trade_journal(
        start=start_utc, end=end_utc, instrument=settings.INSTRUMENT
    )

    out_dir = (
        Path(out_dir) if out_dir is not None else Path(settings.REPORTS_DIR) / "journals"
    )
    sd = session_date(now, settings).isoformat()
    path = out_dir / f"trade_journal_{sd}_{settings.INSTRUMENT}.csv"

    export_trade_journal(rows, path)

    log = get_logger("reports.trade_journal")
    log.info(
        "trade_journal.written",
        path=str(path),
        n_trades=len(rows),
        instrument=settings.INSTRUMENT,
        session_date=sd,
    )
    return path
