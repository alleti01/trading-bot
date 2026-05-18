"""Daily session report — Markdown + JSON.

Builds a per-session report from the ``closed_trades``, ``risk_blocks``,
and ``notifications`` tables. The layout intentionally mirrors the Day 4
backtest report so the operator's eye can compare paper-trading days to
backtest-equivalent runs.

Outputs land in ``settings.REPORTS_DIR/daily/``:

- ``daily_<YYYY-MM-DD>_<INSTRUMENT>.md``   — human-readable report.
- ``daily_<YYYY-MM-DD>_<INSTRUMENT>.json`` — machine-readable payload.

The lightweight :func:`generate_end_of_day_summary` (counts only) used by
the Day 5 scheduler still lives here so the EOD Discord alert keeps
working without paying for the full report build.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select

from app.logging_config import get_logger
from backtesting.metrics import BacktestMetrics, compute_metrics, daily_pnl_table
from backtesting.portfolio import ClosedTradeRecord
from compliance.rules import ComplianceFlag, general_compliance_flags
from compliance.tradeify_rules import TradeifyFlag, tradeify_compliance_flags
from config.instruments import get_instrument
from config.settings import Settings
from scheduler.market_hours import session_date
from storage.db import session_scope
from storage.tables import ClosedTrade, Notification, RiskBlock


# ---------------------------------------------------------------------------
# Lightweight summary (Day-5 EOD Discord alert)
# ---------------------------------------------------------------------------
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


@dataclass(frozen=True)
class DailyReportArtifacts:
    """Everything :func:`write_daily_report` produced for one session."""

    summary: EndOfDaySummary
    md_path: Path
    json_path: Path
    journal_path: Optional[Path]


# ---------------------------------------------------------------------------
# DB → dataclasses
# ---------------------------------------------------------------------------
def _session_window_utc(now: datetime, settings: Settings) -> tuple[datetime, datetime]:
    """Return the [start, end) UTC window for the local session date of ``now``."""
    tz = ZoneInfo(settings.TIMEZONE)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    sd = session_date(now, settings)
    start_local = datetime.combine(sd, time(0, 0), tzinfo=tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _row_to_record(row: ClosedTrade) -> ClosedTradeRecord:
    """Reconstruct a :class:`ClosedTradeRecord` from a DB row.

    The DB stores ``net_pnl`` and ``commission`` separately; ``gross_pnl``
    is implied as ``net + commission``. ``bars_held`` is not persisted —
    we derive it from the timestamp delta in minutes (1m bars). This is
    "good enough" for the report; the engine still has the authoritative
    bar count when it produced the trade.
    """
    bars = max(0, int((row.exit_ts - row.entry_ts).total_seconds() // 60))
    return ClosedTradeRecord(
        setup_id=row.setup_id or "",
        instrument=row.instrument,
        direction=row.direction,
        quantity=float(row.quantity),
        entry_ts=row.entry_ts,
        entry_price=float(row.entry_price),
        exit_ts=row.exit_ts,
        exit_price=float(row.exit_price),
        exit_reason=row.exit_reason,
        gross_pnl=float(row.pnl) + float(row.commission or 0.0),
        commission=float(row.commission or 0.0),
        slippage=float(row.slippage or 0.0),
        net_pnl=float(row.pnl),
        bars_held=bars,
    )


def _equity_curve(records: list[ClosedTradeRecord]) -> list[tuple[datetime, float]]:
    """Cumulative equity curve from the trade ledger (starting at 0)."""
    curve: list[tuple[datetime, float]] = []
    eq = 0.0
    for r in sorted(records, key=lambda x: x.exit_ts):
        eq += r.net_pnl
        curve.append((r.exit_ts, eq))
    return curve


def _query_session(
    settings: Settings, *, now: datetime
) -> tuple[list[ClosedTradeRecord], list[RiskBlock], dict[str, dict[str, int]]]:
    """Pull all the rows the report needs in a single read transaction."""
    start_utc, end_utc = _session_window_utc(now, settings)

    with session_scope() as session:
        closed_rows = session.execute(
            select(ClosedTrade).where(
                ClosedTrade.exit_ts >= start_utc,
                ClosedTrade.exit_ts < end_utc,
            )
        ).scalars().all()
        risk_rows = list(
            session.execute(
                select(RiskBlock).where(
                    RiskBlock.ts >= start_utc,
                    RiskBlock.ts < end_utc,
                )
            ).scalars().all()
        )
        notif_rows = session.execute(
            select(Notification).where(
                Notification.ts >= start_utc,
                Notification.ts < end_utc,
            )
        ).scalars().all()

    records = [_row_to_record(r) for r in closed_rows]

    by_kind: dict[str, dict[str, int]] = {}
    for n in notif_rows:
        bucket = by_kind.setdefault(n.kind, {"delivered": 0, "failed": 0})
        if n.delivered:
            bucket["delivered"] += 1
        else:
            bucket["failed"] += 1

    return records, risk_rows, by_kind


# ---------------------------------------------------------------------------
# Lightweight summary
# ---------------------------------------------------------------------------
def generate_end_of_day_summary(
    settings: Settings, *, now: datetime | None = None
) -> EndOfDaySummary:
    """Counts-only EOD summary. Used by the scheduler's Discord alert."""
    log = get_logger("reports.daily_report")
    now = now or datetime.now(tz=timezone.utc)
    records, risk_rows, _ = _query_session(settings, now=now)

    trades = len(records)
    wins = sum(1 for r in records if r.net_pnl > 0)
    losses = sum(1 for r in records if r.net_pnl < 0)
    breakevens = trades - wins - losses
    net_pnl = float(sum(r.net_pnl for r in records))
    gross_pnl = float(sum(r.gross_pnl for r in records))
    summary = EndOfDaySummary(
        session_date=session_date(now, settings).isoformat(),
        trades=trades,
        wins=wins,
        losses=losses,
        breakevens=breakevens,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        risk_blocks=len(risk_rows),
    )
    log.info("eod.summary", **summary.to_payload())
    return summary


# ---------------------------------------------------------------------------
# Full daily report
# ---------------------------------------------------------------------------
def _flag_to_dict(f: ComplianceFlag | TradeifyFlag) -> dict[str, Any]:
    return {
        "rule": f.rule,
        "triggered": bool(f.triggered),
        "detail": f.detail,
        "value": float(f.value),
    }


def build_daily_report_payload(
    settings: Settings, *, now: datetime
) -> dict[str, Any]:
    """Run all the queries + computations and return a JSON-friendly payload."""
    spec = get_instrument(settings.INSTRUMENT)
    records, risk_rows, notif_counts = _query_session(settings, now=now)

    metrics: BacktestMetrics = compute_metrics(
        records, _equity_curve(records), starting_equity=0.0
    )

    general = general_compliance_flags(records)
    tradeify = tradeify_compliance_flags(
        records,
        market_type=settings.MARKET_TYPE,
        timezone=settings.TIMEZONE,
        session_close=settings.trading_window_end_time(),
        flat_time=settings.force_flat_time(),
        consistency_limit_percent=settings.CONSISTENCY_LIMIT_PERCENT,
    )

    risk_summary: dict[str, int] = {}
    for rb in risk_rows:
        risk_summary[rb.rule] = risk_summary.get(rb.rule, 0) + 1

    sd = session_date(now, settings)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_date": sd.isoformat(),
        "instrument": settings.INSTRUMENT,
        "market_type": spec.market_type,
        "timezone": settings.TIMEZONE,
        "mode": settings.MODE,
        "metrics": metrics.to_dict(),
        "daily_pnl": daily_pnl_table(records),
        "trades": [_trade_row(r) for r in records],
        "risk_blocks": [
            {
                "ts": rb.ts.isoformat() if rb.ts else None,
                "rule": rb.rule,
                "reason": rb.reason,
                "setup_id": rb.setup_id,
            }
            for rb in risk_rows
        ],
        "risk_blocks_summary": risk_summary,
        "notifications_summary": notif_counts,
        "compliance": {
            "general": [_flag_to_dict(f) for f in general],
            "tradeify": [_flag_to_dict(f) for f in tradeify],
        },
        "config": {
            "max_trades_per_day": settings.MAX_TRADES_PER_DAY,
            "max_daily_loss": settings.MAX_DAILY_LOSS,
            "max_daily_profit": settings.MAX_DAILY_PROFIT,
            "max_position_size": settings.MAX_POSITION_SIZE,
            "risk_per_trade": settings.RISK_PER_TRADE,
            "slippage_ticks": settings.SLIPPAGE_TICKS,
            "commission_per_contract": settings.COMMISSION_PER_CONTRACT,
            "force_flat_time": str(settings.force_flat_time()),
            "trading_window_start": str(settings.trading_window_start_time()),
            "trading_window_end": str(settings.trading_window_end_time()),
            "max_hold_bars": settings.MAX_HOLD_BARS,
            "consistency_limit_percent": settings.CONSISTENCY_LIMIT_PERCENT,
        },
    }


def _trade_row(r: ClosedTradeRecord) -> dict[str, Any]:
    return {
        "setup_id": r.setup_id,
        "instrument": r.instrument,
        "direction": r.direction,
        "quantity": float(r.quantity),
        "entry_ts": r.entry_ts.isoformat(),
        "entry_price": float(r.entry_price),
        "exit_ts": r.exit_ts.isoformat(),
        "exit_price": float(r.exit_price),
        "exit_reason": r.exit_reason,
        "gross_pnl": float(r.gross_pnl),
        "commission": float(r.commission),
        "slippage": float(r.slippage),
        "net_pnl": float(r.net_pnl),
        "bars_held": int(r.bars_held),
    }


def render_daily_markdown(payload: dict[str, Any]) -> str:
    m = payload.get("metrics") or {}
    risk_summary = payload.get("risk_blocks_summary") or {}
    notif = payload.get("notifications_summary") or {}
    general = payload["compliance"]["general"]
    tradeify = payload["compliance"]["tradeify"]
    trades = payload.get("trades") or []

    lines: list[str] = []
    lines.append(
        f"# Daily report — {payload['session_date']} — {payload['instrument']}"
    )
    lines.append("")
    lines.append(
        f"_Generated: {payload['generated_at']} UTC · mode "
        f"`{payload['mode']}` · tz `{payload['timezone']}`_"
    )
    lines.append("")

    # ----- Summary -----
    lines.append("## Summary")
    lines.append("")
    if not trades:
        lines.append("- No trades placed today.")
    else:
        lines.extend(
            [
                f"- Trades: **{m['n_trades']}** "
                f"(wins {m['n_wins']} / losses {m['n_losses']} / be {m['n_breakevens']})",
                f"- Win rate: **{m['win_rate']:.2%}**",
                f"- Net PnL: **${m['net_pnl']:.2f}** "
                f"(gross ${m['gross_pnl']:.2f}, commissions ${m['total_commission']:.2f})",
                f"- Expectancy / trade: **${m['expectancy_per_trade']:.2f}**",
                f"- Profit factor: **{m['profit_factor']:.2f}**",
                f"- Max drawdown: **${m['max_drawdown_dollars']:.2f}** "
                f"({m['max_drawdown_pct']:.2%})",
                f"- Sharpe (per-trade): **{m['sharpe_per_trade']:.3f}**",
                f"- Avg bars held: **{m['avg_bars_held']:.2f}**",
                f"- Equity: ${m['starting_equity']:.2f} → ${m['ending_equity']:.2f}",
            ]
        )
    lines.append("")

    # ----- Compliance -----
    lines.append("## Compliance")
    lines.append("")
    if not (general or tradeify):
        lines.append("- No compliance checks ran (no trades).")
    else:
        for f in general + tradeify:
            marker = "FAIL" if f["triggered"] else "OK"
            lines.append(f"- [{marker}] `{f['rule']}` — {f['detail']}")
    lines.append("")

    # ----- Risk blocks -----
    lines.append("## Risk blocks")
    lines.append("")
    if not risk_summary:
        lines.append("- No risk blocks today.")
    else:
        for rule, n in sorted(risk_summary.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {rule}: {n}")
    lines.append("")

    # ----- Trade journal table -----
    lines.append("## Trades")
    lines.append("")
    if not trades:
        lines.append("- (none)")
    else:
        lines.append(
            "| # | Entry (UTC) | Exit (UTC) | Dir | Qty | "
            "Entry $ | Exit $ | Reason | Net $ | Bars |"
        )
        lines.append(
            "|--:|-------------|------------|-----|----:|"
            "-------:|-------:|--------|------:|-----:|"
        )
        for i, t in enumerate(trades, start=1):
            lines.append(
                f"| {i} | {t['entry_ts']} | {t['exit_ts']} | {t['direction']} "
                f"| {t['quantity']:.0f} | {t['entry_price']:.2f} | {t['exit_price']:.2f} "
                f"| {t['exit_reason']} | {t['net_pnl']:.2f} | {t['bars_held']} |"
            )
    lines.append("")

    # ----- Notifications summary -----
    lines.append("## Notifications")
    lines.append("")
    if not notif:
        lines.append("- (no notifications recorded)")
    else:
        for kind, counts in sorted(notif.items()):
            lines.append(
                f"- `{kind}` — delivered {counts.get('delivered', 0)}, "
                f"failed {counts.get('failed', 0)}"
            )
    lines.append("")

    # ----- Config snapshot -----
    cfg = payload.get("config") or {}
    if cfg:
        lines.append("## Config snapshot")
        lines.append("")
        for k in (
            "max_trades_per_day",
            "max_daily_loss",
            "max_daily_profit",
            "risk_per_trade",
            "slippage_ticks",
            "commission_per_contract",
            "force_flat_time",
            "trading_window_start",
            "trading_window_end",
            "max_hold_bars",
            "consistency_limit_percent",
        ):
            if k in cfg:
                lines.append(f"- `{k}`: {cfg[k]}")
        lines.append("")

    return "\n".join(lines)


def write_daily_report(
    settings: Settings,
    *,
    now: datetime | None = None,
    out_dir: Path | None = None,
    include_journal: bool = True,
) -> DailyReportArtifacts:
    """Render and persist the full daily report.

    Returns a :class:`DailyReportArtifacts` describing where the files
    landed. ``include_journal=True`` also writes the per-trade CSV via
    :mod:`reports.trade_journal`.
    """
    log = get_logger("reports.daily_report")
    now = now or datetime.now(tz=timezone.utc)

    out_dir = Path(out_dir) if out_dir is not None else Path(settings.REPORTS_DIR) / "daily"
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_daily_report_payload(settings, now=now)
    sd = payload["session_date"]
    instr = payload["instrument"]
    json_path = out_dir / f"daily_{sd}_{instr}.json"
    md_path = out_dir / f"daily_{sd}_{instr}.md"

    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path.write_text(render_daily_markdown(payload))

    summary = EndOfDaySummary(
        session_date=sd,
        trades=payload["metrics"]["n_trades"],
        wins=payload["metrics"]["n_wins"],
        losses=payload["metrics"]["n_losses"],
        breakevens=payload["metrics"]["n_breakevens"],
        gross_pnl=payload["metrics"]["gross_pnl"],
        net_pnl=payload["metrics"]["net_pnl"],
        risk_blocks=sum(payload["risk_blocks_summary"].values()),
    )

    journal_path: Optional[Path] = None
    if include_journal:
        # Local import keeps a circular-free import graph (trade_journal
        # imports nothing from this module).
        from reports.trade_journal import write_trade_journal_for_session

        journal_path = write_trade_journal_for_session(settings, now=now)

    log.info(
        "daily_report.written",
        json_path=str(json_path),
        md_path=str(md_path),
        journal_path=str(journal_path) if journal_path else None,
        n_trades=summary.trades,
        net_pnl=round(summary.net_pnl, 2),
    )
    return DailyReportArtifacts(
        summary=summary,
        md_path=md_path,
        json_path=json_path,
        journal_path=journal_path,
    )
