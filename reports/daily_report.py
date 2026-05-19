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
from storage.tables import (
    ClosedTrade,
    ImprovementSuggestion,
    Notification,
    RiskBlock,
    TradeAnalysis,
    TradeMistakeTag,
)


# ---------------------------------------------------------------------------
# Lightweight summary (Day-5 EOD Discord alert)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EndOfDaySymbolStats:
    """Per-symbol slice of the daily summary.

    Used by the EOD Discord alert + the full daily report so operators
    running 4–6 symbols can see at a glance which one paid and which
    one bled.
    """

    symbol: str
    trades: int
    wins: int
    losses: int
    net_pnl: float
    profit_factor: float
    expectancy: float
    false_positives: int
    risk_blocks: int

    @property
    def win_rate(self) -> float:
        return float(self.wins) / self.trades if self.trades else 0.0

    def to_payload(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "trades": self.trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 4),
            "net_pnl": round(self.net_pnl, 2),
            "profit_factor": round(self.profit_factor, 4),
            "expectancy": round(self.expectancy, 4),
            "false_positives": self.false_positives,
            "risk_blocks": self.risk_blocks,
        }


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
    by_symbol: tuple[EndOfDaySymbolStats, ...] = ()
    best_symbol: Optional[str] = None
    worst_symbol: Optional[str] = None

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
            "by_symbol": [s.to_payload() for s in self.by_symbol],
            "best_symbol": self.best_symbol,
            "worst_symbol": self.worst_symbol,
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


def _query_analysis_section(
    settings: Settings, *, now: datetime
) -> dict[str, Any]:
    """Day 8: pull mistake-tag counts, false positives, best/worst trades,
    and proposed improvements for today's session."""
    start_utc, end_utc = _session_window_utc(now, settings)

    with session_scope() as session:
        analyses = list(
            session.execute(
                select(TradeAnalysis).where(
                    TradeAnalysis.entry_ts >= start_utc,
                    TradeAnalysis.entry_ts < end_utc,
                )
            ).scalars().all()
        )
        if not analyses:
            return {
                "n_analyzed": 0,
                "top_mistake_tags": [],
                "false_positives": 0,
                "best_trade": None,
                "worst_trade": None,
                "proposed_improvements": [],
            }
        analysis_ids = [a.id for a in analyses]
        tag_rows = list(
            session.execute(
                select(TradeMistakeTag).where(
                    TradeMistakeTag.trade_analysis_id.in_(analysis_ids)
                )
            ).scalars().all()
        )
        proposals = list(
            session.execute(
                select(ImprovementSuggestion)
                .where(
                    ImprovementSuggestion.created_at >= start_utc,
                    ImprovementSuggestion.created_at < end_utc,
                    ImprovementSuggestion.validation_status == "proposed",
                )
                .order_by(ImprovementSuggestion.created_at.desc())
                .limit(20)
            ).scalars().all()
        )

    tag_counts: dict[str, int] = {}
    fp_set: set[str] = set()
    for t in tag_rows:
        tag_counts[t.tag] = tag_counts.get(t.tag, 0) + 1
        if t.tag == "false_positive":
            fp_set.add(t.closed_trade_id)

    best = max(analyses, key=lambda a: float(a.net_pnl))
    worst = min(analyses, key=lambda a: float(a.net_pnl))

    def _slim(a: TradeAnalysis) -> dict[str, Any]:
        return {
            "trade_id": a.closed_trade_id,
            "instrument": a.instrument,
            "direction": a.direction,
            "strategy": a.strategy_name,
            "result": a.result,
            "net_pnl": float(a.net_pnl),
            "exit_reason": a.exit_reason,
            "r_multiple": float(a.r_multiple or 0.0),
        }

    return {
        "n_analyzed": len(analyses),
        "top_mistake_tags": sorted(tag_counts.items(), key=lambda kv: -kv[1])[:10],
        "false_positives": len(fp_set),
        "best_trade": _slim(best),
        "worst_trade": _slim(worst),
        "proposed_improvements": [
            {
                "suggestion_id": p.suggestion_id,
                "reason": p.reason,
                "affected_strategy": p.affected_strategy,
                "affected_condition": p.affected_condition,
                "risk_of_overfitting": p.risk_of_overfitting,
                "validation_status": p.validation_status,
            }
            for p in proposals
        ],
    }


# ---------------------------------------------------------------------------
# Lightweight summary
# ---------------------------------------------------------------------------
def _per_symbol_stats(
    records: list[ClosedTradeRecord],
    risk_rows: list[RiskBlock],
    fp_trade_ids: set[str],
) -> list[EndOfDaySymbolStats]:
    """Slice the trade ledger by symbol and compute the headline stats.

    ``fp_trade_ids`` is the set of closed_trade ids tagged
    ``false_positive`` by the mistake classifier (already extracted by
    the analysis section query). We only count those trades here so
    the per-symbol false-positive number lines up with the analysis
    block.
    """
    by_symbol: dict[str, list[ClosedTradeRecord]] = {}
    for r in records:
        by_symbol.setdefault(r.instrument, []).append(r)
    risk_by_symbol: dict[str, int] = {}
    for rb in risk_rows:
        # Risk blocks live in the DB without a denormalized symbol; we
        # heuristically attribute via the setup_id when it was minted by
        # a known symbol's loop. For the MVP we count totals at the
        # whole-session level too — caller already sums them.
        sym = getattr(rb, "instrument", None)
        if not sym:
            continue
        risk_by_symbol[sym] = risk_by_symbol.get(sym, 0) + 1

    stats: list[EndOfDaySymbolStats] = []
    for sym, trades in by_symbol.items():
        wins = sum(1 for t in trades if t.net_pnl > 0)
        losses = sum(1 for t in trades if t.net_pnl < 0)
        gains = float(sum(t.net_pnl for t in trades if t.net_pnl > 0))
        loss_sum = float(-sum(t.net_pnl for t in trades if t.net_pnl < 0))
        pf = (gains / loss_sum) if loss_sum > 0 else (float("inf") if gains > 0 else 0.0)
        expectancy = (
            float(sum(t.net_pnl for t in trades) / len(trades)) if trades else 0.0
        )
        fp = sum(1 for t in trades if t.setup_id in fp_trade_ids)
        stats.append(
            EndOfDaySymbolStats(
                symbol=sym,
                trades=len(trades),
                wins=wins,
                losses=losses,
                net_pnl=float(sum(t.net_pnl for t in trades)),
                profit_factor=float(pf) if pf != float("inf") else 0.0,
                expectancy=expectancy,
                false_positives=fp,
                risk_blocks=int(risk_by_symbol.get(sym, 0)),
            )
        )
    # Sort by net PnL descending so reports show the winners first.
    stats.sort(key=lambda s: -s.net_pnl)
    return stats


def _false_positive_trade_ids(settings: Settings, *, now: datetime) -> set[str]:
    start_utc, end_utc = _session_window_utc(now, settings)
    with session_scope() as session:
        rows = session.execute(
            select(TradeMistakeTag).where(
                TradeMistakeTag.tag == "false_positive",
            )
        ).scalars().all()
        # Intersect with this session's analyses to keep things bounded.
        analyses = list(
            session.execute(
                select(TradeAnalysis.closed_trade_id).where(
                    TradeAnalysis.entry_ts >= start_utc,
                    TradeAnalysis.entry_ts < end_utc,
                )
            ).scalars().all()
        )
    valid = set(analyses)
    return {r.closed_trade_id for r in rows if r.closed_trade_id in valid}


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

    fp_ids = _false_positive_trade_ids(settings, now=now)
    by_symbol = _per_symbol_stats(records, risk_rows, fp_ids)
    best = by_symbol[0].symbol if by_symbol else None
    worst = by_symbol[-1].symbol if by_symbol else None

    summary = EndOfDaySummary(
        session_date=session_date(now, settings).isoformat(),
        trades=trades,
        wins=wins,
        losses=losses,
        breakevens=breakevens,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        risk_blocks=len(risk_rows),
        by_symbol=tuple(by_symbol),
        best_symbol=best,
        worst_symbol=worst,
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

    analysis_section = _query_analysis_section(settings, now=now)
    fp_ids = _false_positive_trade_ids(settings, now=now)
    by_symbol = _per_symbol_stats(records, risk_rows, fp_ids)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session_date": sd.isoformat(),
        "instrument": settings.INSTRUMENT,
        "enabled_symbols": list(getattr(settings, "ENABLED_SYMBOLS", [])) or [
            settings.INSTRUMENT
        ],
        "market_type": spec.market_type,
        "timezone": settings.TIMEZONE,
        "mode": settings.MODE,
        "metrics": metrics.to_dict(),
        "daily_pnl": daily_pnl_table(records),
        "trades": [_trade_row(r) for r in records],
        "analysis": analysis_section,
        "by_symbol": [s.to_payload() for s in by_symbol],
        "best_symbol": by_symbol[0].symbol if by_symbol else None,
        "worst_symbol": by_symbol[-1].symbol if by_symbol else None,
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

    # ----- Per-symbol breakdown -----
    by_symbol = payload.get("by_symbol") or []
    if by_symbol:
        lines.append("## Performance by symbol")
        lines.append("")
        lines.append(
            "| Symbol | Trades | Wins | Win rate | Net PnL | Profit factor |"
            " Expectancy | False positives | Risk blocks |"
        )
        lines.append(
            "|--------|-------:|-----:|---------:|--------:|--------------:|"
            "-----------:|----------------:|------------:|"
        )
        for s in by_symbol:
            lines.append(
                f"| {s['symbol']} | {s['trades']} | {s['wins']} | "
                f"{s['win_rate']:.2%} | ${s['net_pnl']:.2f} | "
                f"{s['profit_factor']:.2f} | ${s['expectancy']:.2f} | "
                f"{s['false_positives']} | {s['risk_blocks']} |"
            )
        if payload.get("best_symbol") and payload.get("worst_symbol"):
            lines.append("")
            lines.append(
                f"- Best symbol: **{payload['best_symbol']}** · "
                f"Worst symbol: **{payload['worst_symbol']}**"
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

    # ----- Day 8: trade analysis & proposed improvements -----
    a = payload.get("analysis") or {}
    if a.get("n_analyzed", 0) > 0:
        lines.append("## Trade analysis")
        lines.append("")
        lines.append(
            f"- Analyzed: **{a['n_analyzed']}** "
            f"(false positives: {a.get('false_positives', 0)})"
        )
        if a.get("top_mistake_tags"):
            lines.append("- Top mistake tags:")
            for tag, n in a["top_mistake_tags"]:
                lines.append(f"  - `{tag}` x {n}")
        best = a.get("best_trade")
        worst = a.get("worst_trade")
        if best:
            lines.append(
                f"- Best trade: `{best['trade_id']}` "
                f"({best['direction']} {best['strategy'] or '?'} → {best['result']}, "
                f"${best['net_pnl']:.2f}, R={best['r_multiple']:.2f})"
            )
        if worst:
            lines.append(
                f"- Worst trade: `{worst['trade_id']}` "
                f"({worst['direction']} {worst['strategy'] or '?'} → {worst['result']}, "
                f"${worst['net_pnl']:.2f}, R={worst['r_multiple']:.2f})"
            )
        lines.append("")
    proposals = (a.get("proposed_improvements") or []) if a else []
    if proposals:
        lines.append("## Proposed improvements (need backtesting)")
        lines.append("")
        lines.append(
            "These are logged with `validation_status='proposed'`. They are "
            "**never** auto-applied; promote only via `--retrain-from-feedback` "
            "+ `--promote-model`."
        )
        lines.append("")
        for p in proposals:
            lines.append(
                f"- `{p['suggestion_id']}` "
                f"(overfit risk: {p.get('risk_of_overfitting', '?')}) — "
                f"{p['reason']}"
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
