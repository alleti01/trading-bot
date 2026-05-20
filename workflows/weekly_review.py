"""Weekly review workflow — Friday EOD stats + memory + Discord."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import select

from backtesting.metrics import compute_metrics
from backtesting.portfolio import ClosedTradeRecord
from storage.db import session_scope
from storage.tables import ClosedTrade as ClosedTradeRow
from workflows.base import BaseWorkflow, WorkflowContext
from workflows.memory import append_section, is_friday


class WeeklyReviewWorkflow(BaseWorkflow):
    name = "weekly-review"

    def _execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        if not ctx.force and not is_friday(ctx.session_date):
            return {
                "skipped": True,
                "reason": "Weekly review runs Friday after close unless forced.",
            }

        week_start = ctx.now - timedelta(days=7)
        records: list[ClosedTradeRecord] = []
        with session_scope() as session:
            rows = session.execute(select(ClosedTradeRow)).scalars().all()
            for row in rows:
                if row.exit_ts >= week_start:
                    records.append(
                        ClosedTradeRecord(
                            setup_id=str(row.setup_id or ""),
                            instrument=row.instrument,
                            direction=row.direction,
                            quantity=float(row.quantity),
                            entry_ts=row.entry_ts,
                            entry_price=float(row.entry_price),
                            exit_ts=row.exit_ts,
                            exit_price=float(row.exit_price),
                            exit_reason=row.exit_reason,
                            gross_pnl=float(row.pnl),
                            commission=float(row.commission),
                            slippage=float(row.slippage),
                            net_pnl=float(row.pnl),
                            bars_held=0,
                        )
                    )

        metrics = compute_metrics(records) if records else None
        wins = sum(1 for r in records if r.net_pnl > 0)
        losses = sum(1 for r in records if r.net_pnl <= 0)
        net = sum(r.net_pnl for r in records)
        best = max(records, key=lambda r: r.net_pnl, default=None)
        worst = min(records, key=lambda r: r.net_pnl, default=None)

        section = _format_weekly_section(
            session_date=ctx.session_date,
            net_pnl=net,
            wins=wins,
            losses=losses,
            open_count=ctx.broker.pull_state(now=ctx.now).account.open_positions,
            metrics=metrics,
            best=best,
            worst=worst,
        )
        append_section(ctx.memory.weekly_review, section)

        pf = metrics.profit_factor if metrics is not None else 0.0
        headline = (
            f"Weekly {ctx.session_date}: ${net:.2f} | "
            f"{wins}W/{losses}L | PF {pf:.2f}"
        )
        discord_sent = self._notify_safe(
            ctx,
            "agent.summary",
            headline=headline[:200],
            source="workflow.weekly_review",
        )

        return {
            "weekly_return": net,
            "wins": wins,
            "losses": losses,
            "open_positions": ctx.broker.pull_state(now=ctx.now).account.open_positions,
            "profit_factor": metrics.profit_factor if metrics else None,
            "discord_sent": discord_sent,
        }


def _format_weekly_section(
    *,
    session_date: str,
    net_pnl: float,
    wins: int,
    losses: int,
    open_count: int,
    metrics,
    best,
    worst,
) -> str:
    lines = [
        f"## Week ending {session_date}",
        "",
        f"- Weekly return (net): ${net_pnl:.2f}",
        f"- W/L: {wins}/{losses} | Open positions: {open_count}",
    ]
    if metrics is not None:
        lines.append(f"- Win rate: {metrics.win_rate:.1%}")
        lines.append(f"- Profit factor: {metrics.profit_factor:.2f}")
        lines.append(f"- Expectancy: ${metrics.expectancy_per_trade:.2f}")
    if best is not None:
        lines.append(
            f"- Best trade: {best.instrument} ${best.net_pnl:.2f} ({best.exit_reason})"
        )
    if worst is not None:
        lines.append(
            f"- Worst trade: {worst.instrument} ${worst.net_pnl:.2f} ({worst.exit_reason})"
        )
    return "\n".join(lines)
