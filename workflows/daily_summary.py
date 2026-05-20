"""Daily summary workflow — EOD snapshot + Discord."""

from __future__ import annotations

from typing import Any

from reports.daily_report import build_daily_report_payload, generate_end_of_day_summary
from workflows.base import BaseWorkflow, WorkflowContext
from workflows.memory import append_section


class DailySummaryWorkflow(BaseWorkflow):
    name = "daily-summary"

    def _respects_weekdays_only(self) -> bool:
        return False

    def _execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        summary = generate_end_of_day_summary(ctx.settings, now=ctx.now)
        payload = build_daily_report_payload(ctx.settings, now=ctx.now)
        broker_state = ctx.broker.pull_state(now=ctx.now)

        section = _format_trade_log_section(
            session_date=ctx.session_date,
            summary=summary,
            broker=broker_state.account,
            payload=payload,
        )
        append_section(ctx.memory.trade_log, section)

        lines = _discord_lines(summary, ctx.session_date)
        discord_sent = self._notify_safe(
            ctx,
            "eod.summary",
            session_date=ctx.session_date,
            lines=lines,
            headline=lines[0],
            source="workflow.daily_summary",
        )

        win_rate = summary.wins / summary.trades if summary.trades else 0.0
        return {
            "trades": summary.trades,
            "net_pnl": summary.net_pnl,
            "win_rate": win_rate,
            "memory_written": True,
            "discord_sent": discord_sent,
            "discord_lines": len(lines),
        }


def _format_trade_log_section(
    *,
    session_date: str,
    summary,
    broker,
    payload: dict[str, Any],
) -> str:
    metrics = payload.get("metrics", {})
    return "\n".join(
        [
            f"## {session_date} EOD",
            "",
            f"- Net PnL: ${summary.net_pnl:.2f}",
            f"- Trades: {summary.trades} (W {summary.wins} / L {summary.losses})",
            f"- Win rate: {(summary.wins / summary.trades if summary.trades else 0):.1%}",
            f"- Cumulative PnL (broker est.): ${broker.cumulative_pnl:.2f}",
            f"- Profit factor: {metrics.get('profit_factor', 'n/a')}",
            f"- Expectancy: {metrics.get('expectancy', 'n/a')}",
            "",
        ]
    )


def _discord_lines(summary, session_date: str) -> list[str]:  # noqa: ANN001
    win_rate = summary.wins / summary.trades if summary.trades else 0.0
    lines = [
        f"EOD {session_date}: PnL ${summary.net_pnl:.2f}",
        f"Trades {summary.trades} | W {summary.wins} L {summary.losses}",
        f"Win rate {win_rate:.0%}",
    ]
    if summary.by_symbol:
        best = max(summary.by_symbol, key=lambda s: s.net_pnl)
        worst = min(summary.by_symbol, key=lambda s: s.net_pnl)
        lines.append(f"Best {best.symbol} ${best.net_pnl:.2f}")
        lines.append(f"Worst {worst.symbol} ${worst.net_pnl:.2f}")
    lines.append("Workflow: daily-summary (advisory)")
    return lines[:14]
