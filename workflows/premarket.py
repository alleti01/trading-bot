"""Pre-market workflow — research log + optional MacroNewsAgent."""

from __future__ import annotations

from typing import Any, Optional

from agents.schemas import MacroNewsAssessment
from config.instruments import get_instrument
from workflows.base import BaseWorkflow, WorkflowContext
from workflows.memory import append_section, read_tail, read_text
from workflows.schemas import PlannedTradeIdea, ResearchLogEntry


class PremarketWorkflow(BaseWorkflow):
    name = "premarket"

    def _execute(self, ctx: WorkflowContext) -> dict[str, Any]:
        strategy_tail = read_tail(ctx.memory.strategy, max_lines=40)
        trade_tail = read_tail(ctx.memory.trade_log, max_lines=30)
        research_tail = read_tail(ctx.memory.research_log, max_lines=20)
        broker_state = ctx.broker.pull_state(now=ctx.now)
        macro: Optional[MacroNewsAssessment] = None
        urgent = False

        if ctx.gates.macro_research_enabled() and ctx.orchestrator is not None:
            raw = ctx.orchestrator.run_macro_news(
                now=ctx.now,
                enabled_symbols=list(ctx.settings.ENABLED_SYMBOLS),
            )
            if raw is not None:
                macro = raw
                urgent = (
                    macro.risk_level == "high" or bool(macro.blocked_windows)
                )

        ideas = _build_planned_ideas(
            ctx,
            symbols=list(ctx.settings.ENABLED_SYMBOLS),
            macro=macro,
        )
        entry = ResearchLogEntry(
            session_date=ctx.session_date,
            account=broker_state.account,
            market_context=_market_context(macro, strategy_tail),
            planned_trades=ideas,
            macro_summary=macro.summary if macro else None,
            macro_risk_level=macro.risk_level if macro else None,
            default_decision="hold",
            notes=(
                f"Trade log tail ({len(trade_tail)} chars); "
                f"research tail ({len(research_tail)} chars)."
            ),
        )
        section = _format_research_section(entry)
        append_section(ctx.memory.research_log, section)

        discord_sent = False
        if urgent:
            discord_sent = self._notify_safe(
                ctx,
                "high_risk_news",
                session_date=ctx.session_date,
                summary=macro.summary if macro else "",
                risk_level=macro.risk_level if macro else "high",
                source="workflow.premarket",
            )

        return {
            "research_written": True,
            "planned_trades": len(ideas),
            "macro_ran": macro is not None,
            "urgent": urgent,
            "discord_sent": discord_sent,
            "account": broker_state.account.model_dump(),
        }


def _market_context(
    macro: Optional[MacroNewsAssessment], strategy_tail: str
) -> str:
    parts: list[str] = []
    if macro is not None:
        parts.append(macro.summary)
        if macro.key_events:
            parts.append("Key events: " + "; ".join(macro.key_events[:5]))
    if strategy_tail:
        parts.append("Strategy excerpt:\n" + strategy_tail[-500:])
    return "\n\n".join(parts) if parts else "No macro research; strategy doc only."


def _build_planned_ideas(
    ctx: WorkflowContext,
    *,
    symbols: list[str],
    macro: Optional[MacroNewsAssessment],
) -> list[PlannedTradeIdea]:
    ideas: list[PlannedTradeIdea] = []
    affected = {s.upper() for s in (macro.affected_symbols if macro else [])}
    for sym in symbols[:3]:
        spec = get_instrument(sym)
        risk_notes: list[str] = []
        if macro and macro.risk_level in {"medium", "high"}:
            risk_notes.append(f"Macro risk: {macro.risk_level}")
        if sym.upper() in affected:
            risk_notes.append("Symbol flagged in macro scan")
        ideas.append(
            PlannedTradeIdea(
                symbol=sym,
                side="flat",
                catalyst="Session playbook — confirm at market open",
                entry_zone=f"Near prior session value / VWAP ({spec.symbol})",
                stop=f"{int(ctx.settings.WEBHOOK_DEFAULT_STOP_TICKS)} ticks adverse",
                target=f"{int(ctx.settings.WEBHOOK_DEFAULT_TARGET_TICKS)} ticks favorable",
                risk_factors=risk_notes or ["Standard session risk"],
                default_decision="hold",
                thesis=read_text(ctx.memory.strategy)[:400],
            )
        )
    return ideas


def _format_research_section(entry: ResearchLogEntry) -> str:
    lines = [
        f"## {entry.session_date} Pre-market",
        "",
        "### Account snapshot",
        f"- Day PnL: ${entry.account.day_pnl:.2f}",
        f"- Cumulative PnL: ${entry.account.cumulative_pnl:.2f}",
        f"- Trades today: {entry.account.trades_today}",
        f"- Open positions: {entry.account.open_positions}",
        "",
        "### Market context",
        entry.market_context,
        "",
        "### Planned trade ideas (default: hold)",
    ]
    for i, idea in enumerate(entry.planned_trades, start=1):
        lines.extend(
            [
                f"#### Idea {i}: {idea.symbol}",
                f"- Side: {idea.side} (default decision: {idea.default_decision})",
                f"- Catalyst: {idea.catalyst}",
                f"- Entry zone: {idea.entry_zone}",
                f"- Stop: {idea.stop}",
                f"- Target: {idea.target}",
                f"- Risk factors: {', '.join(idea.risk_factors) or 'n/a'}",
            ]
        )
    if entry.macro_risk_level:
        lines.extend(["", f"### Macro risk level: {entry.macro_risk_level}"])
    lines.extend(["", f"**Default decision:** {entry.default_decision}"])
    return "\n".join(lines)
