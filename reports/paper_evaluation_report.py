"""Write per-provider and combined parallel paper evaluation reports."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from evaluation.evaluation_context import EvaluationContext

_log = get_logger("reports.paper_evaluation")


def write_track_report(ctx: EvaluationContext, *, payload: dict[str, Any]) -> Path:
    """Write a per-provider evaluation report to the track's report_path."""
    ctx.report_path.mkdir(parents=True, exist_ok=True)
    session = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    path = ctx.report_path / f"report_{session}.md"
    lines = [
        f"# {ctx.broker_provider} evaluation: {ctx.evaluation_id}",
        "",
        f"**Session:** {session}",
        f"**Provider:** {ctx.broker_provider}",
        f"**Symbols:** {', '.join(ctx.enabled_symbols)}",
        f"**Started:** {ctx.started_at.isoformat()}",
        f"**Trades this session:** {len(ctx.trades)}",
        f"**Blocked:** {ctx.blocked}",
        "",
        "## Workflow results",
        "",
    ]
    workflows = payload.get("workflows", [])
    for wf in workflows:
        name = wf.get("workflow", "?")
        ok = wf.get("success", False)
        lines.append(f"- {name}: {'✓' if ok else '✗'}")
    lines.append("")
    reconcile = payload.get("reconcile", {})
    if reconcile:
        lines.append("## Reconcile snapshot")
        lines.append(f"- Open positions: {reconcile.get('open_positions', 0)}")
        lines.append(f"- Open orders: {reconcile.get('open_orders', 0)}")
        lines.append("")
    if ctx.errors:
        lines.append("## Errors")
        for err in ctx.errors[-10:]:
            lines.append(f"- {err}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    _log.info(
        "report.track_written",
        path=str(path),
        **ctx.discord_tags(),
    )
    return path


def write_combined_summary(
    contexts: dict[str, EvaluationContext],
    *,
    results: dict[str, Any],
    evaluations_dir: Optional[Path] = None,
) -> Path:
    """Write a combined parallel-paper summary comparing tracks at a high level."""
    evaluations_dir = evaluations_dir or Path("data/evaluations")
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    path = evaluations_dir / "parallel_summary.md"

    lines = [
        "# Parallel Paper Evaluation Summary",
        "",
        "> **Important:** These two tracks test fundamentally different things.",
        "> Alpaca tests broker/API plumbing (equities sandbox).",
        "> futures_sim tests futures strategy/model behavior (local simulation).",
        "> Do not compare PnL directly across them.",
        "",
        "## Tracks",
        "",
    ]
    for provider, ctx in sorted(contexts.items()):
        track_result = results.get(provider, {})
        ok = track_result.get("success", False)
        trades = track_result.get("trades", 0)
        lines.extend(
            [
                f"### {provider} ({ctx.evaluation_id})",
                f"- Symbols: {', '.join(ctx.enabled_symbols)}",
                f"- Success: {'✓' if ok else '✗'}",
                f"- Trades: {trades}",
                f"- Blocked: {ctx.blocked}",
                f"- State: `{ctx.state_path}`",
                f"- Reports: `{ctx.report_path}/`",
                "",
            ]
        )
    lines.extend(
        [
            "---",
            "",
            "Alpaca tests broker/API plumbing.",
            "futures_sim tests futures strategy/model behavior.",
            "Do not compare PnL directly across them.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    _log.info("report.combined_summary_written", path=str(path))
    return path
