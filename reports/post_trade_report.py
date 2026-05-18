"""Per-trade Markdown post-mortem.

The :class:`PostTradeAnalysisService` writes one of these files for
every closed trade. The format is intentionally short — operators can
flip through hundreds of these in a session — but it carries every
field the downstream review tooling and the LLM agent need.

The renderer is independent from the analyzer / classifier so tests can
build a fixture :class:`PostTradeAnalysis` and assert on the rendered
Markdown without touching the DB.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from analysis.mistake_classifier import MistakeTagging
from analysis.types import MistakeTag, PostTradeAnalysis
from app.logging_config import get_logger
from config.settings import Settings


@dataclass(frozen=True)
class PostTradeReportArtifacts:
    md_path: Path


def render_post_trade_markdown(
    analysis: PostTradeAnalysis,
    tagging: MistakeTagging,
    *,
    agent_summary: Optional[dict] = None,
) -> str:
    """Return the Markdown body for a single trade.

    ``agent_summary`` is an optional dict matching
    :class:`agents.schemas.TradeAnalysisSummary` — when present, an
    ``Agent commentary`` section is rendered. If absent, the report is
    purely deterministic.
    """
    lines: list[str] = []
    lines.append(
        f"# Trade {analysis.trade_id} — {analysis.instrument} ({analysis.direction})"
    )
    lines.append("")
    lines.append(
        f"_Closed at {analysis.exit_ts.isoformat()} · "
        f"strategy `{analysis.strategy or 'unknown'}` · "
        f"result **{analysis.result}** · net PnL **${analysis.net_pnl:.2f}**_"
    )
    lines.append("")

    # ----- Key facts -----
    lines.append("## Key facts")
    lines.append("")
    lines.append(
        f"- Entry: ${analysis.entry_price:.4f} at {analysis.entry_ts.isoformat()}"
    )
    lines.append(
        f"- Exit:  ${analysis.exit_price:.4f} ({analysis.exit_reason})"
    )
    if analysis.stop_price is not None and analysis.target_price is not None:
        lines.append(
            f"- Plan:  stop ${analysis.stop_price:.4f} / target ${analysis.target_price:.4f}"
        )
    lines.append(
        f"- Hold:  {int(analysis.hold_seconds)}s · R = {analysis.r_multiple:+.2f}"
    )
    if analysis.model_confidence is not None:
        lines.append(
            f"- Model: p={analysis.model_confidence:.3f} (thr "
            f"{analysis.model_threshold or 0.0:.2f})"
        )
    if analysis.mfe is not None or analysis.mae is not None:
        mfe = f"${analysis.mfe:.2f}" if analysis.mfe is not None else "n/a"
        mae = f"${analysis.mae:.2f}" if analysis.mae is not None else "n/a"
        lines.append(f"- MFE: {mfe} · MAE: {mae}")
    lines.append(
        f"- Plan followed: **{'yes' if analysis.followed_plan else 'no'}** · "
        f"Risk approved: **{'yes' if analysis.risk_approved else 'no'}**"
    )
    lines.append(
        f"- Regime: market `{analysis.market_regime}` · "
        f"vol `{analysis.volatility_regime}` · "
        f"news risk `{analysis.news_risk_level}` · "
        f"time-of-day `{analysis.time_of_day_bucket}`"
    )
    lines.append("")

    # ----- Mistake tags -----
    lines.append("## Mistake tags")
    lines.append("")
    if tagging.tags:
        for tag in tagging.tags:
            detail = tagging.details.get(tag, "")
            lines.append(f"- `{tag.value}` — {detail}" if detail else f"- `{tag.value}`")
    else:
        lines.append("- (none)")
    lines.append("")

    # ----- Features at entry -----
    if analysis.features:
        lines.append("## Features at entry")
        lines.append("")
        for k in sorted(analysis.features.keys()):
            v = analysis.features[k]
            lines.append(f"- `{k}` = {v}")
        lines.append("")

    # ----- Optional agent narrative -----
    if agent_summary:
        lines.append("## Agent commentary")
        lines.append("")
        if agent_summary.get("headline"):
            lines.append(f"_{agent_summary['headline']}_")
            lines.append("")
        for label in ("why_taken", "why_outcome", "mistake_summary"):
            v = agent_summary.get(label)
            if v:
                lines.append(f"**{label.replace('_', ' ').title()}:** {v}")
                lines.append("")
        notes = agent_summary.get("review_notes") or []
        if notes:
            lines.append("**Review notes:**")
            lines.extend(f"- {n}" for n in notes)
            lines.append("")

    return "\n".join(lines)


def write_post_trade_report(
    analysis: PostTradeAnalysis,
    tagging: MistakeTagging,
    settings: Settings,
    *,
    out_dir: Optional[Path] = None,
    agent_summary: Optional[dict] = None,
) -> PostTradeReportArtifacts:
    """Write the per-trade Markdown to ``REPORTS_DIR/trades``."""
    log = get_logger("reports.post_trade_report")
    out_dir = (
        Path(out_dir) if out_dir is not None else Path(settings.REPORTS_DIR) / "trades"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    md_path = out_dir / f"trade_{analysis.trade_id}.md"
    md_path.write_text(
        render_post_trade_markdown(
            analysis, tagging, agent_summary=agent_summary
        )
    )
    log.info(
        "post_trade_report.written",
        path=str(md_path),
        trade_id=analysis.trade_id,
        n_tags=len(tagging.tags),
    )
    return PostTradeReportArtifacts(md_path=md_path)
