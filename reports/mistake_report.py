"""Daily mistake-digest report.

Aggregates a session's :class:`PatternMinerResult` plus the list of
proposed :class:`ImprovementCandidate` rows into a single Markdown file
that complements ``reports/daily_report.py``. Operators read this once
per day to spot recurring problems and decide whether to investigate or
backtest a proposed fix.

The writer is the only place that *materializes* the suggestions to the
DB (via ``persist_candidates``). That keeps the pattern miner pure /
read-only and ensures that a mistake report is what flips the audit
trail to "we proposed X today."
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from analysis.improvement_suggester import (
    ImprovementSuggester,
    persist_candidates,
)
from analysis.pattern_miner import PatternMiner, PatternMinerResult
from analysis.types import ImprovementCandidate
from app.logging_config import get_logger
from config.settings import Settings
from scheduler.market_hours import session_date


@dataclass(frozen=True)
class MistakeReportArtifacts:
    md_path: Path
    json_path: Path
    n_proposed: int


def render_mistake_markdown(
    *,
    session_label: str,
    instrument: str,
    pattern: PatternMinerResult,
    candidates: list[ImprovementCandidate],
) -> str:
    lines: list[str] = []
    lines.append(f"# Mistake report — {session_label} — {instrument}")
    lines.append("")
    lines.append(
        f"_Generated: {datetime.now(timezone.utc).isoformat()} UTC · "
        f"{pattern.n_total} analyzed trades_"
    )
    lines.append("")

    # ----- Top mistake tags -----
    lines.append("## Top mistake tags")
    lines.append("")
    if not pattern.top_mistake_tags:
        lines.append("- (none)")
    else:
        for tag, n in pattern.top_mistake_tags:
            stats = pattern.by_mistake_tag.get(tag)
            wr = f"win {stats.win_rate:.1%}" if stats else ""
            exp = f"exp ${stats.expectancy:.2f}" if stats else ""
            lines.append(f"- `{tag}` — {n} ({wr} · {exp})".strip())
    lines.append("")

    # ----- False-positive rates -----
    lines.append("## False-positive rate by strategy")
    lines.append("")
    if not pattern.false_positive_rates_by_strategy:
        lines.append("- (no strategies tracked)")
    else:
        for strategy, rate in sorted(
            pattern.false_positive_rates_by_strategy.items(),
            key=lambda kv: -kv[1],
        ):
            stats = pattern.by_strategy.get(strategy)
            n = stats.n_trades if stats else 0
            lines.append(f"- `{strategy}` — {rate:.2%} (n={n})")
    lines.append("")

    # ----- Time-of-day breakdown -----
    if pattern.by_time_of_day:
        lines.append("## By time-of-day")
        lines.append("")
        lines.append("| Bucket | Trades | Win | Expectancy |")
        lines.append("|--------|-------:|----:|-----------:|")
        for bucket, stats in sorted(pattern.by_time_of_day.items()):
            lines.append(
                f"| {bucket} | {stats.n_trades} | {stats.win_rate:.2%} | ${stats.expectancy:.2f} |"
            )
        lines.append("")

    # ----- Volatility regime -----
    if pattern.by_volatility_regime:
        lines.append("## By volatility regime")
        lines.append("")
        lines.append("| Regime | Trades | Win | Expectancy |")
        lines.append("|--------|-------:|----:|-----------:|")
        for regime, stats in sorted(pattern.by_volatility_regime.items()):
            lines.append(
                f"| {regime} | {stats.n_trades} | {stats.win_rate:.2%} | ${stats.expectancy:.2f} |"
            )
        lines.append("")

    # ----- Confidence buckets -----
    if pattern.by_confidence_bucket:
        lines.append("## By model-confidence bucket")
        lines.append("")
        lines.append("| Bucket | Trades | Win | Expectancy | False-pos rate |")
        lines.append("|--------|-------:|----:|-----------:|---------------:|")
        for bucket, stats in sorted(pattern.by_confidence_bucket.items()):
            lines.append(
                f"| {bucket} | {stats.n_trades} | {stats.win_rate:.2%} | "
                f"${stats.expectancy:.2f} | {stats.false_positive_rate:.2%} |"
            )
        lines.append("")

    # ----- Proposed improvements -----
    lines.append("## Proposed improvements (NOT auto-applied)")
    lines.append("")
    if not candidates:
        lines.append("- (none — nothing crossed sample-size + gap thresholds today)")
    else:
        lines.append(
            "Each suggestion below is logged with `validation_status='proposed'`. "
            "They are **never** auto-applied; the operator must validate them via "
            "`--retrain-from-feedback` (and `--promote-model` if the comparison "
            "report supports promotion)."
        )
        lines.append("")
        for c in candidates:
            lines.append(f"### `{c.suggestion_id}` ({c.risk_of_overfitting} overfit risk)")
            lines.append("")
            lines.append(f"- **Reason:** {c.reason}")
            if c.affected_strategy:
                lines.append(f"- **Strategy:** `{c.affected_strategy}`")
            if c.affected_condition:
                lines.append(f"- **Condition:** `{c.affected_condition}`")
            if c.expected_benefit:
                lines.append(f"- **Expected benefit:** {c.expected_benefit}")
            lines.append(
                "- **Supporting stats:** "
                + ", ".join(f"{k}={v}" for k, v in c.supporting_stats.items())
            )
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def write_mistake_report(
    settings: Settings,
    *,
    now: Optional[datetime] = None,
    out_dir: Optional[Path] = None,
    persist_proposals: bool = True,
) -> MistakeReportArtifacts:
    """Build today's mistake digest + persist proposals (always proposed)."""
    log = get_logger("reports.mistake_report")
    now = now or datetime.now(tz=timezone.utc)

    miner = PatternMiner()
    sd = session_date(now, settings)
    pattern = miner.aggregate(
        start=datetime.combine(sd, datetime.min.time(), tzinfo=timezone.utc),
        end=datetime.combine(sd, datetime.max.time(), tzinfo=timezone.utc),
        instrument=settings.INSTRUMENT,
    )

    suggester = ImprovementSuggester()
    candidates = suggester.propose(pattern)

    n_persisted = 0
    if persist_proposals and candidates:
        n_persisted = persist_candidates(candidates)

    out_dir = (
        Path(out_dir) if out_dir is not None else Path(settings.REPORTS_DIR) / "mistakes"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"mistakes_{sd.isoformat()}_{settings.INSTRUMENT}.md"
    json_path = out_dir / f"mistakes_{sd.isoformat()}_{settings.INSTRUMENT}.json"

    md_path.write_text(
        render_mistake_markdown(
            session_label=sd.isoformat(),
            instrument=settings.INSTRUMENT,
            pattern=pattern,
            candidates=candidates,
        )
    )
    json_path.write_text(
        json.dumps(
            {
                "session_date": sd.isoformat(),
                "instrument": settings.INSTRUMENT,
                "pattern": pattern.to_dict(),
                "proposed_improvements": [c.model_dump() for c in candidates],
            },
            indent=2,
            default=str,
        )
    )
    log.info(
        "mistake_report.written",
        md_path=str(md_path),
        json_path=str(json_path),
        n_candidates=len(candidates),
        n_persisted=n_persisted,
    )
    return MistakeReportArtifacts(
        md_path=md_path,
        json_path=json_path,
        n_proposed=len(candidates),
    )
