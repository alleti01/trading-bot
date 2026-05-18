"""Generate *proposed* improvement candidates from PatternMiner output.

Every candidate is created with ``validation_status="proposed"``. The
suggester never advances state on its own — promotion to ``backtested``
or ``approved`` only happens through the explicit retrain/promotion
workflow (``analysis.promotion``).

Heuristics are intentionally conservative:

- A pattern needs a minimum sample size (``MIN_TRADES_PER_BUCKET``)
  before it can yield a suggestion. This is the simplest defense
  against noise-driven recommendations on a quiet day.
- The win rate / expectancy delta vs. the global mean must clear a gap
  threshold; thin signals do not get suggestions.
- Each candidate carries a ``risk_of_overfitting`` rating that callers
  surface in the daily report so the operator can sanity-check before
  any backtesting work.
"""

from __future__ import annotations

import uuid
from typing import Iterable, Optional

from analysis.pattern_miner import GroupStats, PatternMinerResult
from analysis.types import ImprovementCandidate, OverfitRisk
from app.logging_config import get_logger
from sqlalchemy import select
from storage.db import session_scope
from storage.tables import ImprovementSuggestion as ImprovementSuggestionRow


# Tunables — kept here so all the magic numbers are visible. Production
# usage would expose them via Settings, but for the MVP they're constants.
MIN_TRADES_PER_BUCKET: int = 5
WIN_RATE_GAP_FOR_SUGGESTION: float = 0.20  # 20 pp below global win rate
EXPECTANCY_GAP_DOLLARS: float = 5.0
HIGH_FALSE_POSITIVE_RATE: float = 0.30


def _new_suggestion_id() -> str:
    return str(uuid.uuid4())


def _overfit_risk(n: int) -> OverfitRisk:
    """Tiny sample → high overfit risk; many trades → low."""
    if n < 10:
        return "high"
    if n < 30:
        return "medium"
    return "low"


def _global_stats(result: PatternMinerResult) -> GroupStats | None:
    """Roll up all strategies into a global baseline for delta math."""
    if not result.by_strategy:
        return None
    n_total = sum(g.n_trades for g in result.by_strategy.values())
    if n_total == 0:
        return None
    wins = sum(g.n_wins for g in result.by_strategy.values())
    losses = sum(g.n_losses for g in result.by_strategy.values())
    pnl = sum(g.expectancy * g.n_trades for g in result.by_strategy.values())
    return GroupStats(
        n_trades=n_total,
        n_wins=wins,
        n_losses=losses,
        win_rate=(wins / n_total) if n_total else 0.0,
        expectancy=(pnl / n_total) if n_total else 0.0,
        avg_pnl=(pnl / n_total) if n_total else 0.0,
        false_positive_rate=0.0,
    )


# ---------------------------------------------------------------------------
# Suggester
# ---------------------------------------------------------------------------
class ImprovementSuggester:
    def __init__(self) -> None:
        self.log = get_logger("analysis.improvement_suggester")

    def propose(self, result: PatternMinerResult) -> list[ImprovementCandidate]:
        out: list[ImprovementCandidate] = []
        baseline = _global_stats(result)
        baseline_win = baseline.win_rate if baseline is not None else 0.5
        baseline_exp = baseline.expectancy if baseline is not None else 0.0

        # 1) Time-of-day bucket suggestions.
        for bucket, stats in result.by_time_of_day.items():
            if stats.n_trades < MIN_TRADES_PER_BUCKET:
                continue
            win_gap = baseline_win - stats.win_rate
            exp_gap = baseline_exp - stats.expectancy
            if win_gap >= WIN_RATE_GAP_FOR_SUGGESTION or exp_gap >= EXPECTANCY_GAP_DOLLARS:
                out.append(
                    ImprovementCandidate(
                        suggestion_id=_new_suggestion_id(),
                        reason=(
                            f"Trades during '{bucket}' bucket underperform "
                            f"(win_rate={stats.win_rate:.2%} vs baseline "
                            f"{baseline_win:.2%}; expectancy=${stats.expectancy:.2f})."
                        ),
                        affected_strategy=None,
                        affected_condition=f"time_of_day_bucket={bucket}",
                        supporting_stats=stats.to_dict(),
                        expected_benefit=(
                            "Pause trading during this bucket; backtest first "
                            "on at least 30 sessions to confirm."
                        ),
                        risk_of_overfitting=_overfit_risk(stats.n_trades),
                    )
                )

        # 2) Volatility regime suggestions.
        for regime, stats in result.by_volatility_regime.items():
            if stats.n_trades < MIN_TRADES_PER_BUCKET:
                continue
            if stats.win_rate < 0.4 and stats.expectancy < 0:
                out.append(
                    ImprovementCandidate(
                        suggestion_id=_new_suggestion_id(),
                        reason=(
                            f"Volatility regime '{regime}' has poor win rate "
                            f"({stats.win_rate:.2%}) and negative expectancy "
                            f"(${stats.expectancy:.2f})."
                        ),
                        affected_strategy=None,
                        affected_condition=f"volatility_regime={regime}",
                        supporting_stats=stats.to_dict(),
                        expected_benefit=(
                            "Block entries during this regime; validate via walk-forward."
                        ),
                        risk_of_overfitting=_overfit_risk(stats.n_trades),
                    )
                )

        # 3) Per-strategy false-positive rate.
        for strategy, fp_rate in result.false_positive_rates_by_strategy.items():
            stats = result.by_strategy.get(strategy)
            if stats is None or stats.n_trades < MIN_TRADES_PER_BUCKET:
                continue
            if fp_rate >= HIGH_FALSE_POSITIVE_RATE:
                out.append(
                    ImprovementCandidate(
                        suggestion_id=_new_suggestion_id(),
                        reason=(
                            f"Strategy '{strategy}' has false-positive rate "
                            f"{fp_rate:.2%} (≥ {HIGH_FALSE_POSITIVE_RATE:.0%})."
                        ),
                        affected_strategy=strategy,
                        affected_condition="false_positive_rate",
                        supporting_stats={**stats.to_dict(), "false_positive_rate": fp_rate},
                        expected_benefit=(
                            "Raise CONFIDENCE_THRESHOLD for this strategy after "
                            "retraining and walk-forward validation."
                        ),
                        risk_of_overfitting=_overfit_risk(stats.n_trades),
                    )
                )

        # 4) Confidence-bucket suggestions — only for low-conf buckets.
        for bucket, stats in result.by_confidence_bucket.items():
            if bucket in ("no_model", "0.85+"):
                continue
            if stats.n_trades < MIN_TRADES_PER_BUCKET:
                continue
            if stats.win_rate < 0.4 and stats.expectancy < 0:
                out.append(
                    ImprovementCandidate(
                        suggestion_id=_new_suggestion_id(),
                        reason=(
                            f"Confidence bucket '{bucket}' has poor stats "
                            f"(win_rate={stats.win_rate:.2%}, "
                            f"expectancy=${stats.expectancy:.2f})."
                        ),
                        affected_strategy=None,
                        affected_condition=f"confidence_bucket={bucket}",
                        supporting_stats=stats.to_dict(),
                        expected_benefit=(
                            "Consider raising threshold past this bucket. Requires retrain."
                        ),
                        risk_of_overfitting=_overfit_risk(stats.n_trades),
                    )
                )

        # 5) Mistake-tag-driven suggestions for the most common offenders.
        for tag, count in result.top_mistake_tags[:5]:
            stats = result.by_mistake_tag.get(tag)
            if stats is None or stats.n_trades < MIN_TRADES_PER_BUCKET:
                continue
            if stats.win_rate < 0.5:
                out.append(
                    ImprovementCandidate(
                        suggestion_id=_new_suggestion_id(),
                        reason=(
                            f"Tag '{tag}' appears on {count} trades; "
                            f"win rate post-tag is {stats.win_rate:.2%}."
                        ),
                        affected_strategy=None,
                        affected_condition=f"mistake_tag={tag}",
                        supporting_stats=stats.to_dict(),
                        expected_benefit=(
                            f"Investigate the cluster of '{tag}' trades; consider "
                            "tightening pre-conditions or skipping entirely."
                        ),
                        risk_of_overfitting=_overfit_risk(stats.n_trades),
                    )
                )

        self.log.info("improvement_suggester.proposed", n=len(out))
        return out


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def persist_candidates(candidates: Iterable[ImprovementCandidate]) -> int:
    """Insert every candidate as ``validation_status='proposed'``.

    Returns the count actually written. Existing rows with the same
    ``suggestion_id`` are skipped so re-runs are idempotent.
    """
    log = get_logger("analysis.improvement_suggester")
    written = 0
    candidates = list(candidates)
    if not candidates:
        return 0

    with session_scope() as session:
        existing = {
            row.suggestion_id
            for row in session.execute(
                select(ImprovementSuggestionRow.suggestion_id)
            ).all()
        }
        for c in candidates:
            if c.suggestion_id in existing:
                continue
            session.add(
                ImprovementSuggestionRow(
                    suggestion_id=c.suggestion_id,
                    affected_strategy=c.affected_strategy,
                    affected_condition=c.affected_condition,
                    reason=c.reason,
                    supporting_stats=c.supporting_stats,
                    expected_benefit=c.expected_benefit,
                    risk_of_overfitting=c.risk_of_overfitting,
                    validation_status=c.validation_status,
                )
            )
            written += 1
    log.info("improvement_suggester.persisted", written=written, total=len(candidates))
    return written


def list_proposed(limit: Optional[int] = None) -> list[ImprovementCandidate]:
    """Return all rows with ``validation_status='proposed'`` (newest first)."""
    with session_scope() as session:
        stmt = (
            select(ImprovementSuggestionRow)
            .where(ImprovementSuggestionRow.validation_status == "proposed")
            .order_by(ImprovementSuggestionRow.created_at.desc())
        )
        if limit is not None:
            stmt = stmt.limit(int(limit))
        rows = list(session.execute(stmt).scalars().all())

    return [
        ImprovementCandidate(
            suggestion_id=r.suggestion_id,
            reason=r.reason or "",
            affected_strategy=r.affected_strategy,
            affected_condition=r.affected_condition,
            supporting_stats=dict(r.supporting_stats or {}),
            expected_benefit=r.expected_benefit,
            risk_of_overfitting=r.risk_of_overfitting,  # type: ignore[arg-type]
            validation_status=r.validation_status,  # type: ignore[arg-type]
        )
        for r in rows
    ]
