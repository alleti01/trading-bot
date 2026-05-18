"""Candidate-vs-incumbent comparison for safe model promotion.

The workflow:

1. Build the :class:`FeedbackDataset` for the operator-specified window.
2. Train a candidate model (``models.trainer.train`` with walk-forward
   CV and the same feature set the incumbent saw).
3. Pull the incumbent's recorded validation metrics from
   ``models/<name>/<version>/metadata.json``.
4. Compare on the gates in :data:`analysis.types.PROMOTION_GATES`.
5. Return a :class:`PromotionDecision` — *advisory*. Calling code (CLI
   ``--promote-model``) is the only thing that ever moves a registry
   pointer, and it must check ``decision.promote`` before acting.

This module deliberately does **no** model writes itself.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from analysis.types import (
    ModelComparison,
    PromotionDecision,
    PROMOTION_GATES,
)
from app.logging_config import get_logger
from config.settings import Settings


def _load_metadata(model_dir: Path) -> dict[str, Any]:
    md_path = model_dir / "metadata.json"
    if not md_path.exists():
        raise FileNotFoundError(f"metadata.json missing at {md_path}")
    return json.loads(md_path.read_text())


def _walk_forward_stability(metadata: dict[str, Any]) -> float:
    """Return variance of ``roc_auc`` across walk-forward folds."""
    fold_metrics = metadata.get("fold_metrics") or []
    aucs = [float(f.get("roc_auc", 0.0)) for f in fold_metrics if f.get("roc_auc") is not None]
    if len(aucs) < 2:
        return 0.0
    mean = sum(aucs) / len(aucs)
    var = sum((a - mean) ** 2 for a in aucs) / (len(aucs) - 1)
    return float(var)


def _extract_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    agg = metadata.get("metrics") or {}
    out: dict[str, float] = {
        # Naming convention here is intentional — these are the gate keys.
        "roc_auc": float(agg.get("roc_auc", float("nan"))),
        "pr_auc": float(agg.get("pr_auc", float("nan"))),
        "precision_at_60": float(agg.get("precision_at_60", float("nan"))),
        "recall_at_60": float(agg.get("recall_at_60", float("nan"))),
    }
    out["walk_forward_stability"] = _walk_forward_stability(metadata)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compare(
    *,
    incumbent_metadata: dict[str, Any],
    candidate_metadata: dict[str, Any],
    realized_metrics_incumbent: Optional[dict[str, float]] = None,
    realized_metrics_candidate: Optional[dict[str, float]] = None,
) -> PromotionDecision:
    """Pure comparison: returns advisory :class:`PromotionDecision`.

    ``realized_metrics_*`` are optional overlays from real-world
    measurements (expectancy, profit factor, false-positive rate,
    max-drawdown). When provided, they take precedence over the
    metadata-derived values; the registry typically does not store
    realized post-cost trade metrics.
    """
    log = get_logger("analysis.promotion")

    incumbent_base = _extract_metrics(incumbent_metadata)
    candidate_base = _extract_metrics(candidate_metadata)
    if realized_metrics_incumbent:
        incumbent_base.update(realized_metrics_incumbent)
    if realized_metrics_candidate:
        candidate_base.update(realized_metrics_candidate)

    deltas = {
        k: float(candidate_base.get(k, 0.0)) - float(incumbent_base.get(k, 0.0))
        for k in set(incumbent_base) | set(candidate_base)
    }

    failed_gates: list[str] = []

    # Gate 1: expectancy_per_trade — only checked if both sides have it.
    if "expectancy_per_trade" in incumbent_base and "expectancy_per_trade" in candidate_base:
        if candidate_base["expectancy_per_trade"] <= incumbent_base["expectancy_per_trade"]:
            failed_gates.append("expectancy_per_trade")

    # Gate 2: profit_factor.
    if "profit_factor" in incumbent_base and "profit_factor" in candidate_base:
        if candidate_base["profit_factor"] <= incumbent_base["profit_factor"]:
            failed_gates.append("profit_factor")

    # Gate 3: max_drawdown_pct (lower is better).
    if "max_drawdown_pct" in incumbent_base and "max_drawdown_pct" in candidate_base:
        if candidate_base["max_drawdown_pct"] > incumbent_base["max_drawdown_pct"]:
            failed_gates.append("max_drawdown_pct")

    # Gate 4: false_positive_rate (lower is better).
    if "false_positive_rate" in incumbent_base and "false_positive_rate" in candidate_base:
        if candidate_base["false_positive_rate"] > incumbent_base["false_positive_rate"]:
            failed_gates.append("false_positive_rate")

    # Gate 5: walk-forward stability — variance must not increase.
    if candidate_base["walk_forward_stability"] > incumbent_base["walk_forward_stability"] * 1.10:
        failed_gates.append("walk_forward_stability")

    promote = len(failed_gates) == 0

    comparison = ModelComparison(
        incumbent_name=str(incumbent_metadata.get("name", "")),
        incumbent_version=str(incumbent_metadata.get("version", "")),
        candidate_name=str(candidate_metadata.get("name", "")),
        candidate_version=str(candidate_metadata.get("version", "")),
        incumbent_metrics=incumbent_base,
        candidate_metrics=candidate_base,
        deltas=deltas,
        walk_forward_stability={
            "incumbent": incumbent_base["walk_forward_stability"],
            "candidate": candidate_base["walk_forward_stability"],
        },
    )

    rationale = (
        "All promotion gates passed."
        if promote
        else "Failed gates: " + ", ".join(f"{g} ({PROMOTION_GATES.get(g, '')})" for g in failed_gates)
    )
    log.info(
        "promotion.compared",
        promote=promote,
        failed_gates=failed_gates,
        candidate_version=comparison.candidate_version,
        incumbent_version=comparison.incumbent_version,
    )
    return PromotionDecision(
        promote=promote,
        rationale=rationale,
        comparison=comparison,
        failed_gates=failed_gates,
    )


def write_comparison_report(
    decision: PromotionDecision,
    out_dir: Path,
    *,
    timestamp: Optional[datetime] = None,
) -> Path:
    """Persist a Markdown comparison report. The CLI consumes this for review."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp or datetime.now(timezone.utc)
    name = (
        f"promotion_{decision.comparison.candidate_name}_"
        f"{decision.comparison.candidate_version}_"
        f"{ts.strftime('%Y%m%dT%H%M%SZ')}.md"
    )
    path = out_dir / name

    lines = [
        f"# Promotion comparison — {decision.comparison.candidate_name}",
        "",
        f"_Generated: {ts.isoformat()}_",
        "",
        f"- Incumbent: `{decision.comparison.incumbent_name}` v`{decision.comparison.incumbent_version}`",
        f"- Candidate: `{decision.comparison.candidate_name}` v`{decision.comparison.candidate_version}`",
        f"- Decision: **{'PROMOTE' if decision.promote else 'HOLD'}**",
        f"- Rationale: {decision.rationale}",
        "",
        "## Metrics",
        "",
        "| Metric | Incumbent | Candidate | Δ |",
        "|--------|----------:|----------:|----:|",
    ]
    metric_keys = sorted(set(decision.comparison.incumbent_metrics) | set(decision.comparison.candidate_metrics))
    for k in metric_keys:
        inc = decision.comparison.incumbent_metrics.get(k, float("nan"))
        cand = decision.comparison.candidate_metrics.get(k, float("nan"))
        delta = decision.comparison.deltas.get(k, 0.0)
        lines.append(f"| `{k}` | {inc:.4f} | {cand:.4f} | {delta:+.4f} |")

    if decision.failed_gates:
        lines.extend(["", "## Failed gates", ""])
        for g in decision.failed_gates:
            lines.append(f"- `{g}` — {PROMOTION_GATES.get(g, '')}")
    lines.append("")
    path.write_text("\n".join(lines))
    return path
