"""ModelDriftAgent: compare paper trading vs backtest expectations.

Most of this agent is deterministic statistics: we compare a small
set of metrics observed in paper mode (or the most recent live
sample) against the values recorded at training time and emit a
:class:`ModelDriftReport` with per-metric deltas and a single
overall ``severity``.

An LLM polish step is *optional*. When the operator provides a
provider for ``model_drift`` (via ``MODEL_DRIFT_AGENT_PROVIDER``) the
agent asks it to write a short narrative restating the deterministic
findings — the LLM cannot change ``severity`` or
``retrain_recommended``, only add commentary.

Why both paths?

- Pure stats path keeps the bot honest when no key is configured —
  the most important advisory output (was performance broken today?)
  must not depend on an external API being up.
- The LLM polish path keeps the operator's daily-report tone
  consistent with the other agents.

Architectural property: this module imports nothing from
``execution/`` or ``risk/`` (enforced by
``tests/test_agent_isolation.py``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from agents.base_agent import AgentContext, BaseAgent
from agents.llm_client import LLMClient, LLMClientError
from agents.schemas import (
    AgentResult,
    DriftSeverity,
    ModelDriftMetricDelta,
    ModelDriftReport,
)
from app.logging_config import get_logger


# Severity thresholds expressed as |pct_change|. Tuned for the kinds
# of drift that matter operationally — a 5% drop in expectancy on a
# small sample is noise; a 30% drop is a "watch", a 50% drop is a
# "warn", and a 75% drop or worse is an "alert" and triggers
# ``retrain_recommended``.
_DEFAULT_THRESHOLDS = {
    "watch": 0.30,
    "warn": 0.50,
    "alert": 0.75,
}

# Whether each metric is "higher-is-better" — drives the sign of
# ``pct_change`` we compare against the thresholds. Drawdown is the
# only metric where larger absolute value is worse.
_HIGHER_IS_BETTER = {
    "win_rate": True,
    "expectancy_per_trade": True,
    "profit_factor": True,
    "max_drawdown_dollars": False,
    "false_positive_rate": False,
}


def _classify_severity(
    pct_change: Optional[float],
    *,
    higher_is_better: bool,
    thresholds: dict[str, float] = _DEFAULT_THRESHOLDS,
) -> DriftSeverity:
    """Map a pct-change to a severity bucket.

    For higher-is-better metrics: a negative pct_change is bad. For
    drawdown: a positive pct_change (drawdown grew) is bad. We
    classify the magnitude of the *bad* direction; improvements never
    raise severity above ``"none"``.
    """
    if pct_change is None or math.isnan(pct_change) or math.isinf(pct_change):
        return "none"
    bad_magnitude = -pct_change if higher_is_better else pct_change
    if bad_magnitude <= 0:
        return "none"
    if bad_magnitude >= thresholds["alert"]:
        return "alert"
    if bad_magnitude >= thresholds["warn"]:
        return "warn"
    if bad_magnitude >= thresholds["watch"]:
        return "watch"
    return "none"


def _highest(a: DriftSeverity, b: DriftSeverity) -> DriftSeverity:
    order = {"none": 0, "watch": 1, "warn": 2, "alert": 3}
    return a if order[a] >= order[b] else b


def _delta(expected: Optional[float], observed: Optional[float]) -> Optional[float]:
    if expected is None or observed is None:
        return None
    try:
        return float(observed) - float(expected)
    except (TypeError, ValueError):
        return None


def _pct_change(
    expected: Optional[float], observed: Optional[float]
) -> Optional[float]:
    if expected is None or observed is None:
        return None
    try:
        e = float(expected)
        o = float(observed)
    except (TypeError, ValueError):
        return None
    if e == 0:
        return None
    return (o - e) / abs(e)


@dataclass(frozen=True)
class _MetricSpec:
    name: str
    expected_keys: tuple[str, ...]   # candidate keys in training metrics
    observed_keys: tuple[str, ...]   # candidate keys in paper metrics
    higher_is_better: bool


_METRIC_SPECS: tuple[_MetricSpec, ...] = (
    _MetricSpec(
        name="win_rate",
        expected_keys=("win_rate", "val_win_rate", "test_win_rate"),
        observed_keys=("win_rate",),
        higher_is_better=True,
    ),
    _MetricSpec(
        name="expectancy_per_trade",
        expected_keys=(
            "expectancy_per_trade",
            "expectancy",
            "val_expectancy_per_trade",
        ),
        observed_keys=("expectancy_per_trade", "expectancy"),
        higher_is_better=True,
    ),
    _MetricSpec(
        name="profit_factor",
        expected_keys=("profit_factor", "val_profit_factor"),
        observed_keys=("profit_factor",),
        higher_is_better=True,
    ),
    _MetricSpec(
        name="max_drawdown_dollars",
        expected_keys=(
            "max_drawdown_dollars",
            "max_drawdown",
            "val_max_drawdown_dollars",
        ),
        observed_keys=("max_drawdown_dollars", "max_drawdown"),
        higher_is_better=False,
    ),
    _MetricSpec(
        name="false_positive_rate",
        expected_keys=("false_positive_rate", "val_false_positive_rate"),
        observed_keys=("false_positive_rate",),
        higher_is_better=False,
    ),
)


def _first_match(d: dict[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for k in keys:
        v = d.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def compute_drift_report(
    *,
    model_metadata: Optional[dict[str, Any]],
    paper_metrics: Optional[dict[str, Any]],
) -> ModelDriftReport:
    """Pure-stats drift assessment.

    ``model_metadata`` is the orchestrator's standard metadata dict
    (``{"name", "version", "metrics": {...}, ...}``).
    ``paper_metrics`` is whatever metrics dict the daily report
    payload exposes. We accept ``None`` for either side and degrade
    to ``severity="none"`` with a clear ``reason``.
    """
    name = (
        str((model_metadata or {}).get("name", "unknown"))
        if model_metadata
        else "unknown"
    )
    version = (
        str((model_metadata or {}).get("version", "unknown"))
        if model_metadata
        else "unknown"
    )

    expected_metrics = (model_metadata or {}).get("metrics") or {}
    observed_metrics = paper_metrics or {}

    deltas: list[ModelDriftMetricDelta] = []
    overall_severity: DriftSeverity = "none"
    warnings: list[str] = []

    n_trades = 0
    try:
        n_trades = int(observed_metrics.get("n_trades") or 0)
    except (TypeError, ValueError):
        n_trades = 0

    if not expected_metrics or not observed_metrics:
        return ModelDriftReport(
            model_name=name or "unknown",
            model_version=version or "unknown",
            severity="none",
            metric_deltas=[],
            drift_warnings=[],
            commentary=None,
            narrative=None,
            retrain_recommended=False,
            reason=(
                "Insufficient inputs for drift evaluation: "
                "missing model metadata or paper metrics."
            ),
        )

    if n_trades and n_trades < 20:
        warnings.append(
            f"Sample size small (n_trades={n_trades}); deltas may be noisy."
        )

    for spec in _METRIC_SPECS:
        expected = _first_match(expected_metrics, spec.expected_keys)
        observed = _first_match(observed_metrics, spec.observed_keys)
        if expected is None and observed is None:
            continue
        delta = _delta(expected, observed)
        pct = _pct_change(expected, observed)
        severity = _classify_severity(
            pct, higher_is_better=spec.higher_is_better
        )
        deltas.append(
            ModelDriftMetricDelta(
                name=spec.name,
                expected=expected,
                observed=observed,
                delta=delta,
                pct_change=pct,
                severity=severity,
            )
        )
        overall_severity = _highest(overall_severity, severity)
        if severity != "none":
            warnings.append(
                f"{spec.name}: expected={expected!r} observed={observed!r} "
                f"pct_change={pct!r} -> {severity}"
            )

    retrain = overall_severity == "alert"
    if overall_severity == "none":
        reason = "All tracked metrics within tolerance."
    elif retrain:
        reason = (
            "At least one tracked metric degraded by more than the "
            "alert threshold; retraining is recommended."
        )
    else:
        reason = (
            f"Drift observed at severity={overall_severity}; "
            "monitor and consider scheduling a retrain experiment."
        )

    return ModelDriftReport(
        model_name=name or "unknown",
        model_version=version or "unknown",
        severity=overall_severity,
        metric_deltas=deltas,
        drift_warnings=warnings,
        commentary=None,
        narrative=None,
        retrain_recommended=retrain,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------
_NARRATIVE_SYSTEM_PROMPT = (
    "You are restating a deterministic model-drift report in plain "
    "English for an operator. You will be shown the JSON report; "
    "write a short narrative (<= 5 sentences) that restates the most "
    "important deltas and the agent's recommendation. You MUST NOT "
    "change the numeric findings, the severity, or the "
    "retrain_recommended flag. You MUST NOT propose model promotion "
    "or any change to risk rules."
)


class ModelDriftAgent(BaseAgent):
    """Stats-first drift agent with optional LLM polish.

    Unlike the other agents, this one does *not* require an LLM. The
    deterministic ``compute_drift_report`` produces a complete
    :class:`ModelDriftReport`; if the orchestrator hands us an
    :class:`LLMClient` we additionally ask the model to write a
    ``narrative``. Failing to construct or call the LLM never
    affects the deterministic output.
    """

    name: ClassVar[str] = "model_drift"
    schema_class = ModelDriftReport

    system_prompt: ClassVar[str] = _NARRATIVE_SYSTEM_PROMPT

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        # Override BaseAgent.__init__: the LLM is *optional* here.
        self.llm = llm
        self.log = get_logger(f"agents.{self.name}")

    # ---- BaseAgent contract ------------------------------------
    def build_user_prompt(self, context: AgentContext) -> str:  # pragma: no cover - unused
        # We do not use the standard ``run()`` LLM path; ``run`` is
        # overridden below. Implemented to satisfy the abstract base.
        return ""

    def run(self, context: AgentContext) -> AgentResult:  # type: ignore[override]
        try:
            paper_metrics = (
                context.paper_metrics
                or (context.daily_report or {}).get("metrics")
                or {}
            )
            report = compute_drift_report(
                model_metadata=context.model_metadata,
                paper_metrics=paper_metrics,
            )
        except Exception as e:  # noqa: BLE001 - drift compute must never raise
            self.log.error("model_drift.compute_failed", error=str(e))
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=None,
                error=f"compute_failed: {e}",
            )

        narrative: Optional[str] = None
        if self.llm is not None:
            try:
                user = (
                    f"agent={self.name}\n"
                    f"session_date={context.session_date}\n\n"
                    "Stats report (JSON):\n"
                    + json.dumps(report.model_dump(), indent=2, default=str)
                )
                raw = self.llm.complete(
                    system=_NARRATIVE_SYSTEM_PROMPT, user=user
                )
                narrative = (raw or "").strip() or None
            except LLMClientError as e:
                self.log.warning("model_drift.llm_error", error=str(e))
            except Exception as e:  # noqa: BLE001 - never let polish crash compute
                self.log.warning(
                    "model_drift.llm_unexpected", error=str(e)
                )

        # Re-build the report with the narrative attached. We can't
        # mutate frozen Pydantic models; ``model_copy`` is cheap.
        if narrative is not None:
            report = report.model_copy(update={"narrative": narrative})

        payload = report.model_dump()
        return AgentResult(
            agent_name=self.name,
            schema_valid=True,
            payload=payload,
            raw_text=None,
            error=None,
        )
