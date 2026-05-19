"""Pydantic schemas for LLM agent outputs.

Every agent must produce JSON that validates against one of these models.
The wrapper :class:`AgentResult` is what the orchestrator persists to
``storage.tables.AgentOutput``: the structured payload when valid, plus
the raw text and an error message when validation fails. Failed-parse
results are kept (with ``schema_valid=False``) for the audit trail —
they are never silently dropped, but they are also never used to drive
behavior.

Design rules applied to every schema:

- ``model_config = ConfigDict(frozen=True, extra="forbid")`` — strict.
- Only primitive Python types (no nested rich types) so JSON parses
  trivially and a small LLM cannot wedge an unsupported field shape.
- All fields have explicit constraints where useful (Literals, lengths).
- *Nothing* in any schema below is allowed to encode an action, a
  trade, a risk-rule change, or a model promotion — the agent layer
  is advisory-only, and the schemas are part of how that property
  is enforced.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Base config — applied to every output schema
# ---------------------------------------------------------------------------
_StrictConfig = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# 1. NewsAgent
# ---------------------------------------------------------------------------
Severity = Literal["low", "medium", "high"]


class NewsAssessment(BaseModel):
    """Pre-session / EOD news evaluation.

    The orchestrator may flip an in-memory ``high_risk_news_active`` flag
    when ``high_risk_window=True``. That flag is consumed by the paper
    loop and passed to ``risk_engine.evaluate(..., high_risk_news_window=…)``
    — strictly **block-only**.
    """

    model_config = _StrictConfig

    high_risk_window: bool
    severity: Severity
    events: list[str] = Field(default_factory=list, max_length=20)
    summary: str = Field(min_length=1, max_length=2000)
    recommendation: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# 2. RiskExplainerAgent
# ---------------------------------------------------------------------------
class RiskBlockExplanation(BaseModel):
    model_config = _StrictConfig

    rule: str = Field(min_length=1, max_length=64)
    count: int = Field(ge=0)
    explanation: str = Field(min_length=1, max_length=1000)


class RiskExplainerOutput(BaseModel):
    model_config = _StrictConfig

    session_date: str = Field(min_length=8, max_length=32)
    blocks: list[RiskBlockExplanation] = Field(default_factory=list, max_length=50)
    overall_assessment: str = Field(min_length=1, max_length=2000)
    operator_actions: list[str] = Field(default_factory=list, max_length=20)


# ---------------------------------------------------------------------------
# 3. TradeJournalAgent
# ---------------------------------------------------------------------------
class TradeJournalNarrative(BaseModel):
    model_config = _StrictConfig

    session_date: str = Field(min_length=8, max_length=32)
    highlights: list[str] = Field(default_factory=list, max_length=20)
    mistakes: list[str] = Field(default_factory=list, max_length=20)
    lessons: list[str] = Field(default_factory=list, max_length=20)
    best_trade_setup_id: Optional[str] = Field(default=None, max_length=64)
    worst_trade_setup_id: Optional[str] = Field(default=None, max_length=64)


# ---------------------------------------------------------------------------
# 4. ReportAgent
# ---------------------------------------------------------------------------
class ReportCommentary(BaseModel):
    model_config = _StrictConfig

    session_date: str = Field(min_length=8, max_length=32)
    headline: str = Field(min_length=1, max_length=200)
    bullets: list[str] = Field(default_factory=list, max_length=20)
    compliance_notes: list[str] = Field(default_factory=list, max_length=20)
    tomorrow_focus: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# 5. ModelReviewAgent
# ---------------------------------------------------------------------------
class ModelReviewOutput(BaseModel):
    model_config = _StrictConfig

    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    calibration_comment: str = Field(min_length=1, max_length=2000)
    drift_warnings: list[str] = Field(default_factory=list, max_length=20)
    retrain_recommended: bool
    reason: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# 6. TradeAnalysisAgent (per closed trade)
# ---------------------------------------------------------------------------
AnalysisConfidence = Literal["low", "medium", "high"]


class TradeAnalysisSummary(BaseModel):
    """Plain-English narration of one :class:`PostTradeAnalysis`.

    Strictly explanatory: the agent does not change a single mistake tag
    (those come from the deterministic classifier) and does not propose
    rule changes. ``review_notes`` are operator-facing reminders, not
    instructions.
    """

    model_config = _StrictConfig

    trade_id: str = Field(min_length=1, max_length=64)
    headline: str = Field(min_length=1, max_length=200)
    why_taken: str = Field(min_length=1, max_length=2000)
    why_outcome: str = Field(min_length=1, max_length=2000)
    mistake_summary: str = Field(min_length=1, max_length=2000)
    review_notes: list[str] = Field(default_factory=list, max_length=10)
    confidence_in_analysis: AnalysisConfidence = "medium"


# ---------------------------------------------------------------------------
# 7. MacroNewsAgent (Perplexity-routed)
# ---------------------------------------------------------------------------
RiskLevel = Literal["low", "medium", "high"]


class MacroNewsBlockedWindow(BaseModel):
    """One window during today's session that should be blocked or
    risk-reduced because of a scheduled macro / news event.

    The agent can only *recommend* blocking — it does not modify any
    risk rule directly. The orchestrator surfaces the union of these
    windows to the deterministic risk engine via the existing
    ``high_risk_news_window`` flag (block-only path).
    """

    model_config = _StrictConfig

    start: str = Field(min_length=1, max_length=64)        # ISO local-tz "HH:MM" or full ISO timestamp
    end: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=1, max_length=2000)
    severity: RiskLevel


class MacroNewsCitation(BaseModel):
    model_config = _StrictConfig

    url: str = Field(min_length=1, max_length=2048)
    title: Optional[str] = Field(default=None, max_length=512)
    snippet: Optional[str] = Field(default=None, max_length=2000)


class MacroNewsAssessment(BaseModel):
    """Output of :class:`MacroNewsAgent`.

    ``affected_symbols`` is a subset of the operator's
    ``ENABLED_SYMBOLS`` — the agent should not invent symbols. The
    orchestrator filters the list against the symbol universe before
    surfacing it in any way.
    """

    model_config = _StrictConfig

    risk_level: RiskLevel
    affected_symbols: list[str] = Field(default_factory=list, max_length=64)
    blocked_windows: list[MacroNewsBlockedWindow] = Field(
        default_factory=list, max_length=20
    )
    key_events: list[str] = Field(default_factory=list, max_length=20)
    sources: list[MacroNewsCitation] = Field(
        default_factory=list, max_length=20
    )
    summary: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# 8. BacktestCriticAgent
# ---------------------------------------------------------------------------
class BacktestWeakSpot(BaseModel):
    """One identified weak spot in a backtest.

    ``category`` keeps the schema strict — operators / dashboards can
    bucket weak spots without parsing free-form ``where`` strings.
    ``recommendation`` MUST be experiment-shaped (e.g. "run a
    walk-forward with this filter", "split confidence buckets at
    0.65"); we enforce this in the agent's system prompt and also via
    the convention that ``recommendation`` is short and prose-only.
    """

    model_config = _StrictConfig

    category: Literal[
        "time_window",
        "symbol",
        "confidence_bucket",
        "regime",
        "other",
    ]
    where: str = Field(min_length=1, max_length=200)
    severity: RiskLevel
    evidence: str = Field(min_length=1, max_length=1000)
    suggested_experiment: str = Field(min_length=1, max_length=1000)


class BacktestCritique(BaseModel):
    model_config = _StrictConfig

    overall_assessment: str = Field(min_length=1, max_length=2000)
    weak_spots: list[BacktestWeakSpot] = Field(
        default_factory=list, max_length=20
    )
    bad_time_windows: list[str] = Field(default_factory=list, max_length=20)
    weak_symbols: list[str] = Field(default_factory=list, max_length=64)
    bad_confidence_buckets: list[str] = Field(
        default_factory=list, max_length=20
    )
    bad_regimes: list[str] = Field(default_factory=list, max_length=20)
    suggested_experiments: list[str] = Field(
        default_factory=list, max_length=20
    )


# ---------------------------------------------------------------------------
# 9. ModelDriftAgent
# ---------------------------------------------------------------------------
DriftSeverity = Literal["none", "watch", "warn", "alert"]


class ModelDriftMetricDelta(BaseModel):
    """Per-metric delta vs a baseline (typically the model's training
    or backtest expectation).

    ``observed`` and ``expected`` are *floats* and the schema does not
    constrain their range — different metrics live on different
    scales (win_rate in [0,1], expectancy in $, drawdown in $, etc.).
    The agent's job is to surface the deltas; the consumer renders.
    """

    model_config = _StrictConfig

    name: str = Field(min_length=1, max_length=64)
    expected: Optional[float] = None
    observed: Optional[float] = None
    delta: Optional[float] = None
    pct_change: Optional[float] = None
    severity: DriftSeverity


class ModelDriftReport(BaseModel):
    """Output of :class:`ModelDriftAgent`.

    The deterministic stats path always populates ``metric_deltas``
    and ``severity``. ``commentary`` and ``narrative`` are optional —
    they exist so an LLM polish step can write plain English without
    being able to invent or override the numeric findings.
    """

    model_config = _StrictConfig

    model_name: str = Field(min_length=1, max_length=128)
    model_version: str = Field(min_length=1, max_length=64)
    severity: DriftSeverity
    metric_deltas: list[ModelDriftMetricDelta] = Field(
        default_factory=list, max_length=20
    )
    drift_warnings: list[str] = Field(default_factory=list, max_length=20)
    commentary: Optional[str] = Field(default=None, max_length=2000)
    narrative: Optional[str] = Field(default=None, max_length=2000)
    retrain_recommended: bool = False
    reason: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# 10. StrategyResearchAgent (Perplexity-routed, advisory only)
# ---------------------------------------------------------------------------
class StrategyExperimentIdea(BaseModel):
    """One suggested experiment.

    The schema deliberately omits anything that could read as a
    "change rule X to Y" instruction. ``hypothesis`` is the WHY,
    ``experiment_plan`` is the HOW (still as prose), and
    ``risks`` lists failure modes the operator should think about
    before scheduling time on the backtester.
    """

    model_config = _StrictConfig

    title: str = Field(min_length=1, max_length=200)
    hypothesis: str = Field(min_length=1, max_length=2000)
    experiment_plan: str = Field(min_length=1, max_length=2000)
    risks: list[str] = Field(default_factory=list, max_length=10)
    related_filters: list[str] = Field(default_factory=list, max_length=10)


class StrategyResearchReport(BaseModel):
    model_config = _StrictConfig

    summary: str = Field(min_length=1, max_length=2000)
    experiments: list[StrategyExperimentIdea] = Field(
        default_factory=list, max_length=10
    )
    sources: list[MacroNewsCitation] = Field(
        default_factory=list, max_length=20
    )


# ---------------------------------------------------------------------------
# 11. DataQualityAgent (deterministic — no LLM by default)
# ---------------------------------------------------------------------------
DataQualityIssueKind = Literal[
    "missing_candles",
    "duplicate_timestamps",
    "bad_ohlcv",
    "stale_feed",
    "symbol_data_gap",
    "empty_feed",
    "other",
]


class DataQualityIssue(BaseModel):
    """One detected data-quality problem.

    ``severity`` is a soft signal; the *blocking* decision is captured
    by ``DataQualityReport.blocked_symbols``. Critical issues
    (``stale_feed``, ``empty_feed``, ``bad_ohlcv``) automatically push
    their symbol onto the blocked list — see the deterministic checks
    in :class:`DataQualityAgent`.
    """

    model_config = _StrictConfig

    symbol: str = Field(min_length=1, max_length=32)
    kind: DataQualityIssueKind
    severity: RiskLevel
    detail: str = Field(min_length=1, max_length=2000)
    sample_timestamps: list[str] = Field(
        default_factory=list, max_length=10
    )


class DataQualityReport(BaseModel):
    """Output of :class:`DataQualityAgent`.

    ``blocked_symbols`` is the operative field: the orchestrator
    refuses to start the paper loop on these symbols and the daily
    report flags them. Everything else is advisory.
    """

    model_config = _StrictConfig

    session_date: str = Field(min_length=8, max_length=32)
    checked_symbols: list[str] = Field(default_factory=list, max_length=64)
    issues: list[DataQualityIssue] = Field(
        default_factory=list, max_length=200
    )
    blocked_symbols: list[str] = Field(default_factory=list, max_length=64)
    summary: str = Field(min_length=1, max_length=2000)


# ---------------------------------------------------------------------------
# Wrapper persisted to ``agent_outputs``
# ---------------------------------------------------------------------------
class AgentResult(BaseModel):
    """One agent run, valid or invalid.

    - ``schema_valid=True``: ``payload`` is the validated schema as a dict;
      ``error`` is None.
    - ``schema_valid=False``: ``payload`` is None; ``raw_text`` and ``error``
      describe what the LLM returned and why it could not be used.

    The orchestrator never raises a parse failure — it builds an
    ``AgentResult`` either way and writes it to the DB.
    """

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    schema_valid: bool
    payload: Optional[dict[str, Any]] = None
    raw_text: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Convenience map: agent name → schema class
# ---------------------------------------------------------------------------
AGENT_SCHEMAS: dict[str, type[BaseModel]] = {
    "news": NewsAssessment,
    "risk_explainer": RiskExplainerOutput,
    "trade_journal": TradeJournalNarrative,
    "report": ReportCommentary,
    "model_review": ModelReviewOutput,
    "trade_analysis": TradeAnalysisSummary,
    "macro_news": MacroNewsAssessment,
    "backtest_critic": BacktestCritique,
    "model_drift": ModelDriftReport,
    "strategy_research": StrategyResearchReport,
    "data_quality": DataQualityReport,
}
