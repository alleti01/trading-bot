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
}
