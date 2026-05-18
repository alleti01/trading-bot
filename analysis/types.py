"""Shared dataclasses + Pydantic types for the trade-analysis layer.

Two intentions kept apart:

- :class:`PostTradeAnalysis` is the Pydantic schema persisted to the DB
  (and shown in the per-trade Markdown report). It carries every field
  the spec asks for, including nullable orderflow placeholders and
  computed regime / time buckets.
- :class:`MistakeTag` is the enum of mistake categories the
  ``MistakeClassifier`` can emit. Multiple tags per trade are allowed.

The retrain/promotion code uses :class:`FeedbackDatasetRow` for export
shape and :class:`PromotionDecision` for the comparison report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# MistakeTag enum (deterministic, set by MistakeClassifier)
# ---------------------------------------------------------------------------
class MistakeTag(str, Enum):
    FALSE_POSITIVE = "false_positive"
    LOW_CONFIDENCE_TRADE = "low_confidence_trade"
    BAD_ORDERFLOW_CONFIRMATION = "bad_orderflow_confirmation"
    ORDERFLOW_DIVERGENCE = "orderflow_divergence"
    ENTERED_DURING_CHOP = "entered_during_chop"
    LOW_VOLUME_TRADE = "low_volume_trade"
    BAD_TIME_OF_DAY = "bad_time_of_day"
    HIGH_VOLATILITY_SPIKE = "high_volatility_spike"
    NEWS_RISK_TRADE = "news_risk_trade"
    STOP_TOO_TIGHT = "stop_too_tight"
    TARGET_TOO_FAR = "target_too_far"
    POOR_RISK_REWARD = "poor_risk_reward"
    STRATEGY_CONFLICT = "strategy_conflict"
    SLIPPAGE_LOSS = "slippage_loss"
    TIMEOUT_EXIT = "timeout_exit"
    RULE_VIOLATION = "rule_violation"
    UNKNOWN = "unknown"


# Buckets the analyzer assigns. They double as Pydantic Literals so
# downstream code (and the report) can rely on a fixed vocabulary.
TradeResult = Literal["win", "loss", "breakeven"]
NewsRiskLevel = Literal["low", "medium", "high", "unknown"]
TimeOfDayBucket = Literal["pre_market", "open", "mid_morning", "lunch", "afternoon", "close", "post_market", "off_hours"]
VolatilityRegime = Literal["low", "medium", "high", "unknown"]
MarketRegime = Literal["uptrend", "downtrend", "chop", "unknown"]
OverfitRisk = Literal["low", "medium", "high"]
ValidationStatus = Literal["proposed", "backtested", "approved", "rejected"]


# ---------------------------------------------------------------------------
# Per-trade analysis (Pydantic, persisted as JSON in trade_analyses.analysis)
# ---------------------------------------------------------------------------
class OrderflowFeatures(BaseModel):
    """Optional orderflow snapshot at entry. All fields nullable for now —
    the project does not have a real orderflow feed in the MVP."""

    model_config = ConfigDict(extra="forbid")

    bid_ask_imbalance: Optional[float] = None
    cumulative_delta: Optional[float] = None
    aggressor_ratio: Optional[float] = None
    poc_distance: Optional[float] = None  # distance from point-of-control


class PostTradeAnalysis(BaseModel):
    """Structured post-mortem for a single closed trade.

    Built deterministically by :class:`analysis.trade_analyzer.TradeAnalyzer`.
    The LLM agent later summarizes this object in plain English; it does
    not modify any field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    trade_id: str
    setup_id: Optional[str] = None
    instrument: str
    direction: Literal["long", "short"]
    strategy: Optional[str] = None

    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None

    result: TradeResult
    net_pnl: float
    gross_pnl: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    hold_seconds: float = 0.0
    r_multiple: float = 0.0  # net_pnl / planned_risk_dollars (signed)

    model_confidence: Optional[float] = None
    model_threshold: Optional[float] = None
    risk_approved: bool = True

    # Frozen feature snapshot at entry (canonical FEATURE_COLUMNS).
    features: dict[str, float] = Field(default_factory=dict)
    orderflow: OrderflowFeatures = Field(default_factory=OrderflowFeatures)

    market_regime: MarketRegime = "unknown"
    volatility_regime: VolatilityRegime = "unknown"
    time_of_day_bucket: TimeOfDayBucket = "off_hours"
    news_risk_level: NewsRiskLevel = "unknown"

    exit_reason: str
    followed_plan: bool = True

    mfe: Optional[float] = None  # max favorable excursion ($)
    mae: Optional[float] = None  # max adverse excursion ($)


# ---------------------------------------------------------------------------
# ImprovementCandidate (Pydantic; persisted via ImprovementSuggestion table)
# ---------------------------------------------------------------------------
class ImprovementCandidate(BaseModel):
    """A *proposed* change to the strategy/risk config. Never auto-applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suggestion_id: str
    reason: str
    affected_strategy: Optional[str] = None
    affected_condition: Optional[str] = None
    supporting_stats: dict[str, Any] = Field(default_factory=dict)
    expected_benefit: Optional[str] = None
    risk_of_overfitting: OverfitRisk = "medium"
    validation_status: ValidationStatus = "proposed"


# ---------------------------------------------------------------------------
# Feedback dataset row (used by FeedbackDataset and the retrain workflow)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeedbackDatasetRow:
    """One row per closed trade, suitable for retraining the model."""

    setup_id: str
    closed_trade_id: str
    entry_ts: datetime
    instrument: str
    direction: str
    strategy: Optional[str]
    label: int  # 1 = win (net_pnl > 0), 0 = loss/breakeven
    realized_pnl: float
    mfe: Optional[float]
    mae: Optional[float]
    exit_reason: str
    setup_type: Optional[str]
    model_confidence: Optional[float]
    mistake_tags: list[str]
    features: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Promotion comparison (used by analysis.promotion)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelComparison:
    """Side-by-side metric comparison: incumbent vs candidate."""

    incumbent_name: str
    incumbent_version: str
    candidate_name: str
    candidate_version: str
    incumbent_metrics: dict[str, float]
    candidate_metrics: dict[str, float]
    deltas: dict[str, float]
    walk_forward_stability: dict[str, float]


@dataclass(frozen=True)
class PromotionDecision:
    """Result of the promotion comparison.

    The decision is purely advisory — calling code (CLI ``--promote-model``)
    is what actually moves the registry pointer. We refuse to promote
    unless every gate is satisfied.
    """

    promote: bool
    rationale: str
    comparison: ModelComparison
    failed_gates: list[str]


# Required-improvement gates (the bar to clear before promotion).
PROMOTION_GATES: dict[str, str] = {
    "expectancy_per_trade": "Candidate expectancy per trade must be > incumbent.",
    "profit_factor": "Candidate profit factor must be > incumbent.",
    "max_drawdown_pct": "Candidate max drawdown % must be ≤ incumbent.",
    "false_positive_rate": "Candidate false-positive rate must be ≤ incumbent.",
    "walk_forward_stability": "Candidate walk-forward variance must be ≤ incumbent.",
}
