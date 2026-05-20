"""Structured types for autonomous workflow runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

WorkflowName = Literal[
    "premarket",
    "market-open",
    "midday",
    "daily-summary",
    "weekly-review",
    "run-day",
]

WorkflowExecutionMode = Literal["DRY_RUN", "PAPER", "LIVE"]
TradeDecision = Literal["hold", "enter", "exit", "adjust", "skip"]
PlannedSide = Literal["long", "short", "flat"]


class AccountSnapshot(BaseModel):
    as_of: datetime
    session_date: str
    day_pnl: float = 0.0
    cumulative_pnl: float = 0.0
    trades_today: int = 0
    open_positions: int = 0
    open_orders: int = 0
    equity_estimate: float = 0.0


class PositionSnapshot(BaseModel):
    paper_trade_id: Optional[str] = None
    instrument: str
    direction: Literal["long", "short"]
    quantity: float
    entry_price: float
    stop_price: float
    target_price: float
    current_price: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    unrealized_pnl_pct: Optional[float] = None
    thesis: Optional[str] = None


class OrderSnapshot(BaseModel):
    """Placeholder for broker open orders (paper MVP has none separate)."""

    order_id: str
    instrument: str
    side: str
    quantity: float
    status: str = "open"


class BrokerState(BaseModel):
    account: AccountSnapshot
    positions: list[PositionSnapshot] = Field(default_factory=list)
    orders: list[OrderSnapshot] = Field(default_factory=list)


class PlannedTradeIdea(BaseModel):
    symbol: str
    side: PlannedSide = "flat"
    catalyst: str = ""
    entry_zone: str = ""
    stop: str = ""
    target: str = ""
    risk_factors: list[str] = Field(default_factory=list)
    default_decision: TradeDecision = "hold"
    thesis: str = ""


class ResearchLogEntry(BaseModel):
    session_date: str
    account: AccountSnapshot
    market_context: str = ""
    planned_trades: list[PlannedTradeIdea] = Field(default_factory=list, max_length=5)
    macro_summary: Optional[str] = None
    macro_risk_level: Optional[str] = None
    default_decision: TradeDecision = "hold"
    notes: str = ""


class SimulatedTradeDecision(BaseModel):
    symbol: str
    side: PlannedSide
    decision: TradeDecision
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    thesis: str = ""
    risk_reward: Optional[str] = None
    reason: str = ""


class WorkflowAction(BaseModel):
    kind: str
    detail: str
    symbol: Optional[str] = None
    simulated: bool = False


class WorkflowResult(BaseModel):
    workflow: str
    session_date: str
    execution_mode: WorkflowExecutionMode
    dry_run: bool
    success: bool
    skipped: bool = False
    skip_reason: Optional[str] = None
    actions: list[WorkflowAction] = Field(default_factory=list)
    discord_sent: bool = False
    memory_written: bool = False
    git_committed: bool = False
    errors: list[str] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)
