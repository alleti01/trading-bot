"""Deterministic workflow gates (no LLM)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.settings import Settings
from scheduler.market_hours import is_trading_day
from workflows.memory import has_research_for_date
from workflows.schemas import WorkflowExecutionMode


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str = ""


class WorkflowGates:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def refuse_live(self, execution_mode: WorkflowExecutionMode) -> GateResult:
        if execution_mode == "LIVE":
            return GateResult(
                passed=False,
                reason="WORKFLOW_EXECUTION_MODE=LIVE is locked — no live workflow execution.",
            )
        return GateResult(passed=True)

    def require_weekday(self, now: datetime) -> GateResult:
        if not self.settings.WORKFLOW_WEEKDAYS_ONLY:
            return GateResult(passed=True)
        if is_trading_day(now, self.settings):
            return GateResult(passed=True)
        return GateResult(passed=False, reason="Non-trading day (WORKFLOW_WEEKDAYS_ONLY=true).")

    def require_research(
        self, research_log_path, session_date: str  # noqa: ANN001 — Path
    ) -> GateResult:
        if has_research_for_date(research_log_path, session_date):
            return GateResult(passed=True)
        return GateResult(
            passed=False,
            reason=f"No dated research section for {session_date} in RESEARCH-LOG.md.",
        )

    def autonomous_execution_allowed(
        self, execution_mode: WorkflowExecutionMode
    ) -> bool:
        if not self.settings.AUTONOMOUS_TRADING_ENABLED:
            return False
        if execution_mode != "PAPER":
            return False
        if self.settings.AUTONOMOUS_PAPER_ONLY and execution_mode != "PAPER":
            return False
        return True

    def macro_research_enabled(self) -> bool:
        if not self.settings.PERPLEXITY_ENABLED:
            return False
        if not self.settings.ENABLE_LLM_AGENTS:
            return False
        provider = self.settings.provider_for_agent("macro_news")
        return provider not in {"none", "off", "disabled", "false", ""}

    def high_risk_news_blocks_trading(
        self, macro_risk_level: Optional[str], blocked_windows: bool
    ) -> GateResult:
        if macro_risk_level == "high" or blocked_windows:
            return GateResult(
                passed=False,
                reason="Macro news flagged high risk or blocked windows.",
            )
        return GateResult(passed=True)
