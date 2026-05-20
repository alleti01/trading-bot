"""Run autonomous workflows with safety defaults (DRY_RUN)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from agents.orchestrator import AgentOrchestrator, build_orchestrator
from app.logging_config import get_logger
from config.settings import Settings
from notifications.notification_service import NotificationService
from workflows.base import BaseWorkflow, WorkflowContext
from integrations.broker_router import build_broker
from workflows.broker_interface import PaperBrokerInterface
from workflows.daily_summary import DailySummaryWorkflow
from workflows.gates import WorkflowGates
from workflows.market_open import MarketOpenWorkflow
from workflows.memory import ensure_memory_files
from workflows.midday import MiddayWorkflow
from workflows.premarket import PremarketWorkflow
from workflows.schemas import WorkflowExecutionMode, WorkflowName, WorkflowResult
from scheduler.market_hours import session_date
from workflows.weekly_review import WeeklyReviewWorkflow

_WORKFLOWS: dict[str, BaseWorkflow] = {
    "premarket": PremarketWorkflow(),
    "market-open": MarketOpenWorkflow(),
    "midday": MiddayWorkflow(),
    "daily-summary": DailySummaryWorkflow(),
    "weekly-review": WeeklyReviewWorkflow(),
}

_DAY_SEQUENCE: tuple[str, ...] = (
    "premarket",
    "market-open",
    "midday",
    "daily-summary",
)


class WorkflowRunner:
    def __init__(
        self,
        settings: Settings,
        *,
        notifier: Optional[NotificationService] = None,
        orchestrator: Optional[AgentOrchestrator] = None,
        dry_run: Optional[bool] = None,
    ) -> None:
        self.settings = settings
        self.notifier = notifier or NotificationService.from_settings(settings)
        self.orchestrator = orchestrator
        self.dry_run = (
            dry_run
            if dry_run is not None
            else settings.WORKFLOW_EXECUTION_MODE == "DRY_RUN"
        )
        self.log = get_logger("workflows.runner")

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        cli_dry_run: Optional[bool] = None,
    ) -> "WorkflowRunner":
        orchestrator: Optional[AgentOrchestrator] = None
        if settings.ENABLE_LLM_AGENTS:
            orchestrator = build_orchestrator(settings, notifier=NotificationService.from_settings(settings))
        execution_dry = settings.WORKFLOW_EXECUTION_MODE == "DRY_RUN"
        if cli_dry_run is not None:
            execution_dry = cli_dry_run
        return cls(
            settings,
            orchestrator=orchestrator,
            dry_run=execution_dry,
        )

    def execution_mode(self) -> WorkflowExecutionMode:
        if self.dry_run:
            return "DRY_RUN"
        mode = self.settings.WORKFLOW_EXECUTION_MODE
        if mode == "LIVE":
            return "LIVE"
        return "PAPER" if mode == "PAPER" else "DRY_RUN"

    def run(
        self,
        workflow_name: WorkflowName,
        *,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> WorkflowResult:
        now = now or datetime.now(tz=timezone.utc)

        if self.settings.WORKFLOW_EXECUTION_MODE == "LIVE":
            return WorkflowResult(
                workflow=workflow_name,
                session_date=session_date(now, self.settings).isoformat(),
                execution_mode="LIVE",
                dry_run=self.dry_run,
                success=False,
                skipped=True,
                skip_reason=(
                    "WORKFLOW_EXECUTION_MODE=LIVE is locked — "
                    "no live workflow execution."
                ),
                errors=[
                    "WORKFLOW_EXECUTION_MODE=LIVE is locked — "
                    "no live workflow execution."
                ],
            )

        if workflow_name == "run-day":
            results = self.run_day(now=now, force=force)
            return WorkflowResult(
                workflow="run-day",
                session_date=results[0].session_date if results else "",
                execution_mode=self.execution_mode(),
                dry_run=self.dry_run,
                success=all(r.success for r in results),
                payload={"results": [r.model_dump() for r in results]},
            )

        wf = _WORKFLOWS.get(workflow_name)
        if wf is None:
            return WorkflowResult(
                workflow=workflow_name,
                session_date="",
                execution_mode=self.execution_mode(),
                dry_run=self.dry_run,
                success=False,
                errors=[f"Unknown workflow: {workflow_name}"],
            )

        ctx = self._build_context(now=now, force=force)
        self.log.info(
            "workflow.start",
            workflow=workflow_name,
            dry_run=ctx.dry_run,
            execution_mode=ctx.execution_mode,
        )
        result = wf.run(ctx)
        if result.memory_written or result.payload.get("research_written"):
            result.memory_written = True
        self.log.info("workflow.done", workflow=workflow_name, success=result.success)
        return result

    def run_day(
        self,
        *,
        now: Optional[datetime] = None,
        force: bool = False,
    ) -> list[WorkflowResult]:
        return [
            self.run(name, now=now, force=force)  # type: ignore[arg-type]
            for name in _DAY_SEQUENCE
        ]

    def _build_context(
        self, *, now: datetime, force: bool
    ) -> WorkflowContext:
        memory = ensure_memory_files(self.settings)
        mode = self.execution_mode()
        if mode == "LIVE":
            mode = "DRY_RUN"
            self.dry_run = True
        order_broker = None
        if not self.dry_run and mode == "PAPER":
            try:
                order_broker = build_broker(self.settings)
            except Exception as e:  # noqa: BLE001
                self.log.warning(
                    "workflow.broker_unavailable",
                    error=str(e),
                    provider=self.settings.BROKER_PROVIDER,
                )

        return WorkflowContext(
            settings=self.settings,
            notifier=self.notifier,
            broker=PaperBrokerInterface(self.settings),
            gates=WorkflowGates(self.settings),
            memory=memory,
            orchestrator=self.orchestrator,
            execution_mode=mode,
            dry_run=self.dry_run,
            now=now,
            force=force,
            order_broker=order_broker,
        )
