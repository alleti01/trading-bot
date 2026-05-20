"""Base workflow template — orchestrates memory, broker, agents, gates."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from agents.orchestrator import AgentOrchestrator

if TYPE_CHECKING:
    from integrations.broker_base import BaseBroker
from app.logging_config import get_logger
from config.settings import Settings
from notifications.notification_service import NotificationService
from scheduler.market_hours import session_date
from workflows.broker_interface import BrokerInterface
from workflows.gates import WorkflowGates
from workflows.memory import MemoryPaths, ensure_memory_files
from workflows.schemas import WorkflowExecutionMode, WorkflowResult


class WorkflowContext:
    def __init__(
        self,
        *,
        settings: Settings,
        notifier: NotificationService,
        broker: BrokerInterface,
        gates: WorkflowGates,
        memory: MemoryPaths,
        orchestrator: Optional[AgentOrchestrator],
        execution_mode: WorkflowExecutionMode,
        dry_run: bool,
        now: datetime,
        force: bool = False,
        order_broker: Optional["BaseBroker"] = None,
    ) -> None:
        self.settings = settings
        self.notifier = notifier
        self.broker = broker
        self.gates = gates
        self.memory = memory
        self.orchestrator = orchestrator
        self.execution_mode = execution_mode
        self.dry_run = dry_run or execution_mode == "DRY_RUN"
        self.now = now
        self.force = force
        self.session_date = session_date(now, settings).isoformat()
        # Integrations broker (Tradovate demo / mock) for external orders.
        self.order_broker: Optional[BaseBroker] = order_broker
        self.entries_blocked: bool = False


class BaseWorkflow(ABC):
    name: str = "abstract"

    def __init__(self) -> None:
        self.log = get_logger(f"workflows.{self.name}")

    def run(self, ctx: WorkflowContext) -> WorkflowResult:
        result = WorkflowResult(
            workflow=self.name,
            session_date=ctx.session_date,
            execution_mode=ctx.execution_mode,
            dry_run=ctx.dry_run,
            success=False,
        )
        live_gate = ctx.gates.refuse_live(ctx.execution_mode)
        if not live_gate.passed:
            result.skipped = True
            result.skip_reason = live_gate.reason
            result.errors.append(live_gate.reason)
            return result

        if not ctx.force:
            weekday = ctx.gates.require_weekday(ctx.now)
            if not weekday.passed and self._respects_weekdays_only():
                result.skipped = True
                result.skip_reason = weekday.reason
                return result

        try:
            ensure_memory_files(ctx.settings)
            payload = self._execute(ctx)
            result.payload = payload
            result.success = True
            if ctx.settings.WORKFLOW_GIT_COMMIT:
                result.git_committed = _maybe_git_commit(ctx, self.name)
        except Exception as e:  # noqa: BLE001 — workflow must not crash caller
            self.log.exception("workflow.failed", workflow=self.name, error=str(e))
            result.errors.append(str(e))
            result.success = False
            try:
                ctx.notifier.notify(
                    "system.error",
                    source=f"workflow.{self.name}",
                    error=str(e),
                )
            except Exception:  # noqa: BLE001
                pass
        return result

    @abstractmethod
    def _execute(self, ctx: WorkflowContext) -> dict:
        ...

    def _respects_weekdays_only(self) -> bool:
        return True

    def _notify_safe(self, ctx: WorkflowContext, kind: str, **payload) -> bool:
        try:
            ctx.notifier.notify(kind, **payload)
            return True
        except Exception as e:  # noqa: BLE001
            self.log.warning("workflow.notify_failed", kind=kind, error=str(e))
            return False


def _maybe_git_commit(ctx: WorkflowContext, workflow_name: str) -> bool:
    root = ctx.memory.root.parent
    try:
        subprocess.run(
            ["git", "add", str(ctx.memory.root)],
            cwd=root,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                f"workflow({workflow_name}): update memory for {ctx.session_date}",
            ],
            cwd=root,
            check=True,
            capture_output=True,
        )
        if ctx.settings.WORKFLOW_GIT_PUSH:
            subprocess.run(
                ["git", "push"],
                cwd=root,
                check=True,
                capture_output=True,
            )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
