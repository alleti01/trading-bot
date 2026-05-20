"""APScheduler driver for autonomous workflows (separate from paper bar loop)."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from app.logging_config import get_logger
from config.settings import Settings
from workflows.workflow_runner import WorkflowRunner


class WorkflowScheduler:
    """Cron-style workflow scheduler in WORKFLOW_TIMEZONE."""

    def __init__(
        self,
        settings: Settings,
        runner: WorkflowRunner,
        *,
        blocking: bool = True,
    ) -> None:
        self.settings = settings
        self.runner = runner
        self.log = get_logger("workflows.scheduler")
        self._stop = threading.Event()
        if blocking:
            self.scheduler = BlockingScheduler(timezone=settings.WORKFLOW_TIMEZONE)
        else:
            self.scheduler = BackgroundScheduler(timezone=settings.WORKFLOW_TIMEZONE)

    def _safe_run(self, name: str) -> None:
        try:
            self.runner.run(name)  # type: ignore[arg-type]
        except Exception as e:  # noqa: BLE001
            self.log.exception("workflow.scheduler_job_failed", workflow=name, error=str(e))
            try:
                self.runner.notifier.notify(
                    "system.error",
                    source=f"workflow.scheduler.{name}",
                    error=str(e),
                )
            except Exception:  # noqa: BLE001
                pass

    def _add_jobs(self) -> None:
        tz = self.settings.WORKFLOW_TIMEZONE
        self.scheduler.add_job(
            lambda: self._safe_run("premarket"),
            CronTrigger(hour=8, minute=0, timezone=tz),
            id="workflow_premarket",
            replace_existing=True,
        )
        self.scheduler.add_job(
            lambda: self._safe_run("market-open"),
            CronTrigger(hour=9, minute=31, timezone=tz),
            id="workflow_market_open",
            replace_existing=True,
        )
        self.scheduler.add_job(
            lambda: self._safe_run("midday"),
            CronTrigger(hour=12, minute=0, timezone=tz),
            id="workflow_midday",
            replace_existing=True,
        )
        self.scheduler.add_job(
            lambda: self._safe_run("daily-summary"),
            CronTrigger(hour=16, minute=5, timezone=tz),
            id="workflow_daily_summary",
            replace_existing=True,
        )
        self.scheduler.add_job(
            lambda: self._safe_run("weekly-review"),
            CronTrigger(day_of_week="fri", hour=16, minute=10, timezone=tz),
            id="workflow_weekly_review",
            replace_existing=True,
        )

    def run_forever(self) -> None:
        self._add_jobs()
        self.log.info("workflow.scheduler.start", timezone=self.settings.WORKFLOW_TIMEZONE)
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.log.info("workflow.scheduler.stop")
            self.scheduler.shutdown(wait=False)

    def start_background(self) -> None:
        self._add_jobs()
        self.scheduler.start()

    def stop(self) -> None:
        self.scheduler.shutdown(wait=False)
        self._stop.set()
