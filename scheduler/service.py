"""APScheduler-driven 24/7 paper service.

Jobs (configured at startup):

- ``bar_close``   — every ``BAR_INTERVAL_SECONDS`` seconds, polls the
  feed and lets :class:`PaperTradingLoop` run a bar cycle.
- ``force_flat``  — once per local trading day at ``FORCE_FLAT_TIME``
  (futures only). Forces any open position flat.
- ``end_of_day``  — once per local trading day shortly after flat time;
  generates a minimal :class:`EndOfDaySummary` and notifies Discord.
- ``heartbeat``   — once per local day at ``HEARTBEAT_LOCAL_TIME``;
  proves the service is alive even on quiet days.

Failure mode: every job is wrapped in a guard that catches and logs any
exception, then notifies ``system.error``. The scheduler itself stays
up; one bad bar cycle does not bring down the bot.
"""

from __future__ import annotations

import threading
import time as _time
from datetime import datetime, time, timezone
from typing import Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

from app.logging_config import get_logger
from config.settings import Settings
from notifications.notification_service import NotificationService
from paper.loop import PaperTradingLoop
from reports.daily_report import write_daily_report
from risk.kill_switch import KillSwitch
from scheduler.market_hours import is_trading_day


class SchedulerService:
    """Owns the APScheduler instance and the paper loop.

    ``run_forever`` blocks on a :class:`BlockingScheduler`. ``start`` /
    ``stop`` use a :class:`BackgroundScheduler` for tests.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        loop: PaperTradingLoop,
        notifier: NotificationService,
        kill_switch: Optional[KillSwitch] = None,
        blocking: bool = True,
    ) -> None:
        self.settings = settings
        self.loop = loop
        self.notifier = notifier
        self.kill_switch = kill_switch or KillSwitch()
        self.log = get_logger("scheduler.service")
        self._stop_event = threading.Event()

        if blocking:
            self.scheduler = BlockingScheduler(timezone=settings.TIMEZONE)
        else:
            self.scheduler = BackgroundScheduler(timezone=settings.TIMEZONE)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _add_jobs(self) -> None:
        flat_time: time = self.settings.force_flat_time()
        heartbeat_t = time.fromisoformat(self.settings.HEARTBEAT_LOCAL_TIME)

        self.scheduler.add_job(
            self._safe_bar_cycle,
            trigger=IntervalTrigger(seconds=self.settings.BAR_INTERVAL_SECONDS),
            id="bar_close",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )
        self.scheduler.add_job(
            self._safe_force_flat,
            trigger=CronTrigger(
                hour=flat_time.hour,
                minute=flat_time.minute,
                timezone=self.settings.TIMEZONE,
            ),
            id="force_flat",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._safe_end_of_day,
            # 5 minutes after flat — pad so the last bar's exit is recorded.
            trigger=CronTrigger(
                hour=flat_time.hour,
                minute=(flat_time.minute + 5) % 60,
                timezone=self.settings.TIMEZONE,
            ),
            id="end_of_day",
            replace_existing=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._safe_heartbeat,
            trigger=CronTrigger(
                hour=heartbeat_t.hour,
                minute=heartbeat_t.minute,
                timezone=self.settings.TIMEZONE,
            ),
            id="heartbeat",
            replace_existing=True,
            max_instances=1,
        )

    def _on_startup(self) -> None:
        ks = self.kill_switch.snapshot()
        self.notifier.notify(
            "bot.start",
            mode=self.settings.MODE,
            instrument=self.settings.INSTRUMENT,
            market_type=self.settings.MARKET_TYPE,
            timezone=self.settings.TIMEZONE,
            kill_switch_tripped=ks.tripped,
            trading_enabled=self.loop.trading_enabled,
        )
        self.log.info(
            "scheduler.startup",
            mode=self.settings.MODE,
            instrument=self.settings.INSTRUMENT,
            kill_switch_tripped=ks.tripped,
            trading_enabled=self.loop.trading_enabled,
        )
        if self.settings.MODE != "PAPER":
            self.log.warning(
                "scheduler.mode_mismatch",
                mode=self.settings.MODE,
                note="SchedulerService is paper-specific.",
            )

    def start(self) -> None:
        """Add jobs and start a non-blocking scheduler. Call ``stop`` to halt."""
        self._add_jobs()
        self.scheduler.start()
        self._on_startup()

    def stop(self, *, wait: bool = False) -> None:
        try:
            self.scheduler.shutdown(wait=wait)
        except Exception as e:  # pragma: no cover - shutdown can race
            self.log.warning("scheduler.shutdown_failed", error=str(e))
        self._stop_event.set()
        self.notifier.notify("bot.stop", mode=self.settings.MODE)

    def run_forever(self) -> None:
        """Blocking entry point used by ``app/main.py`` for ``MODE=PAPER``."""
        self._add_jobs()
        self._on_startup()
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):  # pragma: no cover - signal path
            self.log.info("scheduler.signal_exit")
        finally:
            self.notifier.notify("bot.stop", mode=self.settings.MODE)

    # ------------------------------------------------------------------
    # Smoke runner — lets --smoke-paper run a few bar cycles synchronously.
    # ------------------------------------------------------------------
    def run_smoke_cycles(self, n_cycles: int) -> list:
        """Run ``n_cycles`` bar cycles in-process and return their results."""
        if n_cycles < 1:
            raise ValueError("n_cycles must be >= 1")
        self._on_startup()
        results = []
        for _ in range(n_cycles):
            now = datetime.now(tz=timezone.utc)
            results.append(self.loop.on_bar_close(now))
            _time.sleep(0)  # cooperative yield
        # Always run the EOD summary at the end of a smoke run so we
        # exercise the full lifecycle once.
        try:
            self._safe_end_of_day()
        except Exception:  # pragma: no cover - belt-and-braces
            pass
        return results

    # ------------------------------------------------------------------
    # Job wrappers (each one MUST NOT raise)
    # ------------------------------------------------------------------
    def _safe_bar_cycle(self) -> None:
        try:
            now = datetime.now(tz=timezone.utc)
            if not is_trading_day(now, self.settings):
                return
            res = self.loop.on_bar_close(now)
            if res.errors:
                self.log.warning("scheduler.bar_cycle_errors", errors=res.errors)
        except Exception as e:
            self.log.error("scheduler.bar_cycle_unhandled", error=str(e))
            self.notifier.notify("system.error", kind="bar_cycle", error=str(e))

    def _safe_force_flat(self) -> None:
        try:
            if self.settings.MARKET_TYPE != "futures":
                return
            now = datetime.now(tz=timezone.utc)
            if not is_trading_day(now, self.settings):
                return
            self.loop.flatten_now(now, reason="forced_flat")
        except Exception as e:
            self.log.error("scheduler.force_flat_failed", error=str(e))
            self.notifier.notify("system.error", kind="force_flat", error=str(e))

    def _safe_end_of_day(self) -> None:
        try:
            now = datetime.now(tz=timezone.utc)
            artifacts = write_daily_report(self.settings, now=now)
            self.notifier.notify(
                "eod.summary",
                **artifacts.summary.to_payload(),
                md_path=str(artifacts.md_path),
                json_path=str(artifacts.json_path),
                journal_path=(
                    str(artifacts.journal_path) if artifacts.journal_path else ""
                ),
            )
        except Exception as e:
            self.log.error("scheduler.eod_failed", error=str(e))
            self.notifier.notify("system.error", kind="end_of_day", error=str(e))

    def _safe_heartbeat(self) -> None:
        try:
            now = datetime.now(tz=timezone.utc)
            self.notifier.notify(
                "heartbeat",
                mode=self.settings.MODE,
                instrument=self.settings.INSTRUMENT,
                trading_enabled=self.loop.trading_enabled,
                kill_switch_tripped=self.kill_switch.is_tripped(),
                ts=now.isoformat(),
            )
        except Exception as e:
            self.log.error("scheduler.heartbeat_failed", error=str(e))
