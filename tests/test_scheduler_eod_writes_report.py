"""Scheduler EOD job writes the markdown + journal artifacts."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from notifications.notification_service import NotificationService
from reports.daily_report import DailyReportArtifacts, EndOfDaySummary
from scheduler.service import SchedulerService
from storage.db import init_db, session_scope
from storage.tables import ClosedTrade


def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def _seed_one_trade(base: datetime, instrument: str = "MES") -> None:
    with session_scope() as session:
        session.add(ClosedTrade(
            paper_trade_id=None, setup_id="setup-eod-1",
            instrument=instrument, direction="long", quantity=1.0,
            entry_ts=base, entry_price=4500.0,
            exit_ts=base + timedelta(minutes=4), exit_price=4504.0,
            exit_reason="tp", pnl=20.0, commission=0.5, slippage=0.0,
        ))


class _CapturingNotifier(NotificationService):
    """Records every notify() call so the test can assert on the EOD payload."""

    def __init__(self) -> None:
        super().__init__(discord=None)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def notify(self, kind: str, /, **payload: Any) -> None:  # type: ignore[override]
        self.calls.append((kind, dict(payload)))
        super().notify(kind, **payload)


class _DummyLoop:
    """Minimal stand-in for PaperTradingLoop — the EOD job doesn't touch it."""

    trading_enabled = True

    def on_bar_close(self, *_: Any, **__: Any):
        return None

    def flatten_now(self, *_: Any, **__: Any):
        return False


def test_scheduler_eod_writes_markdown_and_notifies_paths(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    # The EOD job filters by today's session date in the configured tz.
    # Hardcoding a fixed datetime made this test silently break once the
    # wall-clock day rolled past it; anchor to ``today()`` instead.
    from zoneinfo import ZoneInfo

    today_local = datetime.now(ZoneInfo(s.TIMEZONE)).date()
    base = datetime(
        today_local.year, today_local.month, today_local.day, 14, 0,
        tzinfo=ZoneInfo(s.TIMEZONE),
    ).astimezone(timezone.utc)
    _seed_one_trade(base)

    notifier = _CapturingNotifier()
    service = SchedulerService(
        settings=s, loop=_DummyLoop(), notifier=notifier, blocking=False  # type: ignore[arg-type]
    )

    # Directly invoke the wrapped EOD job. It MUST NOT raise.
    service._safe_end_of_day()

    # Files exist in the configured reports dir.
    daily_dir = Path(s.REPORTS_DIR) / "daily"
    journal_dir = Path(s.REPORTS_DIR) / "journals"
    md_files = list(daily_dir.glob("daily_*_MES.md"))
    json_files = list(daily_dir.glob("daily_*_MES.json"))
    journal_files = list(journal_dir.glob("trade_journal_*_MES.csv"))
    assert md_files and json_files and journal_files

    # And the EOD notification carried the artifact paths.
    eod_calls = [c for c in notifier.calls if c[0] == "eod.summary"]
    assert eod_calls, "expected at least one eod.summary call"
    payload = eod_calls[-1][1]
    assert payload["trades"] >= 1
    assert payload["md_path"].endswith(".md")
    assert payload["journal_path"].endswith(".csv")


def test_scheduler_eod_does_not_raise_on_empty_db(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    notifier = _CapturingNotifier()
    service = SchedulerService(
        settings=s, loop=_DummyLoop(), notifier=notifier, blocking=False  # type: ignore[arg-type]
    )
    service._safe_end_of_day()  # No exception even with zero trades.

    eod_calls = [c for c in notifier.calls if c[0] == "eod.summary"]
    assert eod_calls
    assert eod_calls[-1][1]["trades"] == 0
