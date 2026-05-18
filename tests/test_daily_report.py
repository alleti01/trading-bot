"""Daily report writer: payload contents, markdown rendering, file artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from config.settings import reload_settings
from reports.daily_report import (
    EndOfDaySummary,
    build_daily_report_payload,
    generate_end_of_day_summary,
    render_daily_markdown,
    write_daily_report,
)
from storage.db import init_db, session_scope
from storage.tables import ClosedTrade, Notification, RiskBlock


NY = ZoneInfo("America/New_York")


def _settings(tmp_path: Path, **overrides):
    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "TRADING_WINDOW_START": "09:30",
        "TRADING_WINDOW_END": "15:55",
        "FORCE_FLAT_TIME": "15:55",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def _seed_trades(*, n_long_wins: int = 1, n_long_losses: int = 0,
                 n_short_wins: int = 0, instrument: str = "MES",
                 base: datetime | None = None) -> None:
    base = base or datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    with session_scope() as session:
        i = 0
        for _ in range(n_long_wins):
            entry_ts = base + timedelta(minutes=10 * i)
            exit_ts = entry_ts + timedelta(minutes=4)
            session.add(ClosedTrade(
                paper_trade_id=None, setup_id=f"setup-{i}",
                instrument=instrument, direction="long", quantity=1.0,
                entry_ts=entry_ts, entry_price=4500.0, exit_ts=exit_ts,
                exit_price=4504.0, exit_reason="tp",
                pnl=20.0, commission=0.5, slippage=0.0,
            ))
            i += 1
        for _ in range(n_long_losses):
            entry_ts = base + timedelta(minutes=10 * i)
            exit_ts = entry_ts + timedelta(minutes=4)
            session.add(ClosedTrade(
                paper_trade_id=None, setup_id=f"setup-{i}",
                instrument=instrument, direction="long", quantity=1.0,
                entry_ts=entry_ts, entry_price=4500.0, exit_ts=exit_ts,
                exit_price=4498.0, exit_reason="sl",
                pnl=-10.0, commission=0.5, slippage=0.0,
            ))
            i += 1
        for _ in range(n_short_wins):
            entry_ts = base + timedelta(minutes=10 * i)
            exit_ts = entry_ts + timedelta(minutes=4)
            session.add(ClosedTrade(
                paper_trade_id=None, setup_id=f"setup-{i}",
                instrument=instrument, direction="short", quantity=1.0,
                entry_ts=entry_ts, entry_price=4505.0, exit_ts=exit_ts,
                exit_price=4501.0, exit_reason="tp",
                pnl=20.0, commission=0.5, slippage=0.0,
            ))
            i += 1


def _seed_risk_block(rule: str, *, base: datetime | None = None) -> None:
    base = base or datetime(2026, 5, 18, 13, 30, tzinfo=timezone.utc)
    with session_scope() as session:
        session.add(RiskBlock(
            setup_id="blocked-1", ts=base, rule=rule, reason="seeded",
        ))


def _seed_notification(kind: str, *, delivered: bool, base: datetime | None = None) -> None:
    base = base or datetime(2026, 5, 18, 14, 30, tzinfo=timezone.utc)
    with session_scope() as session:
        session.add(Notification(
            ts=base, channel="discord", kind=kind, payload={"x": 1},
            delivered=delivered, error=None if delivered else "test",
        ))


# ---------------------------------------------------------------------------
def test_eod_summary_with_no_trades(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    now = datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    summary = generate_end_of_day_summary(s, now=now)
    assert summary.trades == 0
    assert summary.net_pnl == 0.0
    assert summary.session_date == "2026-05-18"


def test_eod_summary_counts_only_today(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    today = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    yesterday = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
    _seed_trades(n_long_wins=2, base=today)
    _seed_trades(n_long_wins=5, base=yesterday)

    now = datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    summary = generate_end_of_day_summary(s, now=now)
    assert summary.trades == 2  # yesterday excluded


def test_payload_contains_all_sections(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _seed_trades(n_long_wins=1, n_long_losses=1, n_short_wins=1)
    _seed_risk_block("max_trades_per_day")
    _seed_notification("trade.opened", delivered=True)
    _seed_notification("system.error", delivered=False)

    now = datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    payload = build_daily_report_payload(s, now=now)

    assert payload["session_date"] == "2026-05-18"
    assert payload["instrument"] == "MES"
    assert payload["mode"] == "PAPER"
    assert payload["metrics"]["n_trades"] == 3
    assert payload["metrics"]["n_wins"] == 2
    assert payload["metrics"]["n_losses"] == 1
    assert payload["metrics"]["net_pnl"] == pytest.approx(30.0)
    assert len(payload["trades"]) == 3
    assert payload["risk_blocks_summary"] == {"max_trades_per_day": 1}
    assert payload["notifications_summary"]["trade.opened"]["delivered"] == 1
    assert payload["notifications_summary"]["system.error"]["failed"] == 1
    assert isinstance(payload["compliance"]["general"], list)
    assert isinstance(payload["compliance"]["tradeify"], list)


def test_render_markdown_handles_empty_session(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    payload = build_daily_report_payload(
        s, now=datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    )
    md = render_daily_markdown(payload)
    assert "# Daily report" in md
    assert "No trades placed today" in md
    assert "(no notifications recorded)" in md or "No risk blocks today" in md


def test_render_markdown_includes_trade_table(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _seed_trades(n_long_wins=2, n_long_losses=1)
    payload = build_daily_report_payload(
        s, now=datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    )
    md = render_daily_markdown(payload)
    assert "| # | Entry (UTC)" in md
    # Three long trades expected (both wins + the loss were seeded as longs).
    assert md.count("| long |") == 3
    assert "| sl |" in md
    assert "| tp |" in md


def test_consistency_compliance_flag_triggers_when_one_day_dominates(tmp_path: Path) -> None:
    s = _settings(tmp_path, CONSISTENCY_LIMIT_PERCENT="30")
    _seed_trades(n_long_wins=1)  # all profit on a single day
    payload = build_daily_report_payload(
        s, now=datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    )
    consistency = next(
        f for f in payload["compliance"]["tradeify"] if f["rule"] == "consistency"
    )
    assert consistency["triggered"] is True


def test_write_daily_report_creates_md_and_json_artifacts(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _seed_trades(n_long_wins=2)
    now = datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)

    artifacts = write_daily_report(s, now=now)
    assert artifacts.md_path.exists()
    assert artifacts.json_path.exists()
    assert artifacts.journal_path is not None
    assert artifacts.journal_path.exists()
    assert artifacts.summary.trades == 2
    assert artifacts.summary.net_pnl == pytest.approx(40.0)

    payload = json.loads(artifacts.json_path.read_text())
    assert payload["instrument"] == "MES"
    assert len(payload["trades"]) == 2

    md = artifacts.md_path.read_text()
    assert "Daily report" in md and "MES" in md


def test_write_daily_report_can_skip_journal(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _seed_trades(n_long_wins=1)
    artifacts = write_daily_report(
        s, now=datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc),
        include_journal=False,
    )
    assert artifacts.journal_path is None
