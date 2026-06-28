"""Live paper report: options book breakout."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace

from config.settings import reload_settings
from reports.live_paper_report import (
    _open_options_summary,
    build_live_paper_report,
    discord_summary_lines,
)


class _FakeBroker:
    provider_name = "alpaca"

    def get_account(self):
        return SimpleNamespace(
            account_id="paper-1",
            equity=100_000.0,
            cash_balance=100_000.0,
            buying_power=200_000.0,
            realized_pnl=-12.34,
        )

    def get_positions(self):
        return []

    def get_open_orders(self):
        return []


def _write_state(path, **overrides):
    pos = {
        "occ_symbol": "QQQ260717C00480000",
        "underlying": "QQQ",
        "option_type": "call",
        "strike": 480.0,
        "expiry": "2026-07-17",
        "qty": 1,
        "entry_price": 5.00,
        "opened_at": "2026-06-27T14:30:00+00:00",
        "current_price": 6.50,
    }
    pos.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"positions": {pos["occ_symbol"]: pos}}), encoding="utf-8")


def _settings(monkeypatch, tmp_path, **overrides):
    monkeypatch.setenv("OPTIONS_ENABLED", "true")
    monkeypatch.setenv("OPTIONS_STATE_PATH", str(tmp_path / "opt.json"))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_open_options_summary_computes_unrealized(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _write_state(tmp_path / "opt.json")  # entry 5.00 → current 6.50
    now = datetime(2026, 6, 27, 20, 0, tzinfo=timezone.utc)
    summary = _open_options_summary(settings, now)
    assert summary["n_open"] == 1
    # (6.50 - 5.00) * 100 * 1 = 150.00
    assert summary["unrealized_pnl"] == 150.0
    row = summary["open"][0]
    assert row["underlying"] == "QQQ"
    assert row["dte"] == 20


def test_open_options_summary_handles_missing_state(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    now = datetime(2026, 6, 27, 20, 0, tzinfo=timezone.utc)
    summary = _open_options_summary(settings, now)
    assert summary["n_open"] == 0
    assert summary["open"] == []


def test_report_includes_options_section(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    _write_state(tmp_path / "opt.json")
    now = datetime(2026, 6, 27, 20, 0, tzinfo=timezone.utc)
    payload = build_live_paper_report(settings, broker=_FakeBroker(), now=now)
    assert payload["options"]["n_open"] == 1
    # Markdown report written with an options section.
    md = (tmp_path / "reports" / "live_paper" / "live_2026-06-27.md").read_text()
    assert "Open options (1)" in md
    assert "QQQ call" in md
    # Discord summary surfaces an options line.
    lines = discord_summary_lines(payload)
    assert any("Open options: 1" in ln for ln in lines)


def test_report_omits_options_when_disabled(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, OPTIONS_ENABLED="false")
    _write_state(tmp_path / "opt.json")
    now = datetime(2026, 6, 27, 20, 0, tzinfo=timezone.utc)
    payload = build_live_paper_report(settings, broker=_FakeBroker(), now=now)
    assert "options" not in payload
