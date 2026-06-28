"""Continuous intraday loop + dynamic universe (watchlist) tests."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from config.settings import reload_settings
from storage.db import init_db
from tests._signal_stub import make_signal
from workflows.watchlist import build_scan_universe


def _settings(monkeypatch, tmp_path, **overrides):
    memory = tmp_path / "memory"
    memory.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKFLOW_MEMORY_DIR", str(memory))
    monkeypatch.setenv("WORKFLOW_EXECUTION_MODE", "PAPER")
    monkeypatch.setenv("BROKER_PROVIDER", "mock")
    monkeypatch.setenv("AUTONOMOUS_TRADING_ENABLED", "true")
    monkeypatch.setenv("ENABLED_SYMBOLS", "SPY,QQQ")
    monkeypatch.setenv("WORKFLOW_REFRESH_DATA_EACH_SCAN", "false")
    monkeypatch.setenv("WORKFLOW_LONG_ONLY", "true")
    monkeypatch.setenv("MAX_OPEN_POSITIONS", "4")
    monkeypatch.setenv("MAX_ACTIVE_SYMBOLS", "4")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


# ---------------------------------------------------------------------------
# Watchlist / universe
# ---------------------------------------------------------------------------
def test_universe_is_enabled_symbols_when_static(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_DYNAMIC_UNIVERSE="false")
    universe = build_scan_universe(settings)
    assert universe == ["SPY", "QQQ"]


def test_universe_expands_with_allowlist_when_dynamic(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch, tmp_path,
        WORKFLOW_DYNAMIC_UNIVERSE="true",
        WORKFLOW_MAX_UNIVERSE="8",
    )
    universe = build_scan_universe(settings)
    assert universe[:2] == ["SPY", "QQQ"]  # pinned first
    assert len(universe) == 8
    # All dynamically added names must be on the allowlist.
    from config.equity_allowlist import is_allowed_equity

    assert all(is_allowed_equity(s) for s in universe)


def test_agent_candidates_are_allowlist_gated(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch, tmp_path,
        WORKFLOW_DYNAMIC_UNIVERSE="true",
        WORKFLOW_MAX_UNIVERSE="6",
    )
    # "FAKE" is not on the allowlist and must be dropped; AAPL is allowed.
    universe = build_scan_universe(
        settings, agent_candidates=["AAPL", "FAKE", "NVDA"]
    )
    assert "FAKE" not in universe
    assert "AAPL" in universe
    assert "NVDA" in universe


def test_universe_respects_cap(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch, tmp_path,
        WORKFLOW_DYNAMIC_UNIVERSE="true",
        WORKFLOW_MAX_UNIVERSE="3",
    )
    universe = build_scan_universe(settings, agent_candidates=["AAPL", "NVDA", "AMD"])
    assert len(universe) == 3
    assert universe[:2] == ["SPY", "QQQ"]


# ---------------------------------------------------------------------------
# Intraday loop scan_once
# ---------------------------------------------------------------------------
def test_scan_once_dry_run_would_enter_not_order(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=True)
    now = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    with patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=lambda self, sym: make_signal(sym, price=100.0),
    ):
        summary = loop.scan_once(now=now)
    actions = {r["action"] for r in summary["results"]}
    assert "would_enter" in actions
    assert "enter" not in actions  # dry run never places orders


def test_scan_once_long_only_skips_shorts(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_LONG_ONLY="true")
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=True)
    now = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    with patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=lambda self, sym: make_signal(sym, direction="short", price=100.0),
    ):
        summary = loop.scan_once(now=now)
    reasons = {r.get("reason") for r in summary["results"]}
    assert "long_only" in reasons


def test_scan_once_places_orders_when_autonomous(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path)
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=False)
    now = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    with patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=lambda self, sym: make_signal(sym, price=100.0),
    ):
        summary = loop.scan_once(now=now)
    assert summary["entered"] >= 1


def test_scan_once_dry_run_routes_options_for_underlyings(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch, tmp_path,
        OPTIONS_ENABLED="true",
        OPTIONS_ENABLED_UNDERLYINGS="SPY,QQQ",
        WORKFLOW_DYNAMIC_UNIVERSE="false",
    )
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=True)
    now = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    with patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=lambda self, sym: make_signal(sym, price=100.0),
    ):
        summary = loop.scan_once(now=now)
    actions = {r["action"] for r in summary["results"]}
    assert "would_enter_option" in actions   # SPY/QQQ → options
    assert "would_enter" not in actions       # not equity shares
    assert "enter_option" not in actions      # dry run never places orders


def test_scan_once_live_routes_enabled_underlyings_to_options(monkeypatch, tmp_path) -> None:
    settings = _settings(
        monkeypatch, tmp_path,
        OPTIONS_ENABLED="true",
        OPTIONS_STRATEGY="atm_directional",
        OPTIONS_ENABLED_UNDERLYINGS="SPY,QQQ",
        OPTIONS_MAX_PREMIUM_PER_TRADE="100000",  # don't let premium cap block the test
        OPTIONS_STATE_PATH=str(tmp_path / "opt_positions.json"),
        WORKFLOW_DYNAMIC_UNIVERSE="false",
    )
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=False)
    now = datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc)
    with patch(
        "workflows.signal_engine.SignalEngine.generate_signal",
        new=lambda self, sym: make_signal(sym, price=100.0),
    ):
        summary = loop.scan_once(now=now)
    actions = [r["action"] for r in summary["results"]]
    assert "enter_option" in actions   # SPY/QQQ opened as options
    assert "enter" not in actions       # never an equity bracket for these names


def test_run_forever_skips_scan_outside_window(monkeypatch, tmp_path) -> None:
    # Outside the trading window → no scans, loop exits via max_cycles.
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_SCAN_INTERVAL_MINUTES="1")
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=True)
    loop._stop.wait = lambda *_a, **_k: None  # type: ignore[assignment]
    with patch("workflows.intraday_loop.is_in_trading_window", return_value=False):
        loop.run_forever(max_cycles=2)
    assert loop._scans == 0  # outside window: no scans


def test_run_forever_scans_inside_window(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_SCAN_INTERVAL_MINUTES="1")
    init_db()
    from unittest.mock import patch

    from workflows.intraday_loop import IntradayLoop

    loop = IntradayLoop(settings, dry_run=True)
    loop._stop.wait = lambda *_a, **_k: None  # type: ignore[assignment]
    with patch("workflows.intraday_loop.is_in_trading_window", return_value=True):
        loop.run_forever(max_cycles=2)
    assert loop._scans == 2  # inside window: scanned each cycle


def test_live_mode_scan_refused(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_EXECUTION_MODE="LIVE")
    init_db()
    from workflows.intraday_loop import IntradayLoop

    # dry_run=False so execution_mode resolves to LIVE in scan_once.
    loop = IntradayLoop(settings, dry_run=False)
    summary = loop.scan_once(now=datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc))
    assert summary.get("skipped") == "live_refused"
