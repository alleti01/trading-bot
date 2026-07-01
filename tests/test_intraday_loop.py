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


# ---------------------------------------------------------------------------
# Close detection + realised P&L notifications
# ---------------------------------------------------------------------------
class _CapturingNotifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def notify(self, kind: str, **payload) -> None:
        self.calls.append((kind, payload))


class _FakeExitBroker:
    """Minimal broker stub exposing get_last_exit_fill for close tests."""

    def __init__(self, fills=None) -> None:
        from integrations.broker_base import ExitFill

        self._fills = fills or {}
        self._ExitFill = ExitFill
        self.closed: list[str] = []

    def get_last_exit_fill(self, *, symbol, exit_side):
        fill = self._fills.get(symbol.upper())
        if fill is not None and fill.side == exit_side:
            return fill
        return None

    def close_position(self, *, symbol):
        from integrations.broker_base import OrderResult

        self.closed.append(symbol.upper())
        return OrderResult(
            success=True, simulated=True, order_id="x", symbol=symbol.upper(),
            side="sell", quantity=0.0, order_type="market", status="filled",
        )


def test_detect_close_reports_realized_pnl_win(monkeypatch, tmp_path) -> None:
    from integrations.broker_base import ExitFill
    from workflows.intraday_loop import IntradayLoop

    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    note = _CapturingNotifier()
    loop.notifier = note  # type: ignore[assignment]

    # Cycle 1: AAPL is open (entry 100, 10 shares long).
    recon1 = {"positions": [
        {"symbol": "AAPL", "average_price": 100.0, "quantity": 10, "direction": "long"}
    ], "orders": []}
    broker = _FakeExitBroker()
    loop._detect_and_notify_closes(broker, recon1)
    assert note.calls == []  # nothing closed yet

    # Cycle 2: AAPL gone; its take-profit filled at 105 → +$50 win.
    broker._fills["AAPL"] = ExitFill(
        symbol="AAPL", price=105.0, quantity=10, side="sell", exit_kind="take_profit"
    )
    loop._detect_and_notify_closes(broker, {"positions": [], "orders": []})

    assert len(note.calls) == 1
    kind, payload = note.calls[0]
    assert kind == "trade.closed"
    assert payload["net_pnl"] == 50.0
    assert payload["result"] == "WIN"
    assert payload["exit_reason"] == "take_profit"
    assert payload["entry_price"] == 100.0
    assert payload["exit_price"] == 105.0


def test_detect_close_reports_realized_pnl_loss(monkeypatch, tmp_path) -> None:
    from integrations.broker_base import ExitFill
    from workflows.intraday_loop import IntradayLoop

    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    note = _CapturingNotifier()
    loop.notifier = note  # type: ignore[assignment]

    loop._tracked_positions = {
        "MSFT": {"entry_price": 400.0, "quantity": 5, "direction": "long"}
    }
    broker = _FakeExitBroker(fills={
        "MSFT": ExitFill(
            symbol="MSFT", price=396.0, quantity=5, side="sell", exit_kind="stop_loss"
        )
    })
    loop._detect_and_notify_closes(broker, {"positions": [], "orders": []})

    assert len(note.calls) == 1
    _, payload = note.calls[0]
    assert payload["net_pnl"] == -20.0
    assert payload["result"] == "LOSS"
    assert payload["exit_reason"] == "stop_loss"


def test_detect_close_still_open_no_notification(monkeypatch, tmp_path) -> None:
    from workflows.intraday_loop import IntradayLoop

    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    note = _CapturingNotifier()
    loop.notifier = note  # type: ignore[assignment]

    loop._tracked_positions = {
        "AAPL": {"entry_price": 100.0, "quantity": 10, "direction": "long"}
    }
    recon = {"positions": [
        {"symbol": "AAPL", "average_price": 100.0, "quantity": 10, "direction": "long"}
    ], "orders": []}
    loop._detect_and_notify_closes(broker=_FakeExitBroker(), reconcile=recon)
    assert note.calls == []  # still open → silent


def test_detect_close_without_fill_still_notifies(monkeypatch, tmp_path) -> None:
    from workflows.intraday_loop import IntradayLoop

    settings = _settings(monkeypatch, tmp_path)
    loop = IntradayLoop(settings, dry_run=False)
    note = _CapturingNotifier()
    loop.notifier = note  # type: ignore[assignment]

    loop._tracked_positions = {
        "AAPL": {"entry_price": 100.0, "quantity": 10, "direction": "long"}
    }
    # Broker returns no exit fill → we can't compute P&L but must still alert.
    loop._detect_and_notify_closes(broker=_FakeExitBroker(), reconcile={"positions": [], "orders": []})
    assert len(note.calls) == 1
    kind, payload = note.calls[0]
    assert kind == "trade.closed"
    assert "net_pnl" not in payload  # no dollar figure without a fill


def test_live_mode_scan_refused(monkeypatch, tmp_path) -> None:
    settings = _settings(monkeypatch, tmp_path, WORKFLOW_EXECUTION_MODE="LIVE")
    init_db()
    from workflows.intraday_loop import IntradayLoop

    # dry_run=False so execution_mode resolves to LIVE in scan_once.
    loop = IntradayLoop(settings, dry_run=False)
    summary = loop.scan_once(now=datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc))
    assert summary.get("skipped") == "live_refused"
