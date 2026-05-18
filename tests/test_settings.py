"""Settings & LIVE-mode lockout."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from config.settings import reload_settings


def test_defaults_load_in_paper_mode() -> None:
    s = reload_settings()
    assert s.MODE == "PAPER"
    assert s.LIVE_ADAPTER_CONFIRMED is False
    assert 0.0 <= s.CONFIDENCE_THRESHOLD <= 1.0


def test_live_mode_without_adapter_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODE", "LIVE")
    monkeypatch.setenv("LIVE_ADAPTER_CONFIRMED", "false")
    with pytest.raises(ValidationError) as ei:
        reload_settings()
    assert "LIVE mode is locked" in str(ei.value)


def test_live_mode_with_flag_loads_settings_but_executor_still_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flag alone is not enough — the placeholder executor still refuses."""
    monkeypatch.setenv("MODE", "LIVE")
    monkeypatch.setenv("LIVE_ADAPTER_CONFIRMED", "true")
    s = reload_settings()
    assert s.MODE == "LIVE"
    assert s.LIVE_ADAPTER_CONFIRMED is True

    from execution.live_executor_placeholder import LiveExecutor, LiveExecutorRefusedError

    with pytest.raises(LiveExecutorRefusedError):
        LiveExecutor()


def test_invalid_timezone_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIMEZONE", "Mars/Olympus_Mons")
    with pytest.raises(ValidationError):
        reload_settings()


def test_invalid_time_format_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_WINDOW_START", "9am")
    with pytest.raises(ValidationError):
        reload_settings()


def test_window_must_be_increasing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_WINDOW_START", "16:00")
    monkeypatch.setenv("TRADING_WINDOW_END", "09:30")
    with pytest.raises(ValidationError):
        reload_settings()


def test_confidence_threshold_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "1.5")
    with pytest.raises(ValidationError):
        reload_settings()


def test_force_flat_must_not_precede_window_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADING_WINDOW_START", "09:30")
    monkeypatch.setenv("TRADING_WINDOW_END", "15:55")
    monkeypatch.setenv("FORCE_FLAT_TIME", "08:00")
    with pytest.raises(ValidationError):
        reload_settings()
