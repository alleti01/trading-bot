"""LIVE-mode dual lockout.

Two independent gates must both be cleared before the bot can ever route
a real-money order:

1. ``Settings`` refuses to load with ``MODE=LIVE`` unless
   ``LIVE_ADAPTER_CONFIRMED=true``.
2. ``LiveExecutor`` itself raises on construction — even if (1) is
   somehow satisfied — until a real adapter has been implemented.
3. The CLI dispatcher returns a non-zero exit code if MODE=LIVE reaches
   the dispatch step instead of silently no-op'ing.
"""

from __future__ import annotations

import os

import pytest

from config.settings import Settings, reload_settings
from execution.live_executor_placeholder import (
    LiveExecutor,
    LiveExecutorRefusedError,
)


def test_settings_refuses_live_without_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("MODE", "LIVE")
    monkeypatch.delenv("LIVE_ADAPTER_CONFIRMED", raising=False)
    with pytest.raises(Exception) as exc:
        Settings()
    assert "LIVE mode is locked" in str(exc.value)


def test_settings_still_loads_when_live_confirmed(monkeypatch) -> None:
    """Settings *can* be constructed with both flags set — the second
    gate (LiveExecutor) is what stops actual execution. We don't want
    Settings to refuse, otherwise the dispatcher wouldn't even get a
    chance to log its own refusal."""
    monkeypatch.setenv("MODE", "LIVE")
    monkeypatch.setenv("LIVE_ADAPTER_CONFIRMED", "true")
    s = reload_settings()
    assert s.MODE == "LIVE"
    assert s.LIVE_ADAPTER_CONFIRMED is True


def test_live_executor_refuses_construction() -> None:
    with pytest.raises(LiveExecutorRefusedError):
        LiveExecutor()


def test_main_returns_nonzero_for_live_mode(monkeypatch, tmp_path) -> None:
    """End-to-end: even with both flags set, ``main()`` must NOT silently
    succeed — it returns a non-zero exit code so operators see the
    misconfiguration in CI / shell exit status."""
    db_path = tmp_path / "live.db"
    monkeypatch.setenv("MODE", "LIVE")
    monkeypatch.setenv("LIVE_ADAPTER_CONFIRMED", "true")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from app import main as app_main

    rc = app_main.main(argv=[])
    # 0 = success/dry-run, 6 = live-mode refusal in dispatcher.
    assert rc == 6
