"""Scheduler/loop respects trading window — no entries outside it."""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config.settings import reload_settings
from scheduler.market_hours import (
    is_force_flat_due,
    is_in_trading_window,
    is_trading_day,
    next_bar_boundary,
)


NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _settings(**overrides):
    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "TRADING_WINDOW_START": "09:30",
        "TRADING_WINDOW_END": "15:55",
        "FORCE_FLAT_TIME": "15:55",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    return reload_settings()


def test_inside_window_returns_true() -> None:
    s = _settings()
    now = datetime(2024, 1, 15, 14, 0, tzinfo=NY)  # Mon 2pm NY
    assert is_in_trading_window(now, s) is True
    assert is_trading_day(now, s) is True


def test_before_window_returns_false() -> None:
    s = _settings()
    now = datetime(2024, 1, 15, 8, 0, tzinfo=NY)
    assert is_in_trading_window(now, s) is False


def test_after_window_returns_false() -> None:
    s = _settings()
    now = datetime(2024, 1, 15, 16, 30, tzinfo=NY)
    assert is_in_trading_window(now, s) is False


def test_weekend_for_futures_returns_false() -> None:
    s = _settings()
    now = datetime(2024, 1, 13, 14, 0, tzinfo=NY)  # Saturday
    assert is_trading_day(now, s) is False
    assert is_in_trading_window(now, s) is False


def test_weekend_for_crypto_still_trades_in_window() -> None:
    s = _settings(MARKET_TYPE="crypto")
    now = datetime(2024, 1, 13, 14, 0, tzinfo=NY)  # Saturday
    assert is_trading_day(now, s) is True
    assert is_in_trading_window(now, s) is True


def test_force_flat_due_at_or_after_flat_time() -> None:
    s = _settings(FORCE_FLAT_TIME="15:55")
    assert is_force_flat_due(datetime(2024, 1, 15, 15, 55, tzinfo=NY), s) is True
    assert is_force_flat_due(datetime(2024, 1, 15, 16, 0, tzinfo=NY), s) is True
    assert is_force_flat_due(datetime(2024, 1, 15, 15, 54, tzinfo=NY), s) is False


def test_utc_input_is_converted_to_local_for_window_check() -> None:
    s = _settings()
    # 14:30 NY = 19:30 UTC in winter (EST = UTC-5).
    utc_now = datetime(2024, 1, 15, 19, 30, tzinfo=UTC)
    assert is_in_trading_window(utc_now, s) is True


def test_next_bar_boundary_rounds_up_to_minute() -> None:
    now = datetime(2024, 1, 15, 14, 30, 25, tzinfo=NY)
    nxt = next_bar_boundary(now, timeframe_minutes=1)
    assert nxt.minute == 31
    assert nxt.second == 0


def test_next_bar_boundary_5min() -> None:
    now = datetime(2024, 1, 15, 14, 32, 0, tzinfo=NY)
    nxt = next_bar_boundary(now, timeframe_minutes=5)
    assert nxt.minute == 35


# ---------------------------------------------------------------------------
# Loop respects window: outside-window cycle must produce zero entries.
# ---------------------------------------------------------------------------
def test_loop_outside_window_emits_no_setups(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: a loop tick outside the trading window opens nothing."""
    from data.market_data_service import SyntheticLiveFeed
    from notifications.notification_service import NotificationService
    from paper.loop import build_paper_loop
    from storage.db import init_db

    init_db()
    s = _settings(TRADING_WINDOW_START="09:30", TRADING_WINDOW_END="15:55")
    feed = SyntheticLiveFeed(
        instrument="MES", timeframe="1m", tz="America/New_York",
        max_bars=10, window_bars=5,
    )
    notifier = NotificationService(discord=None)
    loop = build_paper_loop(settings=s, feed=feed, notifier=notifier)

    # Force "now" outside the window (Sat 03:00 NY).
    outside = datetime(2024, 1, 13, 3, 0, tzinfo=NY)
    res = loop.on_bar_close(outside)
    assert res.in_window is False
    assert res.setups_filled == 0
    assert res.setups_risk_blocked == 0
