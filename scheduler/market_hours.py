"""Market session helpers used by the scheduler and the paper loop.

Day 5 keeps the calendar simple:

- Futures: trade Mon–Fri inside ``[TRADING_WINDOW_START, TRADING_WINDOW_END]``
  in the configured timezone. Holidays are out of scope for the MVP — a
  proper exchange holiday calendar slots in here later without changing
  callers.
- Crypto: trade 24/7. We still respect the configured trading window, so
  an operator who sets a tight window in ``.env`` is honored (e.g.
  "only trade 18:00–02:00").

All comparisons are done in the local timezone; the inputs may be
tz-aware UTC.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo


class _SettingsLike(Protocol):
    TIMEZONE: str
    MARKET_TYPE: str

    def trading_window_start_time(self) -> time: ...
    def trading_window_end_time(self) -> time: ...
    def force_flat_time(self) -> time: ...


def _local(now: datetime, tz: ZoneInfo) -> datetime:
    if now.tzinfo is None:
        # Defensive: do not silently treat naive as local. Treat as UTC.
        now = now.replace(tzinfo=ZoneInfo("UTC"))
    return now.astimezone(tz)


def is_trading_day(now: datetime, settings: _SettingsLike) -> bool:
    """Return True if ``now``'s local date is a tradeable session day.

    Crypto is 24/7. Futures trade Mon–Fri only.
    """
    tz = ZoneInfo(settings.TIMEZONE)
    local = _local(now, tz)
    if settings.MARKET_TYPE == "crypto":
        return True
    # Mon=0..Fri=4
    return local.weekday() < 5


def is_in_trading_window(now: datetime, settings: _SettingsLike) -> bool:
    """Return True if ``now`` is inside the configured trading window."""
    if not is_trading_day(now, settings):
        return False
    tz = ZoneInfo(settings.TIMEZONE)
    local_t = _local(now, tz).time()
    return (
        settings.trading_window_start_time()
        <= local_t
        <= settings.trading_window_end_time()
    )


def is_force_flat_due(now: datetime, settings: _SettingsLike) -> bool:
    """Return True iff local time is at/after the configured flat time."""
    tz = ZoneInfo(settings.TIMEZONE)
    local_t = _local(now, tz).time()
    return local_t >= settings.force_flat_time()


def next_bar_boundary(now: datetime, *, timeframe_minutes: int = 1) -> datetime:
    """Round ``now`` up to the next bar boundary.

    Useful for aligning an APScheduler interval job to wall-clock minutes.
    Operates in the timezone of ``now``; if ``now`` is naive it is
    interpreted as UTC.
    """
    if timeframe_minutes < 1:
        raise ValueError("timeframe_minutes must be >= 1")
    if now.tzinfo is None:
        now = now.replace(tzinfo=ZoneInfo("UTC"))

    floor = now.replace(second=0, microsecond=0)
    minute_block = (floor.minute // timeframe_minutes) * timeframe_minutes
    aligned = floor.replace(minute=minute_block)
    return aligned + timedelta(minutes=timeframe_minutes)


def session_date(now: datetime, settings: _SettingsLike) -> date:
    """The local-tz session date for ``now``."""
    tz = ZoneInfo(settings.TIMEZONE)
    return _local(now, tz).date()
