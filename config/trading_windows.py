"""Trading-window helpers.

Day 1 stub. Full implementation (with holiday calendars, half-day handling,
and crypto 24/7 mode) lands with the scheduler on Day 5.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo


def is_within_window(
    now: datetime,
    start: time,
    end: time,
    tz: ZoneInfo,
) -> bool:
    """Return True iff ``now`` (in ``tz``) is within [start, end]."""
    local = now.astimezone(tz)
    return start <= local.time() <= end


def is_after_force_flat(now: datetime, force_flat: time, tz: ZoneInfo) -> bool:
    local = now.astimezone(tz)
    return local.time() >= force_flat
