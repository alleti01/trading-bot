"""Kill switch persistence."""

from __future__ import annotations

from risk.kill_switch import KillSwitch
from storage.db import init_db


def test_default_state_is_not_tripped() -> None:
    init_db()
    ks = KillSwitch()
    assert ks.is_tripped() is False


def test_trip_persists_across_new_instances() -> None:
    init_db()
    ks1 = KillSwitch()
    ks1.trip("synthetic test reason")
    ks2 = KillSwitch()
    snap = ks2.snapshot()
    assert snap.tripped is True
    assert snap.reason == "synthetic test reason"


def test_manual_reset_clears_state() -> None:
    init_db()
    ks = KillSwitch()
    ks.trip("trip then reset")
    assert ks.is_tripped() is True
    ks.reset_manual()
    assert ks.is_tripped() is False


def test_trip_does_not_auto_reset_between_init_db_calls() -> None:
    """A crash-loop should not silently re-enable trading."""
    init_db()
    KillSwitch().trip("simulating prior crash")
    # Reinitializing the schema (a typical startup path) must not clear the row.
    init_db()
    assert KillSwitch().is_tripped() is True
