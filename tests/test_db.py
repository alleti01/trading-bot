"""Database init + tables."""

from __future__ import annotations

from sqlalchemy import inspect

from storage.db import get_engine, init_db, session_scope
from storage.tables import KillSwitchState


def test_init_db_creates_all_tables() -> None:
    init_db()
    engine = get_engine()
    tables = set(inspect(engine).get_table_names())

    expected = {
        "candles",
        "feature_snapshots",
        "setups",
        "model_predictions",
        "paper_trades",
        "closed_trades",
        "daily_metrics",
        "risk_blocks",
        "notifications",
        "agent_outputs",
        "model_metadata",
        "kill_switch_state",
    }
    missing = expected - tables
    assert not missing, f"tables missing after init_db: {missing}"


def test_kill_switch_default_is_not_tripped() -> None:
    init_db()
    with session_scope() as s:
        s.add(KillSwitchState(id=1, tripped=False))
    with session_scope() as s:
        row = s.get(KillSwitchState, 1)
        assert row is not None
        assert row.tripped is False
