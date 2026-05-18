"""Trade journal CSV: schema stability, date range, instrument filter."""

from __future__ import annotations

import csv
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from config.settings import reload_settings
from reports.trade_journal import (
    CSV_COLUMNS,
    export_trade_journal,
    query_trade_journal,
    write_trade_journal_for_session,
)
from storage.db import init_db, session_scope
from storage.tables import ClosedTrade


def _settings(tmp_path: Path, **overrides):
    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def _seed(*, instrument: str, base: datetime, n: int = 1, pnl: float = 20.0) -> None:
    with session_scope() as session:
        for i in range(n):
            entry_ts = base + timedelta(minutes=10 * i)
            exit_ts = entry_ts + timedelta(minutes=4)
            session.add(ClosedTrade(
                paper_trade_id=None, setup_id=f"setup-{instrument}-{i}",
                instrument=instrument, direction="long", quantity=1.0,
                entry_ts=entry_ts, entry_price=4500.0,
                exit_ts=exit_ts, exit_price=4500.0 + pnl,
                exit_reason="tp", pnl=pnl, commission=0.5, slippage=0.0,
            ))


def test_csv_columns_are_stable() -> None:
    expected = (
        "setup_id", "paper_trade_id", "instrument", "direction", "quantity",
        "entry_ts", "entry_price", "exit_ts", "exit_price", "hold_seconds",
        "exit_reason", "gross_pnl", "commission", "slippage", "net_pnl",
    )
    assert CSV_COLUMNS == expected


def test_query_filters_by_window_and_instrument(tmp_path: Path) -> None:
    _settings(tmp_path)
    today = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    yesterday = datetime(2026, 5, 17, 14, 0, tzinfo=timezone.utc)
    _seed(instrument="MES", base=today, n=2)
    _seed(instrument="MES", base=yesterday, n=3)
    _seed(instrument="MNQ", base=today, n=1)

    rows = query_trade_journal(
        start=datetime(2026, 5, 18, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc),
        instrument="MES",
    )
    assert len(rows) == 2
    assert all(r.instrument == "MES" for r in rows)


def test_query_rejects_naive_datetimes(tmp_path: Path) -> None:
    _settings(tmp_path)
    with pytest.raises(ValueError, match="timezone-aware"):
        query_trade_journal(
            start=datetime(2026, 5, 18),
            end=datetime(2026, 5, 19),
        )


def test_export_writes_header_even_with_zero_trades(tmp_path: Path) -> None:
    out = tmp_path / "empty.csv"
    export_trade_journal([], out)
    assert out.exists()
    with out.open() as f:
        header = next(csv.reader(f))
    assert header == list(CSV_COLUMNS)


def test_export_writes_one_row_per_trade(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    today = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    _seed(instrument="MES", base=today, n=3)
    rows = query_trade_journal(
        start=datetime(2026, 5, 18, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 5, 19, 0, 0, tzinfo=timezone.utc),
        instrument="MES",
    )
    out = tmp_path / "j.csv"
    export_trade_journal(rows, out)

    with out.open() as f:
        reader = csv.DictReader(f)
        data = list(reader)
    assert len(data) == 3
    assert data[0]["instrument"] == "MES"
    assert data[0]["direction"] == "long"
    assert data[0]["exit_reason"] == "tp"
    assert float(data[0]["net_pnl"]) == pytest.approx(20.0)


def test_write_trade_journal_for_session_uses_settings_dir(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    today = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    _seed(instrument="MES", base=today, n=2)

    path = write_trade_journal_for_session(
        s, now=datetime(2026, 5, 18, 21, 0, tzinfo=timezone.utc)
    )
    assert path.exists()
    assert path.parent == Path(s.REPORTS_DIR) / "journals"
    assert "trade_journal_2026-05-18_MES.csv" in str(path)
