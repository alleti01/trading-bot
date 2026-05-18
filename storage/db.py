"""SQLAlchemy engine + session helpers.

Day 1 uses sync SQLAlchemy on SQLite. Switching to Postgres later is just a
DATABASE_URL change; switching to async only matters once we put DB calls on
a hot path (which Day 1–7 don't).
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from config.settings import get_settings
from storage.tables import Base

_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker[Session]] = None


def _make_engine(database_url: str) -> Engine:
    connect_args: dict = {}
    if database_url.startswith("sqlite"):
        # SQLite + multi-thread access (APScheduler workers) needs this off.
        connect_args["check_same_thread"] = False
        # Make sure the parent directory exists for file-backed SQLite.
        if database_url.startswith("sqlite:///"):
            raw = database_url.removeprefix("sqlite:///")
            if raw and raw != ":memory:":
                Path(raw).parent.mkdir(parents=True, exist_ok=True)

    return create_engine(
        database_url,
        future=True,
        connect_args=connect_args,
    )


def init_engine() -> Engine:
    """Initialize the global engine + session factory (idempotent)."""
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        _engine = _make_engine(settings.DATABASE_URL)
        _SessionLocal = sessionmaker(
            bind=_engine,
            autoflush=False,
            expire_on_commit=False,
            future=True,
        )
    return _engine


def init_db() -> None:
    """Create all tables. Safe to call repeatedly."""
    engine = init_engine()
    Base.metadata.create_all(bind=engine)


def get_engine() -> Engine:
    return _engine if _engine is not None else init_engine()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context. Commits on success, rolls back on error."""
    if _SessionLocal is None:
        init_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_engine_for_tests() -> None:
    """Drop the cached engine so tests can rebuild against a fresh URL."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
