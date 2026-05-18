"""SQLAlchemy ORM tables.

Naming note: the ML side of the codebase lives in ``models/`` (predictors,
trainers). To avoid an ambiguous ``from models import ...`` we keep ORM
classes in ``storage.tables`` rather than ``storage.models``.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """Declarative base for all tables."""


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
class Candle(Base):
    __tablename__ = "candles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    timeframe: Mapped[str] = mapped_column(String(16))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)

    __table_args__ = (
        Index(
            "ix_candle_instrument_tf_ts",
            "instrument",
            "timeframe",
            "ts",
            unique=True,
        ),
    )


class FeatureSnapshot(Base):
    """Frozen feature vector at the moment a setup fires.

    Persisting this prevents the most common lookahead bug: recomputing
    features later from raw bars and accidentally including a future bar.
    """

    __tablename__ = "feature_snapshots"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    features: Mapped[dict] = mapped_column(JSON)


# ---------------------------------------------------------------------------
# Setups, predictions, trades
# ---------------------------------------------------------------------------
class Setup(Base):
    __tablename__ = "setups"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    strategy_name: Mapped[str] = mapped_column(String(64), index=True)
    direction: Mapped[str] = mapped_column(String(8))  # "long" | "short"
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    entry_price: Mapped[float] = mapped_column(Float)
    stop_price: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    atr_at_entry: Mapped[float] = mapped_column(Float)
    feature_snapshot_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("feature_snapshots.id"), nullable=True
    )
    label: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class ModelPrediction(Base):
    __tablename__ = "model_predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    setup_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("setups.id"), index=True
    )
    model_name: Mapped[str] = mapped_column(String(128))
    model_version: Mapped[str] = mapped_column(String(64))
    probability: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    approved: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )


class PaperTrade(Base):
    __tablename__ = "paper_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    setup_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("setups.id"), nullable=True, index=True
    )
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[float] = mapped_column(Float)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Float)
    stop_price: Mapped[float] = mapped_column(Float)
    target_price: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)


class ClosedTrade(Base):
    __tablename__ = "closed_trades"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_trade_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("paper_trades.id"), nullable=True, index=True
    )
    setup_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("setups.id"), nullable=True, index=True
    )
    instrument: Mapped[str] = mapped_column(String(32), index=True)
    direction: Mapped[str] = mapped_column(String(8))
    quantity: Mapped[float] = mapped_column(Float)
    entry_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    entry_price: Mapped[float] = mapped_column(Float)
    exit_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    exit_price: Mapped[float] = mapped_column(Float)
    exit_reason: Mapped[str] = mapped_column(String(32))  # tp, sl, time, flat, manual
    pnl: Mapped[float] = mapped_column(Float)
    commission: Mapped[float] = mapped_column(Float, default=0.0)
    slippage: Mapped[float] = mapped_column(Float, default=0.0)


# ---------------------------------------------------------------------------
# Daily metrics, risk blocks
# ---------------------------------------------------------------------------
class DailyMetric(Base):
    __tablename__ = "daily_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String(10), unique=True, index=True)  # YYYY-MM-DD
    trades: Mapped[int] = mapped_column(Integer, default=0)
    wins: Mapped[int] = mapped_column(Integer, default=0)
    losses: Mapped[int] = mapped_column(Integer, default=0)
    gross_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    net_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    max_drawdown: Mapped[float] = mapped_column(Float, default=0.0)
    risk_blocks: Mapped[int] = mapped_column(Integer, default=0)


class RiskBlock(Base):
    __tablename__ = "risk_blocks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    setup_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("setups.id"), nullable=True
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )
    rule: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(Text)


# ---------------------------------------------------------------------------
# Notifications, agents
# ---------------------------------------------------------------------------
class Notification(Base):
    __tablename__ = "notifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )
    channel: Mapped[str] = mapped_column(String(32))  # discord | log | ...
    kind: Mapped[str] = mapped_column(String(64))    # bot.start, signal, trade, ...
    payload: Mapped[dict] = mapped_column(JSON)
    delivered: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class AgentOutput(Base):
    __tablename__ = "agent_outputs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, index=True
    )
    agent_name: Mapped[str] = mapped_column(String(64), index=True)
    schema_valid: Mapped[bool] = mapped_column(Boolean)
    payload: Mapped[dict] = mapped_column(JSON)
    raw_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Model registry & kill switch
# ---------------------------------------------------------------------------
class ModelMetadata(Base):
    __tablename__ = "model_metadata"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    version: Mapped[str] = mapped_column(String(64))
    trained_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    data_start: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    data_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    features: Mapped[list] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(64))
    artifact_path: Mapped[str] = mapped_column(Text)


class KillSwitchState(Base):
    """Persisted kill switch.

    A single row (id=1) represents the global state. Once tripped it stays
    tripped across restarts until manually cleared, so a crash loop cannot
    silently re-enable trading.
    """

    __tablename__ = "kill_switch_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    tripped: Mapped[bool] = mapped_column(Boolean, default=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    tripped_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
