"""Build a retraining-ready dataset from logged closed trades.

The export materializes one :class:`FeedbackDatasetRow` per closed
trade by joining ``closed_trades`` → ``setups`` →
``feature_snapshots`` → ``model_predictions`` and pulling tags from
``trade_mistake_tags``. The output is the input to
``analysis.promotion.compare_against_incumbent`` and to any future
candidate-model trainer.

The exporter is a *pure read*: it never modifies state. Promotion is
the only path that touches the model registry.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from sqlalchemy import select

from analysis.types import FeedbackDatasetRow
from app.logging_config import get_logger
from features.feature_builder import FEATURE_COLUMNS
from storage.db import session_scope
from storage.tables import (
    ClosedTrade as ClosedTradeRow,
    FeatureSnapshot as FeatureSnapshotRow,
    ModelPrediction as ModelPredictionRow,
    Setup as SetupRow,
    TradeMistakeTag as TradeMistakeTagRow,
)


# Columns written to the CSV (stable schema for downstream tooling).
CSV_COLUMNS: tuple[str, ...] = (
    "closed_trade_id",
    "setup_id",
    "instrument",
    "direction",
    "strategy",
    "entry_ts",
    "label",
    "realized_pnl",
    "model_confidence",
    "exit_reason",
    "mfe",
    "mae",
    "mistake_tags",
)


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------
class FeedbackDataset:
    """Build, export, and reload the post-trade feedback dataset."""

    def __init__(self) -> None:
        self.log = get_logger("analysis.feedback_dataset")

    # ------------------------------------------------------------------
    def build(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        instrument: Optional[str] = None,
    ) -> list[FeedbackDatasetRow]:
        """Load every applicable closed trade as a dataset row."""
        with session_scope() as session:
            stmt = select(ClosedTradeRow)
            if start is not None:
                start_utc = (
                    start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
                )
                stmt = stmt.where(ClosedTradeRow.exit_ts >= start_utc)
            if end is not None:
                end_utc = (
                    end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
                )
                stmt = stmt.where(ClosedTradeRow.exit_ts < end_utc)
            if instrument is not None:
                stmt = stmt.where(ClosedTradeRow.instrument == instrument)
            stmt = stmt.order_by(ClosedTradeRow.exit_ts.asc())
            closed_rows = list(session.execute(stmt).scalars().all())
            tag_rows = list(session.execute(select(TradeMistakeTagRow)).scalars().all())
            session.expunge_all()

            setups: dict[str, SetupRow] = {}
            snapshots: dict[str, FeatureSnapshotRow] = {}
            predictions: dict[str, ModelPredictionRow] = {}

            setup_ids = {r.setup_id for r in closed_rows if r.setup_id}
            if setup_ids:
                with session_scope() as session2:
                    setup_rows = list(
                        session2.execute(
                            select(SetupRow).where(SetupRow.id.in_(setup_ids))
                        ).scalars().all()
                    )
                    snapshot_ids = {
                        s.feature_snapshot_id for s in setup_rows if s.feature_snapshot_id
                    }
                    snapshot_rows = (
                        list(
                            session2.execute(
                                select(FeatureSnapshotRow).where(
                                    FeatureSnapshotRow.id.in_(snapshot_ids)
                                )
                            ).scalars().all()
                        )
                        if snapshot_ids
                        else []
                    )
                    prediction_rows = list(
                        session2.execute(
                            select(ModelPredictionRow).where(
                                ModelPredictionRow.setup_id.in_(setup_ids)
                            )
                        ).scalars().all()
                    )
                    session2.expunge_all()

                setups = {s.id: s for s in setup_rows}
                snapshots = {s.id: s for s in snapshot_rows}
                # Latest prediction per setup_id wins.
                for p in sorted(prediction_rows, key=lambda r: r.created_at):
                    predictions[p.setup_id] = p

        tags_by_trade: dict[str, list[str]] = {}
        for t in tag_rows:
            tags_by_trade.setdefault(t.closed_trade_id, []).append(t.tag)

        out: list[FeedbackDatasetRow] = []
        for r in closed_rows:
            setup = setups.get(r.setup_id) if r.setup_id else None
            snap = (
                snapshots.get(setup.feature_snapshot_id)
                if setup is not None and setup.feature_snapshot_id
                else None
            )
            pred = predictions.get(r.setup_id) if r.setup_id else None
            features: dict[str, float] = {}
            if snap is not None and isinstance(snap.features, dict):
                features = {
                    k: float(v)
                    for k, v in snap.features.items()
                    if isinstance(v, (int, float))
                }

            out.append(
                FeedbackDatasetRow(
                    setup_id=str(r.setup_id or ""),
                    closed_trade_id=str(r.id),
                    entry_ts=r.entry_ts,
                    instrument=r.instrument,
                    direction=r.direction,
                    strategy=setup.strategy_name if setup else None,
                    label=int(r.pnl > 0),
                    realized_pnl=float(r.pnl),
                    mfe=None,
                    mae=None,
                    exit_reason=str(r.exit_reason),
                    setup_type=setup.strategy_name if setup else None,
                    model_confidence=float(pred.probability) if pred else None,
                    mistake_tags=list(tags_by_trade.get(str(r.id), [])),
                    features=features,
                )
            )

        self.log.info("feedback_dataset.built", n=len(out))
        return out

    # ------------------------------------------------------------------
    def to_dataframe(self, rows: list[FeedbackDatasetRow]) -> pd.DataFrame:
        """Return a DataFrame with one row per trade. Features are
        expanded into individual columns matching ``FEATURE_COLUMNS``."""
        records = []
        for row in rows:
            base = {
                "closed_trade_id": row.closed_trade_id,
                "setup_id": row.setup_id,
                "instrument": row.instrument,
                "direction": row.direction,
                "strategy": row.strategy or "",
                "entry_ts": row.entry_ts,
                "label": int(row.label),
                "realized_pnl": float(row.realized_pnl),
                "exit_reason": row.exit_reason,
                "model_confidence": row.model_confidence,
                "mistake_tags": ",".join(row.mistake_tags),
            }
            for col in FEATURE_COLUMNS:
                base[col] = row.features.get(col)
            records.append(base)
        df = pd.DataFrame(records)
        if "entry_ts" in df.columns and not df.empty:
            df = df.set_index("entry_ts").sort_index()
        return df

    # ------------------------------------------------------------------
    def export_csv(self, rows: list[FeedbackDatasetRow], path: Path) -> Path:
        """Stable-schema CSV export. Always writes a header."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "closed_trade_id": row.closed_trade_id,
                        "setup_id": row.setup_id,
                        "instrument": row.instrument,
                        "direction": row.direction,
                        "strategy": row.strategy or "",
                        "entry_ts": row.entry_ts.isoformat(),
                        "label": row.label,
                        "realized_pnl": f"{row.realized_pnl:.4f}",
                        "model_confidence": (
                            f"{row.model_confidence:.4f}"
                            if row.model_confidence is not None
                            else ""
                        ),
                        "exit_reason": row.exit_reason,
                        "mfe": "" if row.mfe is None else f"{row.mfe:.4f}",
                        "mae": "" if row.mae is None else f"{row.mae:.4f}",
                        "mistake_tags": ",".join(row.mistake_tags),
                    }
                )
        return path

    def export_json(self, rows: list[FeedbackDatasetRow], path: Path) -> Path:
        """JSON export including features (CSV path keeps the schema flat)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for row in rows:
            d = asdict(row)
            d["entry_ts"] = row.entry_ts.isoformat()
            payload.append(d)
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
