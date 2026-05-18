"""Aggregate trade analyses to surface recurring patterns.

Reads ``trade_analyses`` + ``trade_mistake_tags`` from the DB and
produces deterministic, JSON-friendly summaries:

- losing-trade counts and win rate per axis (strategy, time bucket,
  volatility regime, confidence bucket, market session, mistake tag)
- false-positive rate by setup (strategy) and by feature condition
- expectancy + win rate after each mistake tag

The miner is **read-only** and never modifies tags or strategies. The
:class:`ImprovementSuggester` is the sole consumer that turns these
stats into proposed changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import select

from analysis.types import MistakeTag
from app.logging_config import get_logger
from storage.db import session_scope
from storage.tables import TradeAnalysis as TradeAnalysisRow
from storage.tables import TradeMistakeTag as TradeMistakeTagRow


# Canonical confidence buckets — kept tight (5) so each bucket has a
# meaningful sample even on quiet weeks.
CONFIDENCE_BUCKETS: tuple[tuple[float, float, str], ...] = (
    (0.00, 0.55, "<0.55"),
    (0.55, 0.65, "0.55-0.65"),
    (0.65, 0.75, "0.65-0.75"),
    (0.75, 0.85, "0.75-0.85"),
    (0.85, 1.01, "0.85+"),
)


def _confidence_bucket(p: Optional[float]) -> str:
    if p is None:
        return "no_model"
    for lo, hi, label in CONFIDENCE_BUCKETS:
        if lo <= p < hi:
            return label
    return "0.85+"


# ---------------------------------------------------------------------------
# Aggregate result types (plain dataclasses for easy JSON serialization).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupStats:
    n_trades: int
    n_wins: int
    n_losses: int
    win_rate: float
    expectancy: float
    avg_pnl: float
    false_positive_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "n_trades": self.n_trades,
            "n_wins": self.n_wins,
            "n_losses": self.n_losses,
            "win_rate": self.win_rate,
            "expectancy": self.expectancy,
            "avg_pnl": self.avg_pnl,
            "false_positive_rate": self.false_positive_rate,
        }


@dataclass
class PatternMinerResult:
    n_total: int
    by_strategy: dict[str, GroupStats] = field(default_factory=dict)
    by_time_of_day: dict[str, GroupStats] = field(default_factory=dict)
    by_volatility_regime: dict[str, GroupStats] = field(default_factory=dict)
    by_confidence_bucket: dict[str, GroupStats] = field(default_factory=dict)
    by_market_regime: dict[str, GroupStats] = field(default_factory=dict)
    by_mistake_tag: dict[str, GroupStats] = field(default_factory=dict)
    false_positive_rates_by_strategy: dict[str, float] = field(default_factory=dict)
    top_mistake_tags: list[tuple[str, int]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_total": self.n_total,
            "by_strategy": {k: v.to_dict() for k, v in self.by_strategy.items()},
            "by_time_of_day": {k: v.to_dict() for k, v in self.by_time_of_day.items()},
            "by_volatility_regime": {k: v.to_dict() for k, v in self.by_volatility_regime.items()},
            "by_confidence_bucket": {k: v.to_dict() for k, v in self.by_confidence_bucket.items()},
            "by_market_regime": {k: v.to_dict() for k, v in self.by_market_regime.items()},
            "by_mistake_tag": {k: v.to_dict() for k, v in self.by_mistake_tag.items()},
            "false_positive_rates_by_strategy": dict(self.false_positive_rates_by_strategy),
            "top_mistake_tags": list(self.top_mistake_tags),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
@dataclass
class _RowSummary:
    strategy: Optional[str]
    time_bucket: Optional[str]
    vol_regime: Optional[str]
    market_regime: Optional[str]
    confidence_bucket: str
    net_pnl: float
    is_win: bool
    is_loss: bool
    is_false_positive: bool
    mistake_tags: list[str]


def _build_group_stats(
    rows: list[_RowSummary],
    *,
    key_fn,
) -> dict[str, GroupStats]:
    groups: dict[str, list[_RowSummary]] = {}
    for r in rows:
        key = key_fn(r)
        if key is None:
            continue
        groups.setdefault(str(key), []).append(r)

    out: dict[str, GroupStats] = {}
    for key, members in groups.items():
        n = len(members)
        wins = sum(1 for m in members if m.is_win)
        losses = sum(1 for m in members if m.is_loss)
        fps = sum(1 for m in members if m.is_false_positive)
        total_pnl = sum(m.net_pnl for m in members)
        out[key] = GroupStats(
            n_trades=n,
            n_wins=wins,
            n_losses=losses,
            win_rate=(wins / n) if n else 0.0,
            expectancy=(total_pnl / n) if n else 0.0,
            avg_pnl=(total_pnl / n) if n else 0.0,
            false_positive_rate=(fps / n) if n else 0.0,
        )
    return out


def _by_mistake_tag(rows: list[_RowSummary]) -> dict[str, GroupStats]:
    """Per-tag stats: aggregate trades that *contain* the tag.

    A single trade can fall into multiple buckets when it has multiple
    tags — that's intentional for "win rate AFTER tag X" semantics.
    """
    groups: dict[str, list[_RowSummary]] = {}
    for r in rows:
        for tag in r.mistake_tags:
            groups.setdefault(tag, []).append(r)
    out: dict[str, GroupStats] = {}
    for tag, members in groups.items():
        n = len(members)
        wins = sum(1 for m in members if m.is_win)
        losses = sum(1 for m in members if m.is_loss)
        total_pnl = sum(m.net_pnl for m in members)
        fps = sum(1 for m in members if m.is_false_positive)
        out[tag] = GroupStats(
            n_trades=n,
            n_wins=wins,
            n_losses=losses,
            win_rate=(wins / n) if n else 0.0,
            expectancy=(total_pnl / n) if n else 0.0,
            avg_pnl=(total_pnl / n) if n else 0.0,
            false_positive_rate=(fps / n) if n else 0.0,
        )
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class PatternMiner:
    """Read-only batch aggregator over ``trade_analyses``."""

    def __init__(self) -> None:
        self.log = get_logger("analysis.pattern_miner")

    def aggregate(
        self,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        instrument: Optional[str] = None,
    ) -> PatternMinerResult:
        rows = self._load_rows(start=start, end=end, instrument=instrument)
        result = PatternMinerResult(n_total=len(rows))

        result.by_strategy = _build_group_stats(rows, key_fn=lambda r: r.strategy)
        result.by_time_of_day = _build_group_stats(rows, key_fn=lambda r: r.time_bucket)
        result.by_volatility_regime = _build_group_stats(
            rows, key_fn=lambda r: r.vol_regime
        )
        result.by_confidence_bucket = _build_group_stats(
            rows, key_fn=lambda r: r.confidence_bucket
        )
        result.by_market_regime = _build_group_stats(
            rows, key_fn=lambda r: r.market_regime
        )
        result.by_mistake_tag = _by_mistake_tag(rows)

        # False-positive rates per strategy (subset of by_strategy.false_positive_rate).
        result.false_positive_rates_by_strategy = {
            k: round(v.false_positive_rate, 4)
            for k, v in result.by_strategy.items()
        }

        # Top tags by count.
        tag_counts: dict[str, int] = {}
        for r in rows:
            for tag in r.mistake_tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        result.top_mistake_tags = sorted(
            tag_counts.items(), key=lambda kv: -kv[1]
        )[:20]

        self.log.info(
            "pattern_miner.aggregated",
            n=len(rows),
            n_strategies=len(result.by_strategy),
            n_tags=len(result.by_mistake_tag),
        )
        return result

    # ------------------------------------------------------------------
    def _load_rows(
        self,
        *,
        start: Optional[datetime],
        end: Optional[datetime],
        instrument: Optional[str],
    ) -> list[_RowSummary]:
        with session_scope() as session:
            stmt = select(TradeAnalysisRow)
            if start is not None:
                start_utc = start.astimezone(timezone.utc) if start.tzinfo else start.replace(tzinfo=timezone.utc)
                stmt = stmt.where(TradeAnalysisRow.entry_ts >= start_utc)
            if end is not None:
                end_utc = end.astimezone(timezone.utc) if end.tzinfo else end.replace(tzinfo=timezone.utc)
                stmt = stmt.where(TradeAnalysisRow.entry_ts < end_utc)
            if instrument is not None:
                stmt = stmt.where(TradeAnalysisRow.instrument == instrument)
            analysis_rows = list(session.execute(stmt).scalars().all())

            tag_rows = list(
                session.execute(select(TradeMistakeTagRow)).scalars().all()
            )

        tags_by_trade: dict[str, list[str]] = {}
        for tag_row in tag_rows:
            tags_by_trade.setdefault(tag_row.closed_trade_id, []).append(tag_row.tag)

        out: list[_RowSummary] = []
        fp = MistakeTag.FALSE_POSITIVE.value
        for r in analysis_rows:
            tags = tags_by_trade.get(r.closed_trade_id, [])
            out.append(
                _RowSummary(
                    strategy=r.strategy_name,
                    time_bucket=r.time_of_day_bucket,
                    vol_regime=r.volatility_regime,
                    market_regime=r.market_regime,
                    confidence_bucket=_confidence_bucket(r.model_confidence),
                    net_pnl=float(r.net_pnl or 0.0),
                    is_win=(r.result == "win"),
                    is_loss=(r.result == "loss"),
                    is_false_positive=(fp in tags),
                    mistake_tags=tags,
                )
            )
        return out
