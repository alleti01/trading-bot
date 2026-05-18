"""Build a structured :class:`PostTradeAnalysis` from a closed trade.

The analyzer is **deterministic** — it does not call the LLM. It joins
``closed_trades`` with ``setups``, ``feature_snapshots``, and
``model_predictions`` and computes a small set of derived buckets
(time-of-day, volatility regime, market regime, news risk level,
``followed_plan``, ``r_multiple``).

The output is the input for :class:`MistakeClassifier` and the
per-trade Markdown report (``reports/post_trade_report.py``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select

from analysis.types import (
    MarketRegime,
    NewsRiskLevel,
    OrderflowFeatures,
    PostTradeAnalysis,
    TimeOfDayBucket,
    TradeResult,
    VolatilityRegime,
)
from app.logging_config import get_logger
from config.instruments import get_instrument
from config.settings import Settings
from storage.db import session_scope
from storage.tables import (
    ClosedTrade as ClosedTradeRow,
    FeatureSnapshot as FeatureSnapshotRow,
    ModelPrediction as ModelPredictionRow,
    Setup as SetupRow,
)


@dataclass(frozen=True)
class _Joined:
    closed: ClosedTradeRow
    setup: Optional[SetupRow]
    snapshot: Optional[FeatureSnapshotRow]
    prediction: Optional[ModelPredictionRow]


def _bucket_time_of_day(local_dt: datetime) -> TimeOfDayBucket:
    """Map a local-tz datetime to a coarse session bucket.

    The ranges below match an RTH-style session (09:30–16:00 NY) and
    label everything outside as ``off_hours``. Crypto users will mostly
    see ``off_hours``; that's fine — buckets are advisory and pattern
    miners read them as opaque strings.
    """
    t = local_dt.time()
    if t < time(8, 0):
        return "pre_market"
    if t < time(10, 0):
        return "open"
    if t < time(11, 30):
        return "mid_morning"
    if t < time(13, 30):
        return "lunch"
    if t < time(15, 0):
        return "afternoon"
    if t < time(16, 0):
        return "close"
    if t < time(18, 0):
        return "post_market"
    return "off_hours"


def _classify_volatility(features: dict[str, float]) -> VolatilityRegime:
    """Best-effort regime read off the canonical features.

    ``volatility_regime`` is one of the persisted feature columns; the
    feature builder produces an integer code. We map low/mid/high.
    """
    val = features.get("volatility_regime")
    if val is None:
        return "unknown"
    try:
        v = int(val)
    except (TypeError, ValueError):
        return "unknown"
    if v <= 0:
        return "low"
    if v == 1:
        return "medium"
    return "high"


def _classify_market_regime(features: dict[str, float]) -> MarketRegime:
    val = features.get("trend_regime")
    if val is None:
        return "unknown"
    try:
        v = int(val)
    except (TypeError, ValueError):
        return "unknown"
    if v > 0:
        return "uptrend"
    if v < 0:
        return "downtrend"
    return "chop"


def _classify_result(net_pnl: float) -> TradeResult:
    if net_pnl > 0:
        return "win"
    if net_pnl < 0:
        return "loss"
    return "breakeven"


def _r_multiple(net_pnl: float, planned_risk_dollars: float) -> float:
    """Realized R: net PnL divided by planned $ at risk (sign preserved)."""
    if planned_risk_dollars <= 0:
        return 0.0
    return float(net_pnl) / float(planned_risk_dollars)


def _planned_risk_dollars(setup: SetupRow, point_value: float) -> float:
    """``|entry - stop| * point_value`` — what the strategy *planned* to risk."""
    return abs(float(setup.entry_price) - float(setup.stop_price)) * float(point_value)


def _planned_reward_dollars(setup: SetupRow, point_value: float) -> float:
    return abs(float(setup.target_price) - float(setup.entry_price)) * float(point_value)


def _followed_plan(
    *,
    closed: ClosedTradeRow,
    setup: Optional[SetupRow],
) -> bool:
    """Heuristic: did execution match the strategy's plan?

    A trade ``followed_plan`` if it exited via a strategy-aligned reason
    (tp / sl / time / forced_flat) and the entry price is within a few
    ticks of the planned entry. ``end_of_data`` and ``manual`` exits do
    *not* count as followed plan — they are framework-level overrides.
    """
    if closed.exit_reason in {"end_of_data", "manual"}:
        return False
    if setup is None:
        # Without a setup row we can't verify the plan; default to True
        # (the deterministic close path is still rule-based).
        return True
    planned = float(setup.entry_price)
    realized = float(closed.entry_price)
    distance = abs(planned - realized)
    # Allow up to 0.5 * |entry - stop| of slippage before flagging as
    # "did not follow plan" — same spirit as the position-sizing tolerance.
    risk_distance = abs(planned - float(setup.stop_price))
    if risk_distance <= 0:
        return True
    return distance <= 0.5 * risk_distance


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
class TradeAnalyzer:
    """Builds :class:`PostTradeAnalysis` objects from the DB."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.spec = get_instrument(settings.INSTRUMENT)
        self.tz = ZoneInfo(settings.TIMEZONE)
        self.log = get_logger("analysis.trade_analyzer")

    # ------------------------------------------------------------------
    def analyze_closed_trade(
        self,
        closed_trade_id: str,
        *,
        mfe: Optional[float] = None,
        mae: Optional[float] = None,
        news_risk_at_entry: Optional[bool] = None,
        confidence_override: Optional[float] = None,
    ) -> Optional[PostTradeAnalysis]:
        """Load + join + return a :class:`PostTradeAnalysis`.

        Returns ``None`` if the trade is not found or critical joins fail.
        Never raises — this runs on every close.
        """
        try:
            joined = self._load_joined(closed_trade_id)
        except Exception as e:
            self.log.error("analysis.load_failed", trade=closed_trade_id, error=str(e))
            return None
        if joined is None:
            self.log.warning("analysis.trade_not_found", trade=closed_trade_id)
            return None

        try:
            return self._build(joined, mfe=mfe, mae=mae,
                               news_risk_at_entry=news_risk_at_entry,
                               confidence_override=confidence_override)
        except Exception as e:
            self.log.error("analysis.build_failed", trade=closed_trade_id, error=str(e))
            return None

    # ------------------------------------------------------------------
    def _load_joined(self, closed_trade_id: str) -> Optional[_Joined]:
        with session_scope() as session:
            closed = session.execute(
                select(ClosedTradeRow).where(ClosedTradeRow.id == closed_trade_id)
            ).scalar_one_or_none()
            if closed is None:
                return None
            setup = None
            snapshot = None
            prediction = None
            if closed.setup_id:
                setup = session.execute(
                    select(SetupRow).where(SetupRow.id == closed.setup_id)
                ).scalar_one_or_none()
                if setup is not None and setup.feature_snapshot_id:
                    snapshot = session.execute(
                        select(FeatureSnapshotRow).where(
                            FeatureSnapshotRow.id == setup.feature_snapshot_id
                        )
                    ).scalar_one_or_none()
                prediction = session.execute(
                    select(ModelPredictionRow)
                    .where(ModelPredictionRow.setup_id == closed.setup_id)
                    .order_by(ModelPredictionRow.created_at.desc())
                    .limit(1)
                ).scalar_one_or_none()
            # Detach SQLAlchemy rows from the session — _build will read
            # the values without re-issuing queries.
            session.expunge_all()
            return _Joined(closed=closed, setup=setup, snapshot=snapshot, prediction=prediction)

    # ------------------------------------------------------------------
    def _build(
        self,
        joined: _Joined,
        *,
        mfe: Optional[float],
        mae: Optional[float],
        news_risk_at_entry: Optional[bool],
        confidence_override: Optional[float],
    ) -> PostTradeAnalysis:
        closed = joined.closed
        setup = joined.setup
        snapshot = joined.snapshot
        prediction = joined.prediction

        features: dict[str, float] = {}
        if snapshot is not None and isinstance(snapshot.features, dict):
            features = {k: float(v) for k, v in snapshot.features.items() if isinstance(v, (int, float))}

        # Local-time entry for the time-of-day bucket.
        entry_local = closed.entry_ts.astimezone(self.tz) if closed.entry_ts.tzinfo else closed.entry_ts
        time_bucket = _bucket_time_of_day(entry_local)
        vol_regime = _classify_volatility(features)
        market_regime = _classify_market_regime(features)

        if news_risk_at_entry is True:
            news_level: NewsRiskLevel = "high"
        elif news_risk_at_entry is False:
            news_level = "low"
        else:
            news_level = "unknown"

        result = _classify_result(float(closed.pnl))
        hold_seconds = (closed.exit_ts - closed.entry_ts).total_seconds()
        gross_pnl = float(closed.pnl) + float(closed.commission or 0.0)

        if confidence_override is not None:
            conf: Optional[float] = float(confidence_override)
        elif prediction is not None:
            conf = float(prediction.probability)
        else:
            conf = None
        threshold = float(prediction.threshold) if prediction is not None else None

        risk_dollars = (
            _planned_risk_dollars(setup, self.spec.point_value) if setup else 0.0
        )
        r_mult = _r_multiple(float(closed.pnl), risk_dollars)

        return PostTradeAnalysis(
            trade_id=str(closed.id),
            setup_id=closed.setup_id,
            instrument=closed.instrument,
            direction=closed.direction,
            strategy=setup.strategy_name if setup else None,
            entry_ts=closed.entry_ts,
            exit_ts=closed.exit_ts,
            entry_price=float(closed.entry_price),
            exit_price=float(closed.exit_price),
            stop_price=float(setup.stop_price) if setup else None,
            target_price=float(setup.target_price) if setup else None,
            result=result,
            net_pnl=float(closed.pnl),
            gross_pnl=gross_pnl,
            commission=float(closed.commission or 0.0),
            slippage=float(closed.slippage or 0.0),
            hold_seconds=float(hold_seconds),
            r_multiple=float(r_mult),
            model_confidence=conf,
            model_threshold=threshold,
            risk_approved=True,  # the trade reached close → it cleared the risk gate
            features=features,
            orderflow=OrderflowFeatures(),  # MVP: no real orderflow feed yet
            market_regime=market_regime,
            volatility_regime=vol_regime,
            time_of_day_bucket=time_bucket,
            news_risk_level=news_level,
            exit_reason=str(closed.exit_reason),
            followed_plan=_followed_plan(closed=closed, setup=setup),
            mfe=float(mfe) if mfe is not None else None,
            mae=float(mae) if mae is not None else None,
        )
