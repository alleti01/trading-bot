"""Trade analyzer + post-trade analysis service end-to-end."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select

from agents.llm_client import LLMClient, LLMClientError, MockLLMClient
from analysis.service import PostTradeAnalysisService
from analysis.trade_analyzer import TradeAnalyzer
from analysis.types import MistakeTag
from features.feature_builder import FEATURE_COLUMNS
from notifications.notification_service import NotificationService
from storage.db import init_db, session_scope
from storage.tables import (
    AgentOutput,
    ClosedTrade,
    FeatureSnapshot,
    ImprovementSuggestion,
    ModelPrediction,
    Setup as SetupRow,
    TradeAnalysis,
    TradeMistakeTag,
)


NY = ZoneInfo("America/New_York")


def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings

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


def _seed_trade(
    *,
    instrument: str = "MES",
    direction: str = "long",
    entry: float = 4500.0,
    exit_: float = 4504.0,
    stop: float = 4498.0,
    target: float = 4504.0,
    exit_reason: str = "tp",
    pnl: float = 20.0,
    confidence: float | None = None,
    threshold: float | None = None,
    base: datetime | None = None,
    features: dict[str, float] | None = None,
) -> str:
    """Insert one fully-joined trade and return its closed_trades.id."""
    base = base or datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    feat = {col: 0.0 for col in FEATURE_COLUMNS}
    feat.update({"volatility_regime": 1, "trend_regime": 1, "volume_ratio_20": 1.0})
    if features:
        feat.update(features)

    with session_scope() as session:
        snap = FeatureSnapshot(instrument=instrument, ts=base, features=feat)
        session.add(snap)
        session.flush()

        setup_id = str(uuid4())
        session.add(
            SetupRow(
                id=setup_id,
                instrument=instrument,
                strategy_name="vwap_ema_pullback",
                direction=direction,
                ts=base,
                entry_price=entry,
                stop_price=stop,
                target_price=target,
                atr_at_entry=1.0,
                feature_snapshot_id=snap.id,
            )
        )
        if confidence is not None and threshold is not None:
            session.add(
                ModelPrediction(
                    setup_id=setup_id,
                    model_name="test_lr",
                    model_version="t",
                    probability=confidence,
                    threshold=threshold,
                    approved=confidence >= threshold,
                )
            )
        closed_id = str(uuid4())
        session.add(
            ClosedTrade(
                id=closed_id,
                paper_trade_id=None,
                setup_id=setup_id,
                instrument=instrument,
                direction=direction,
                quantity=1.0,
                entry_ts=base,
                entry_price=entry,
                exit_ts=base + timedelta(minutes=4),
                exit_price=exit_,
                exit_reason=exit_reason,
                pnl=pnl,
                commission=0.5,
                slippage=0.0,
            )
        )
        return closed_id


# ---------------------------------------------------------------------------
# TradeAnalyzer (pure)
# ---------------------------------------------------------------------------
def test_winning_trade_has_no_tags(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(pnl=20.0, exit_reason="tp", confidence=0.72, threshold=0.60)
    analyzer = TradeAnalyzer(s)
    a = analyzer.analyze_closed_trade(cid)
    assert a is not None
    assert a.result == "win"
    assert a.r_multiple > 0
    assert a.followed_plan is True


def test_losing_trade_marked_loss(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(
        direction="long",
        entry=4500.0,
        exit_=4498.0,
        stop=4498.0,
        target=4504.0,
        exit_reason="sl",
        pnl=-10.0,
    )
    a = TradeAnalyzer(s).analyze_closed_trade(cid)
    assert a is not None
    assert a.result == "loss"
    assert a.r_multiple < 0


def test_analyzer_loads_feature_snapshot(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(features={"volume_ratio_20": 0.5, "volatility_regime": 2})
    a = TradeAnalyzer(s).analyze_closed_trade(cid)
    assert a is not None
    assert a.features.get("volume_ratio_20") == 0.5
    # vol regime int 2 → high
    assert a.volatility_regime == "high"


def test_analyzer_handles_missing_setup(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    base = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    closed_id = str(uuid4())
    with session_scope() as session:
        session.add(
            ClosedTrade(
                id=closed_id,
                paper_trade_id=None,
                setup_id=None,
                instrument="MES",
                direction="long",
                quantity=1.0,
                entry_ts=base,
                entry_price=4500.0,
                exit_ts=base + timedelta(minutes=4),
                exit_price=4500.0,
                exit_reason="manual",
                pnl=0.0,
                commission=0.0,
                slippage=0.0,
            )
        )
    a = TradeAnalyzer(s).analyze_closed_trade(closed_id)
    assert a is not None
    assert a.result == "breakeven"
    assert a.strategy is None
    assert a.followed_plan is False  # manual exit → not following plan


def test_analyzer_returns_none_for_unknown_trade(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    a = TradeAnalyzer(s).analyze_closed_trade("not-a-real-id")
    assert a is None


# ---------------------------------------------------------------------------
# PostTradeAnalysisService end-to-end
# ---------------------------------------------------------------------------
def test_service_persists_analysis_and_tags(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(
        direction="short",
        entry=4505.0,
        exit_=4509.0,
        stop=4509.0,
        target=4501.0,
        exit_reason="sl",
        pnl=-20.0,
        confidence=0.78,
        threshold=0.60,
    )
    service = PostTradeAnalysisService(
        s,
        notifier=NotificationService(discord=None),
        llm=MockLLMClient(),
    )
    outcome = service.on_trade_closed(cid, mfe=2.0, mae=20.0, news_risk_at_entry=False)

    assert outcome.persisted is True
    assert outcome.analysis is not None and outcome.analysis.result == "loss"
    assert outcome.tagging is not None
    assert MistakeTag.FALSE_POSITIVE in outcome.tagging.tags
    assert outcome.md_path is not None and outcome.md_path.exists()

    with session_scope() as session:
        analyses = session.execute(select(TradeAnalysis)).scalars().all()
        tags = session.execute(select(TradeMistakeTag)).scalars().all()
    assert len(analyses) == 1
    tag_strings = {t.tag for t in tags}
    assert "false_positive" in tag_strings


def test_service_does_not_raise_on_llm_failure(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(pnl=-5.0, exit_reason="sl")

    class _AngryLLM(LLMClient):
        def complete(self, *, system: str, user: str) -> str:
            raise LLMClientError("simulated network failure")

    service = PostTradeAnalysisService(
        s, notifier=NotificationService(discord=None), llm=_AngryLLM()
    )
    # Must not raise; outcome should still have analysis + persisted rows.
    outcome = service.on_trade_closed(cid)
    assert outcome.analysis is not None
    assert outcome.persisted is True
    # LLM failure persists an invalid AgentOutput row.
    with session_scope() as session:
        agent_rows = session.execute(select(AgentOutput)).scalars().all()
    assert any(not r.schema_valid for r in agent_rows)


def test_service_runs_without_llm(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(pnl=20.0)
    service = PostTradeAnalysisService(
        s, notifier=NotificationService(discord=None), llm=None
    )
    outcome = service.on_trade_closed(cid)
    assert outcome.analysis is not None
    assert outcome.agent_summary is None
    with session_scope() as session:
        n = session.execute(select(TradeAnalysis)).scalars().all()
    assert len(n) == 1


def test_service_notifies_with_trade_analysis_kind(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    cid = _seed_trade(pnl=20.0)

    captured: list[tuple[str, dict[str, Any]]] = []

    class _Notifier:
        def notify(self, kind: str, /, **payload: Any) -> None:
            captured.append((kind, dict(payload)))

    service = PostTradeAnalysisService(s, notifier=_Notifier(), llm=MockLLMClient())
    service.on_trade_closed(cid)

    kinds = {k for k, _ in captured}
    assert "trade.analysis" in kinds
    payload = next(p for k, p in captured if k == "trade.analysis")
    assert payload["result"] == "win"
    assert "report_path" in payload
