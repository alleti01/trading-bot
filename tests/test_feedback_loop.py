"""Feedback loop safety: pattern miner + suggester + promotion gates."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import select

from analysis.feedback_dataset import FeedbackDataset
from analysis.improvement_suggester import (
    ImprovementSuggester,
    persist_candidates,
)
from analysis.pattern_miner import PatternMiner
from analysis.promotion import compare, write_comparison_report
from analysis.types import PROMOTION_GATES
from features.feature_builder import FEATURE_COLUMNS
from storage.db import init_db, session_scope
from storage.tables import (
    ClosedTrade,
    FeatureSnapshot,
    ImprovementSuggestion,
    ModelPrediction,
    Setup as SetupRow,
    TradeAnalysis,
    TradeMistakeTag,
)


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


# ---------------------------------------------------------------------------
# DB seeding helpers
# ---------------------------------------------------------------------------
def _seed_analyses(n: int, *, win_rate: float = 0.5, strategy: str = "vwap_ema_pullback") -> None:
    """Insert ``n`` minimal trade_analyses rows, alternating wins/losses
    to hit the requested win rate."""
    base = datetime(2026, 5, 18, 14, 0, tzinfo=timezone.utc)
    n_wins = int(n * win_rate)
    with session_scope() as session:
        for i in range(n):
            is_win = i < n_wins
            cid = str(uuid4())
            session.add(
                ClosedTrade(
                    id=cid,
                    paper_trade_id=None,
                    setup_id=None,
                    instrument="MES",
                    direction="long",
                    quantity=1.0,
                    entry_ts=base + timedelta(minutes=i),
                    entry_price=4500.0,
                    exit_ts=base + timedelta(minutes=i + 4),
                    exit_price=4504.0 if is_win else 4498.0,
                    exit_reason="tp" if is_win else "sl",
                    pnl=20.0 if is_win else -10.0,
                    commission=0.5,
                    slippage=0.0,
                )
            )
            ta = TradeAnalysis(
                closed_trade_id=cid,
                setup_id=None,
                instrument="MES",
                strategy_name=strategy,
                direction="long",
                entry_ts=base + timedelta(minutes=i),
                exit_ts=base + timedelta(minutes=i + 4),
                result="win" if is_win else "loss",
                net_pnl=20.0 if is_win else -10.0,
                r_multiple=2.0 if is_win else -1.0,
                model_confidence=0.7,
                risk_approved=True,
                followed_plan=True,
                exit_reason="tp" if is_win else "sl",
                time_of_day_bucket="afternoon",
                volatility_regime="medium",
                market_regime="uptrend",
                news_risk_level="low",
                analysis={"result": "win" if is_win else "loss"},
            )
            session.add(ta)
            session.flush()
            if not is_win:
                session.add(
                    TradeMistakeTag(
                        trade_analysis_id=ta.id,
                        closed_trade_id=cid,
                        tag="false_positive" if i % 2 == 0 else "stop_too_tight",
                        detail="seed",
                    )
                )


# ---------------------------------------------------------------------------
# PatternMiner
# ---------------------------------------------------------------------------
def test_pattern_miner_groups_by_strategy(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_analyses(10, win_rate=0.4)
    res = PatternMiner().aggregate(instrument="MES")
    assert res.n_total == 10
    s = res.by_strategy.get("vwap_ema_pullback")
    assert s is not None
    assert s.n_trades == 10
    assert s.win_rate == pytest.approx(0.4)


def test_pattern_miner_computes_false_positive_rate(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_analyses(10, win_rate=0.4)
    res = PatternMiner().aggregate(instrument="MES")
    fp = res.false_positive_rates_by_strategy.get("vwap_ema_pullback")
    assert fp is not None
    # 6 losses, 3 of them tagged false_positive (every other loss in seeder).
    assert fp > 0


def test_pattern_miner_top_tags_sorted(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_analyses(10, win_rate=0.4)
    res = PatternMiner().aggregate(instrument="MES")
    assert res.top_mistake_tags  # non-empty
    counts = [n for _, n in res.top_mistake_tags]
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------------------------------
# ImprovementSuggester — proposed only
# ---------------------------------------------------------------------------
def test_suggestions_are_only_proposed(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_analyses(20, win_rate=0.2)  # poor win rate → suggestions likely
    res = PatternMiner().aggregate(instrument="MES")
    candidates = ImprovementSuggester().propose(res)
    assert all(c.validation_status == "proposed" for c in candidates)


def test_suggestions_persist_with_proposed_status(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_analyses(20, win_rate=0.2)
    res = PatternMiner().aggregate(instrument="MES")
    candidates = ImprovementSuggester().propose(res)
    persist_candidates(candidates)

    with session_scope() as session:
        rows = session.execute(select(ImprovementSuggestion)).scalars().all()
    assert all(r.validation_status == "proposed" for r in rows)


def test_suggester_skips_thin_samples(tmp_path: Path) -> None:
    """Below MIN_TRADES_PER_BUCKET we must not propose anything (no noise-driven recs)."""
    _settings(tmp_path)
    _seed_analyses(3, win_rate=0.0)
    res = PatternMiner().aggregate(instrument="MES")
    candidates = ImprovementSuggester().propose(res)
    assert candidates == []


# ---------------------------------------------------------------------------
# FeedbackDataset
# ---------------------------------------------------------------------------
def test_feedback_dataset_export_csv(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    _seed_analyses(5, win_rate=0.6)

    dataset = FeedbackDataset()
    rows = dataset.build()
    assert len(rows) == 5

    csv_path = dataset.export_csv(rows, tmp_path / "fb.csv")
    assert csv_path.exists()
    contents = csv_path.read_text()
    assert "closed_trade_id" in contents
    # Five trades = 5 data rows + header.
    assert contents.count("\n") >= 6


def test_feedback_dataset_dataframe_has_feature_columns(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_analyses(3, win_rate=0.66)
    dataset = FeedbackDataset()
    rows = dataset.build()
    df = dataset.to_dataframe(rows)
    for col in FEATURE_COLUMNS:
        assert col in df.columns


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------
def _metadata(name: str, version: str, **metrics) -> dict:
    return {
        "name": name,
        "version": version,
        "metrics": dict(metrics),
        "fold_metrics": [
            {"fold_id": 0, "roc_auc": 0.65},
            {"fold_id": 1, "roc_auc": 0.66},
        ],
    }


def test_promotion_refuses_when_candidate_worse() -> None:
    incumbent = _metadata("m", "v1", roc_auc=0.70)
    candidate = _metadata("m", "v2", roc_auc=0.60)
    decision = compare(
        incumbent_metadata=incumbent,
        candidate_metadata=candidate,
        realized_metrics_incumbent={
            "expectancy_per_trade": 5.0,
            "profit_factor": 1.3,
            "max_drawdown_pct": 0.10,
            "false_positive_rate": 0.20,
        },
        realized_metrics_candidate={
            "expectancy_per_trade": 3.0,  # lower → fail
            "profit_factor": 1.1,         # lower → fail
            "max_drawdown_pct": 0.12,     # higher → fail
            "false_positive_rate": 0.30,  # higher → fail
        },
    )
    assert decision.promote is False
    assert decision.failed_gates  # at least one gate fired


def test_promotion_promotes_when_candidate_better() -> None:
    incumbent = _metadata("m", "v1", roc_auc=0.65)
    candidate = _metadata("m", "v2", roc_auc=0.70)
    decision = compare(
        incumbent_metadata=incumbent,
        candidate_metadata=candidate,
        realized_metrics_incumbent={
            "expectancy_per_trade": 4.0,
            "profit_factor": 1.2,
            "max_drawdown_pct": 0.15,
            "false_positive_rate": 0.30,
        },
        realized_metrics_candidate={
            "expectancy_per_trade": 6.0,
            "profit_factor": 1.4,
            "max_drawdown_pct": 0.12,
            "false_positive_rate": 0.20,
        },
    )
    assert decision.promote is True
    assert decision.failed_gates == []


def test_promotion_blocks_when_walk_forward_variance_grows() -> None:
    incumbent = _metadata("m", "v1")
    incumbent["fold_metrics"] = [
        {"fold_id": 0, "roc_auc": 0.65},
        {"fold_id": 1, "roc_auc": 0.66},
    ]
    candidate = _metadata("m", "v2")
    candidate["fold_metrics"] = [
        {"fold_id": 0, "roc_auc": 0.50},
        {"fold_id": 1, "roc_auc": 0.85},  # huge variance
    ]
    decision = compare(
        incumbent_metadata=incumbent,
        candidate_metadata=candidate,
    )
    assert decision.promote is False
    assert "walk_forward_stability" in decision.failed_gates


def test_promotion_writes_markdown_report(tmp_path: Path) -> None:
    incumbent = _metadata("m", "v1", roc_auc=0.70)
    candidate = _metadata("m", "v2", roc_auc=0.60)
    decision = compare(
        incumbent_metadata=incumbent,
        candidate_metadata=candidate,
    )
    path = write_comparison_report(decision, tmp_path)
    assert path.exists()
    assert "Promotion comparison" in path.read_text()
    assert "PROMOTE" in path.read_text() or "HOLD" in path.read_text()
