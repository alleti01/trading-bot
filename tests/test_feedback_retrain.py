"""Feedback-based candidate retraining: end-to-end safety + correctness.

Coverage targets:

1. Feedback dataset materializes one row per closed trade (with features
   when a FeatureSnapshot exists).
2. ``train_candidate_from_feedback`` raises :class:`InsufficientFeedbackError`
   below ``min_rows`` and the CLI runner exits 4 without writing a model.
3. The chronological split is strictly time-ordered — no shuffling.
4. The candidate is saved under
   ``data/models/<candidate-name>/<version>/`` with ``candidate=True``
   in metadata.
5. Promotion stays manual: ``_run_retrain_from_feedback`` never calls
   ``_run_promote_model`` and the file system shows no "current"
   pointer flip.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from sqlalchemy import select

from analysis.feedback_dataset import FeedbackDataset
from analysis.feedback_trainer import (
    CandidateTrainResult,
    InsufficientFeedbackError,
    build_feedback_dataframe,
    train_candidate_from_feedback,
)
from analysis.types import FeedbackDatasetRow
from features.feature_builder import FEATURE_COLUMNS
from storage.db import init_db, session_scope
from storage.tables import (
    ClosedTrade,
    FeatureSnapshot,
    Setup as SetupRow,
    TradeAnalysis,
    TradeMistakeTag,
)


# ---------------------------------------------------------------------------
# Settings + DB helpers
# ---------------------------------------------------------------------------
def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "MODELS_DIR": str(tmp_path / "models"),
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def _synthetic_feature_vec(rng: random.Random, *, win: bool) -> dict[str, float]:
    """Deterministic feature vector — slightly biased by ``win`` so the
    classifier can actually learn something on toy data without making
    the test trivially memorize labels."""
    base = {col: rng.gauss(0.0, 1.0) for col in FEATURE_COLUMNS}
    bias = 0.6 if win else -0.6
    base["dist_from_vwap"] += bias
    base["trend_regime"] += bias
    base["ret_5"] += 0.4 if win else -0.4
    return base


def _seed_closed_trades_with_features(
    n: int,
    *,
    win_rate: float = 0.55,
    instrument: str = "MES",
    strategy: str = "vwap_ema_pullback",
    base_ts: datetime | None = None,
    seed: int = 0,
    tag_every_loss: bool = False,
) -> list[str]:
    """Seed ``n`` closed trades each backed by a Setup + FeatureSnapshot.

    Returns the list of closed_trade ids in chronological order.
    """
    rng = random.Random(seed)
    base = base_ts or datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc)
    ids: list[str] = []
    n_wins = int(round(n * win_rate))
    win_flags = [True] * n_wins + [False] * (n - n_wins)
    rng.shuffle(win_flags)

    with session_scope() as session:
        for i, win in enumerate(win_flags):
            entry_ts = base + timedelta(minutes=15 * i)
            exit_ts = entry_ts + timedelta(minutes=4)

            snapshot = FeatureSnapshot(
                id=str(uuid4()),
                instrument=instrument,
                ts=entry_ts,
                features=_synthetic_feature_vec(rng, win=win),
            )
            session.add(snapshot)
            session.flush()

            setup = SetupRow(
                id=str(uuid4()),
                instrument=instrument,
                strategy_name=strategy,
                direction="long",
                ts=entry_ts,
                entry_price=4500.0,
                stop_price=4495.0,
                target_price=4510.0,
                atr_at_entry=2.5,
                feature_snapshot_id=snapshot.id,
                label=1 if win else 0,
            )
            session.add(setup)
            session.flush()

            cid = str(uuid4())
            session.add(
                ClosedTrade(
                    id=cid,
                    paper_trade_id=None,
                    setup_id=setup.id,
                    instrument=instrument,
                    direction="long",
                    quantity=1.0,
                    entry_ts=entry_ts,
                    entry_price=4500.0,
                    exit_ts=exit_ts,
                    exit_price=4510.0 if win else 4495.0,
                    exit_reason="tp" if win else "sl",
                    pnl=20.0 if win else -10.0,
                    commission=0.5,
                    slippage=0.0,
                )
            )
            ta = TradeAnalysis(
                closed_trade_id=cid,
                setup_id=setup.id,
                instrument=instrument,
                strategy_name=strategy,
                direction="long",
                entry_ts=entry_ts,
                exit_ts=exit_ts,
                result="win" if win else "loss",
                net_pnl=20.0 if win else -10.0,
                r_multiple=2.0 if win else -1.0,
                model_confidence=0.7,
                risk_approved=True,
                followed_plan=True,
                exit_reason="tp" if win else "sl",
                time_of_day_bucket="afternoon",
                volatility_regime="medium",
                market_regime="uptrend",
                news_risk_level="low",
                analysis={"result": "win" if win else "loss"},
            )
            session.add(ta)
            session.flush()
            should_tag = (not win) and (tag_every_loss or i % 3 == 0)
            if should_tag:
                session.add(
                    TradeMistakeTag(
                        trade_analysis_id=ta.id,
                        closed_trade_id=cid,
                        tag="false_positive",
                        detail="seed",
                    )
                )
            ids.append(cid)
    return ids


def _make_synthetic_rows(n: int, *, win_rate: float = 0.55, seed: int = 0) -> list[FeedbackDatasetRow]:
    """Build feedback rows directly (no DB) — for trainer-unit tests."""
    rng = random.Random(seed)
    base = datetime(2026, 4, 1, 14, 0, tzinfo=timezone.utc)
    rows: list[FeedbackDatasetRow] = []
    n_wins = int(round(n * win_rate))
    win_flags = [True] * n_wins + [False] * (n - n_wins)
    rng.shuffle(win_flags)
    for i, win in enumerate(win_flags):
        rows.append(
            FeedbackDatasetRow(
                setup_id=str(uuid4()),
                closed_trade_id=str(uuid4()),
                entry_ts=base + timedelta(minutes=15 * i),
                instrument="MES",
                direction="long",
                strategy="vwap_ema_pullback",
                label=1 if win else 0,
                realized_pnl=20.0 if win else -10.0,
                mfe=None,
                mae=None,
                exit_reason="tp" if win else "sl",
                setup_type="vwap_ema_pullback",
                model_confidence=0.7,
                mistake_tags=["false_positive"] if (not win and i % 5 == 0) else [],
                features=_synthetic_feature_vec(rng, win=win),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 1. Feedback dataset creation from seeded closed trades
# ---------------------------------------------------------------------------
def test_feedback_dataset_builds_rows_with_features(tmp_path: Path) -> None:
    _settings(tmp_path)
    _seed_closed_trades_with_features(8, seed=1)

    dataset = FeedbackDataset()
    rows = dataset.build()

    assert len(rows) == 8
    # Every seeded row must carry a full feature vector.
    for row in rows:
        assert row.features
        for col in FEATURE_COLUMNS:
            assert col in row.features
    # Rows must come back ordered by exit_ts ascending.
    timestamps = [r.entry_ts for r in rows]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# 2. Insufficient rows blocks training
# ---------------------------------------------------------------------------
def test_trainer_raises_below_min_rows(tmp_path: Path) -> None:
    rows = _make_synthetic_rows(20, seed=2)
    with pytest.raises(InsufficientFeedbackError) as exc:
        train_candidate_from_feedback(rows, min_rows=100)
    msg = str(exc.value)
    assert "20" in msg
    assert ">= 100" in msg


def test_trainer_raises_when_all_labels_identical() -> None:
    rows = _make_synthetic_rows(120, win_rate=1.0, seed=3)
    with pytest.raises(InsufficientFeedbackError):
        train_candidate_from_feedback(rows, min_rows=10)


def test_cli_retrain_exits_nonzero_when_too_few_rows(tmp_path: Path) -> None:
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(20, seed=4)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        ["--retrain-from-feedback", "--model-name", "vwap_ema_pullback_lr"]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)

    assert rc == 4
    # Nothing should have been written into the registry.
    candidate_root = Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate"
    assert not candidate_root.exists()


# ---------------------------------------------------------------------------
# 3. Chronological split — never random
# ---------------------------------------------------------------------------
def test_chronological_split_is_time_ordered() -> None:
    rows = _make_synthetic_rows(150, seed=5)
    df = build_feedback_dataframe(rows)
    assert df.index.is_monotonic_increasing
    # Any row is later than the previous one.
    deltas = np.diff(df.index.view("i8"))
    assert np.all(deltas >= 0)


def test_train_val_test_slices_are_strictly_later() -> None:
    rows = _make_synthetic_rows(200, seed=6)
    result = train_candidate_from_feedback(rows, min_rows=100)
    train_end = result.train_range[1]
    val_start, val_end = result.val_range
    test_start = result.test_range[0]
    assert train_end <= val_start, "validation set must start at or after train_end"
    assert val_end <= test_start, "test set must start at or after val_end"


def test_split_is_deterministic_and_not_shuffled() -> None:
    """Two trainings on the same rows must produce identical split ranges.
    A random shuffle would break this."""
    rows = _make_synthetic_rows(160, seed=7)
    a = train_candidate_from_feedback(rows, min_rows=100)
    b = train_candidate_from_feedback(rows, min_rows=100)
    assert a.train_range == b.train_range
    assert a.val_range == b.val_range
    assert a.test_range == b.test_range


# ---------------------------------------------------------------------------
# 4. Candidate model saved under the candidate path
# ---------------------------------------------------------------------------
def test_cli_retrain_saves_candidate_model(tmp_path: Path) -> None:
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(140, seed=8)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        ["--retrain-from-feedback", "--model-name", "vwap_ema_pullback_lr"]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0

    candidate_root = Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate"
    assert candidate_root.is_dir()
    versions = sorted(candidate_root.iterdir())
    assert len(versions) == 1
    version_dir = versions[0]
    assert (version_dir / "model.pkl").exists()
    metadata_path = version_dir / "metadata.json"
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text())
    assert metadata["candidate"] is True
    assert metadata["source"] == "feedback"
    assert metadata["incumbent_name"] == "vwap_ema_pullback_lr"
    assert metadata["label_strategy"] == "pnl_positive"
    assert metadata["model_kind"] == "logreg"
    assert "test_metrics" in metadata
    for key in (
        "expectancy_per_trade",
        "profit_factor",
        "max_drawdown_pct",
        "false_positive_rate",
        "calibration_mae",
    ):
        assert key in metadata["realized_metrics"]


def test_candidate_model_name_override(tmp_path: Path) -> None:
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(140, seed=9)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        [
            "--retrain-from-feedback",
            "--model-name",
            "vwap_ema_pullback_lr",
            "--candidate-model-name",
            "my_custom_candidate",
        ]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0
    assert (Path(s.MODELS_DIR) / "my_custom_candidate").is_dir()


# ---------------------------------------------------------------------------
# 5. No auto-promotion
# ---------------------------------------------------------------------------
def test_retrain_does_not_promote_or_modify_incumbent(tmp_path: Path) -> None:
    """Even when the candidate looks great, the registry pointer for the
    incumbent must not move and no ``current`` pointer file may be
    created. Promotion is operator-only."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")

    incumbent_dir = (
        Path(s.MODELS_DIR) / "vwap_ema_pullback_lr" / "20260101T000000Z-aaaaaaaaaaaa"
    )
    incumbent_dir.mkdir(parents=True)
    incumbent_metadata = {
        "name": "vwap_ema_pullback_lr",
        "version": incumbent_dir.name,
        "metrics": {
            "roc_auc": 0.55,
            "pr_auc": 0.55,
            "precision_at_60": 0.50,
            "recall_at_60": 0.50,
        },
        "fold_metrics": [
            {"fold_id": 0, "roc_auc": 0.55},
            {"fold_id": 1, "roc_auc": 0.55},
        ],
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_names_used": list(FEATURE_COLUMNS),
        "model_kind": "logreg",
        "params": {},
        "calibration_table": [],
        "slice_metrics": {},
        "config_hash": "aaaaaaaaaaaa",
        "n_train": 100,
        "n_val": 20,
    }
    (incumbent_dir / "metadata.json").write_text(json.dumps(incumbent_metadata))
    # We don't need a real model.pkl for this test — load_model is only
    # called by the runner, which only reads metadata.
    import joblib
    joblib.dump({"placeholder": True}, incumbent_dir / "model.pkl")

    incumbent_mtime_before = (incumbent_dir / "metadata.json").stat().st_mtime_ns
    incumbent_listing_before = sorted(incumbent_dir.iterdir())

    _seed_closed_trades_with_features(140, seed=10)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        ["--retrain-from-feedback", "--model-name", "vwap_ema_pullback_lr"]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0

    # Incumbent files must be byte-for-byte unchanged.
    assert (incumbent_dir / "metadata.json").stat().st_mtime_ns == incumbent_mtime_before
    assert sorted(incumbent_dir.iterdir()) == incumbent_listing_before
    incumbent_after = json.loads((incumbent_dir / "metadata.json").read_text())
    assert incumbent_after == incumbent_metadata

    # No ``current`` symlink/marker created.
    assert not (Path(s.MODELS_DIR) / "vwap_ema_pullback_lr" / "current").exists()
    assert not (Path(s.MODELS_DIR) / "current").exists()

    # Comparison report must be written.
    feedback_reports = Path(s.REPORTS_DIR) / "feedback"
    md_reports = list(feedback_reports.glob("promotion_*.md"))
    assert md_reports, "Comparison report missing — operator can't review"


def test_retrain_without_incumbent_still_saves_candidate(tmp_path: Path) -> None:
    """First-time retrain (no incumbent yet): candidate must be saved
    so the operator can later promote it; no exception is raised."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(140, seed=11)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        ["--retrain-from-feedback", "--model-name", "vwap_ema_pullback_lr"]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0
    candidate_root = Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate"
    assert candidate_root.is_dir()


# ---------------------------------------------------------------------------
# 6. Mistake tags as metadata, not labels (default behavior)
# ---------------------------------------------------------------------------
def test_mistake_tags_are_metadata_by_default() -> None:
    rows = _make_synthetic_rows(150, seed=12)
    result = train_candidate_from_feedback(rows, min_rows=100)
    assert result.label_strategy == "pnl_positive"
    # At least one row in our synthetic generator carries the tag.
    assert sum(result.mistake_tag_counts.values()) >= 1


def test_mistake_tags_become_labels_when_opted_in() -> None:
    rows = _make_synthetic_rows(150, seed=13)
    result = train_candidate_from_feedback(
        rows, min_rows=100, use_mistake_tags_as_label=True
    )
    assert result.label_strategy == "mistake_tag_inverse"


# ---------------------------------------------------------------------------
# 7. Realized trade metrics on test split
# ---------------------------------------------------------------------------
def test_realized_metrics_have_expected_keys() -> None:
    rows = _make_synthetic_rows(200, win_rate=0.6, seed=14)
    result = train_candidate_from_feedback(rows, min_rows=100)
    expected = {
        "expectancy_per_trade",
        "profit_factor",
        "max_drawdown_pct",
        "false_positive_rate",
        "calibration_mae",
        "win_rate_approved",
        "all_trades_expectancy",
        "all_trades_profit_factor",
        "n_test",
        "n_approved",
    }
    assert expected.issubset(result.realized_metrics.keys())


def test_excluded_no_features_counted() -> None:
    """Rows missing features are dropped + counted, never silently
    imputed."""
    rows = _make_synthetic_rows(120, seed=15)
    # Strip features from a few rows.
    bare = []
    for i, r in enumerate(rows):
        if i % 10 == 0:
            bare.append(
                FeedbackDatasetRow(
                    setup_id=r.setup_id,
                    closed_trade_id=r.closed_trade_id,
                    entry_ts=r.entry_ts,
                    instrument=r.instrument,
                    direction=r.direction,
                    strategy=r.strategy,
                    label=r.label,
                    realized_pnl=r.realized_pnl,
                    mfe=None,
                    mae=None,
                    exit_reason=r.exit_reason,
                    setup_type=r.setup_type,
                    model_confidence=r.model_confidence,
                    mistake_tags=list(r.mistake_tags),
                    features={},
                )
            )
        else:
            bare.append(r)
    result = train_candidate_from_feedback(bare, min_rows=100)
    assert result.excluded_no_features == 12


# ---------------------------------------------------------------------------
# 8. New CLI flag spec parity (--min-feedback-rows / --use-mistake-tags-as-label / --feedback-model-kind)
# ---------------------------------------------------------------------------
def test_cli_min_feedback_rows_flag_overrides_settings(tmp_path: Path) -> None:
    """``--min-feedback-rows`` is the canonical flag and must override
    the env-derived ``FEEDBACK_MIN_ROWS``."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="500")
    _seed_closed_trades_with_features(140, seed=20)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        [
            "--retrain-from-feedback",
            "--model-name",
            "vwap_ema_pullback_lr",
            "--min-feedback-rows",
            "100",
        ]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0
    assert (Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate").is_dir()


def test_cli_legacy_feedback_min_rows_alias_still_works(tmp_path: Path) -> None:
    """The old ``--feedback-min-rows`` flag is kept as an alias so prior
    operator scripts and earlier README revisions don't break."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="500")
    _seed_closed_trades_with_features(140, seed=21)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        [
            "--retrain-from-feedback",
            "--model-name",
            "vwap_ema_pullback_lr",
            "--feedback-min-rows",
            "100",
        ]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0


def test_cli_use_mistake_tags_as_label_flag(tmp_path: Path) -> None:
    """``--use-mistake-tags-as-label`` swaps the candidate's label
    strategy and is reflected in saved metadata.

    The seeding path tags *every* loss so that under the inverse-mistake
    label mapping the dataset still has both classes after the
    chronological 70/15/15 split (otherwise calibration on a 21-row
    val slice can degenerate)."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(
        140, win_rate=0.5, seed=22, tag_every_loss=True
    )

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        [
            "--retrain-from-feedback",
            "--model-name",
            "vwap_ema_pullback_lr",
            "--use-mistake-tags-as-label",
        ]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0

    version_dir = next(
        (Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate").iterdir()
    )
    metadata = json.loads((version_dir / "metadata.json").read_text())
    assert metadata["label_strategy"] == "mistake_tag_inverse"


def test_cli_feedback_model_kind_logreg_default(tmp_path: Path) -> None:
    """The default model_kind is logreg and ends up in candidate metadata."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(140, seed=23)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(
        [
            "--retrain-from-feedback",
            "--model-name",
            "vwap_ema_pullback_lr",
            "--feedback-model-kind",
            "logreg",
        ]
    )
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 0
    version_dir = next(
        (Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate").iterdir()
    )
    metadata = json.loads((version_dir / "metadata.json").read_text())
    assert metadata["model_kind"] == "logreg"


def test_cli_retrain_requires_model_name(tmp_path: Path) -> None:
    """Without --model-name the runner refuses and exits 4."""
    s = _settings(tmp_path, FEEDBACK_MIN_ROWS="100")
    _seed_closed_trades_with_features(140, seed=24)

    from app.logging_config import get_logger
    from app.main import _run_retrain_from_feedback, parse_args

    args = parse_args(["--retrain-from-feedback"])
    log = get_logger("test")
    rc = _run_retrain_from_feedback(s, log, args=args)
    assert rc == 4
    # No candidate should have been created.
    assert not (Path(s.MODELS_DIR) / "vwap_ema_pullback_lr_candidate").exists()
