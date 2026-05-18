"""End-to-end tests for ``MODE=TRAIN`` (real CSV training).

Covers:

1. Valid CSV → model saved under
   ``data/models/<name>/<version>/`` with rich metadata.
2. Missing CSV path → exit code 7, no model written.
3. CSV exists but is too short → exit code 4.
4. CSV that produces too few setups → exit code 4.
5. CSV that produces only-positive (or only-negative) labels → exit 4.
6. Split is **strictly chronological** — train end ≤ val start ≤ test
   start.
7. Re-running the runner does not shuffle: split ranges are stable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests.fixtures.synthetic import synthetic_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings
    from storage.db import init_db

    defaults = {
        "MODE": "TRAIN",
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


def _write_csv(path: Path, *, n_bars: int, seed: int = 42, tz: str = "UTC") -> Path:
    """Write a synthetic OHLCV CSV in the schema MODE=TRAIN expects."""
    df = synthetic_ohlcv(n_bars=n_bars, seed=seed, tz=tz)
    out = df.reset_index().rename(columns={"timestamp": "timestamp"})
    # ``synthetic_ohlcv`` already emits a tz-aware index; format as ISO so
    # the loader's UTC fallback isn't relied on.
    out["timestamp"] = out["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    out = out[["timestamp", "open", "high", "low", "close", "volume"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def _run(args_list: list[str], settings) -> int:
    from app.logging_config import get_logger
    from app.main import _run_train_from_csv, parse_args

    args = parse_args(["--mode", "TRAIN", *args_list])
    log = get_logger("test.train_mode")
    return _run_train_from_csv(settings, log, args)


# ---------------------------------------------------------------------------
# 1. Valid CSV saves a model
# ---------------------------------------------------------------------------
def test_train_from_csv_saves_model(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "data.csv", n_bars=12_000)

    rc = _run(
        [
            "--train-csv", str(csv),
            "--model-name", "vwap_ema_pullback_lr",
            "--model-kind", "logreg",
        ],
        s,
    )
    assert rc == 0

    model_root = Path(s.MODELS_DIR) / "vwap_ema_pullback_lr"
    assert model_root.is_dir()
    versions = sorted(p for p in model_root.iterdir() if p.is_dir())
    assert len(versions) == 1
    version_dir = versions[0]
    assert (version_dir / "model.pkl").exists()
    metadata = json.loads((version_dir / "metadata.json").read_text())

    # Required metadata bits
    assert metadata["source"] == "train_from_csv"
    assert metadata["strategy"] == "vwap_ema_pullback"
    assert metadata["model_kind"] == "logreg"
    assert metadata["csv_path"].endswith("data.csv")
    assert metadata["n_train"] > 0
    assert metadata["n_val"] > 0
    assert metadata["n_test"] > 0
    assert "split_ranges" in metadata
    for key in ("train", "val", "test"):
        assert key in metadata["split_ranges"]
    assert "label_distribution" in metadata
    assert "test_metrics" in metadata


# ---------------------------------------------------------------------------
# 2. Missing CSV is refused
# ---------------------------------------------------------------------------
def test_missing_csv_fails_with_code_7(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    rc = _run(
        [
            "--train-csv", str(tmp_path / "does_not_exist.csv"),
            "--model-name", "anything",
        ],
        s,
    )
    assert rc == 7
    assert not (Path(s.MODELS_DIR) / "anything").exists()


def test_missing_train_csv_arg_fails(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    rc = _run(["--model-name", "anything"], s)
    assert rc == 4
    assert not (Path(s.MODELS_DIR) / "anything").exists()


def test_missing_model_name_fails(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "data.csv", n_bars=12_000)
    rc = _run(["--train-csv", str(csv)], s)
    assert rc == 4


# ---------------------------------------------------------------------------
# 3. CSV too short
# ---------------------------------------------------------------------------
def test_short_csv_refused(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "tiny.csv", n_bars=80)  # below MIN_OHLCV_ROWS
    rc = _run(
        ["--train-csv", str(csv), "--model-name", "tiny_model"],
        s,
    )
    assert rc == 4
    assert not (Path(s.MODELS_DIR) / "tiny_model").exists()


# ---------------------------------------------------------------------------
# 4. Too few setups
# ---------------------------------------------------------------------------
def test_too_few_setups_refused(tmp_path: Path) -> None:
    """A CSV with enough OHLCV rows for warmup but where the strategy
    doesn't fire often enough to clear the 100-setup floor must exit
    cleanly with code 4 rather than train on noise."""
    s = _settings(tmp_path)
    # 800 bars of synthetic data is well above warmup but way below the
    # 12_000 used in the success path. The VWAPEMAPullback strategy
    # produces only a handful of setups in this window.
    csv = _write_csv(tmp_path / "few_setups.csv", n_bars=800)
    rc = _run(
        [
            "--train-csv", str(csv),
            "--model-name", "few_setups_model",
        ],
        s,
    )
    assert rc == 4
    assert not (Path(s.MODELS_DIR) / "few_setups_model").exists()


# ---------------------------------------------------------------------------
# 5. Single-class labels
# ---------------------------------------------------------------------------
def test_single_class_labels_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every setup to label==1 by stubbing ``label_setups`` so we
    can verify the runner refuses to train on a degenerate dataset."""
    from labeling.tp_sl_labeler import LabelResult

    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "data.csv", n_bars=12_000)

    def _all_positive(setups, ohlcv_df, max_hold_bars):
        return [
            LabelResult(
                label=1,
                exit_reason="tp",
                exit_bar=1,
                exit_price=float(setup.target_price),
                bars_held=1,
            )
            for setup in setups
        ]

    # The runner imports ``label_setups`` lazily inside the function via
    # ``from labeling.tp_sl_labeler import label_setups``, so patching
    # the source module is what intercepts the call.
    monkeypatch.setattr(
        "labeling.tp_sl_labeler.label_setups", _all_positive, raising=True
    )

    rc = _run(
        [
            "--train-csv", str(csv),
            "--model-name", "single_class_model",
        ],
        s,
    )
    assert rc == 4
    assert not (Path(s.MODELS_DIR) / "single_class_model").exists()


# ---------------------------------------------------------------------------
# 6. Chronological split — never random
# ---------------------------------------------------------------------------
def test_split_is_chronological(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "data.csv", n_bars=12_000)

    rc = _run(
        ["--train-csv", str(csv), "--model-name", "chronotest"],
        s,
    )
    assert rc == 0

    version_dir = next((Path(s.MODELS_DIR) / "chronotest").iterdir())
    metadata = json.loads((version_dir / "metadata.json").read_text())
    train_start, train_end = metadata["split_ranges"]["train"]
    val_start, val_end = metadata["split_ranges"]["val"]
    test_start, test_end = metadata["split_ranges"]["test"]

    assert train_start <= train_end
    assert val_start <= val_end
    assert test_start <= test_end
    # Strict ordering between slices.
    assert train_end <= val_start, "val must start at or after train_end"
    assert val_end <= test_start, "test must start at or after val_end"


def test_split_is_deterministic_not_shuffled(tmp_path: Path) -> None:
    """Two runs on the same CSV must produce identical split ranges.
    A random shuffle would break this immediately."""
    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "data.csv", n_bars=12_000)

    rc1 = _run(["--train-csv", str(csv), "--model-name", "det_a"], s)
    rc2 = _run(["--train-csv", str(csv), "--model-name", "det_b"], s)
    assert rc1 == 0 and rc2 == 0

    md_a = json.loads(
        (next((Path(s.MODELS_DIR) / "det_a").iterdir()) / "metadata.json").read_text()
    )
    md_b = json.loads(
        (next((Path(s.MODELS_DIR) / "det_b").iterdir()) / "metadata.json").read_text()
    )

    assert md_a["split_ranges"] == md_b["split_ranges"]
    assert md_a["n_train"] == md_b["n_train"]
    assert md_a["n_val"] == md_b["n_val"]
    assert md_a["n_test"] == md_b["n_test"]


# ---------------------------------------------------------------------------
# 7. Bad split fractions are rejected
# ---------------------------------------------------------------------------
def test_bad_split_fractions_refused(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    csv = _write_csv(tmp_path / "data.csv", n_bars=12_000)
    rc = _run(
        [
            "--train-csv", str(csv),
            "--model-name", "bad_frac_model",
            "--train-frac", "0.9",
            "--val-frac", "0.2",  # 0.9 + 0.2 = 1.1 — invalid
        ],
        s,
    )
    assert rc == 4
    assert not (Path(s.MODELS_DIR) / "bad_frac_model").exists()
