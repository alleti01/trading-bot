"""Model registry round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features.feature_builder import FEATURE_COLUMNS
from models.model_registry import load_model, save_model
from models.trainer import train
from storage.db import init_db


def _train_a_tiny_model() -> "tuple":
    rng = np.random.default_rng(0)
    n, d = 200, len(FEATURE_COLUMNS)
    X = pd.DataFrame(rng.standard_normal((n, d)), columns=list(FEATURE_COLUMNS))
    y = (X.iloc[:, 0] > 0).astype(int)
    X.index = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    y.index = X.index
    return train(X.iloc[:140], y.iloc[:140], X.iloc[140:170], y.iloc[140:170], model_kind="logreg")


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    init_db()
    result = _train_a_tiny_model()
    version = save_model(result, name="round_trip", registry_path=tmp_path)
    assert version is not None and "-" in version  # YYYYMMDDTHHMMSSZ-<hash>

    loaded = load_model("round_trip", version=version, registry_path=tmp_path)
    assert loaded.metadata["name"] == "round_trip"
    assert loaded.metadata["version"] == version
    assert loaded.metadata["feature_columns"] == list(FEATURE_COLUMNS)
    assert loaded.metadata["model_kind"] == "logreg"
    # Metrics survive the round trip.
    assert loaded.metadata["metrics"]["accuracy"] == result.aggregate_metrics["accuracy"]


def test_load_latest(tmp_path: Path) -> None:
    init_db()
    r1 = _train_a_tiny_model()
    r2 = _train_a_tiny_model()
    save_model(r1, name="latest_test", registry_path=tmp_path)
    v2 = save_model(r2, name="latest_test", registry_path=tmp_path)
    loaded = load_model("latest_test", version="latest", registry_path=tmp_path)
    assert loaded.metadata["version"] == v2


def test_metadata_json_is_valid(tmp_path: Path) -> None:
    init_db()
    result = _train_a_tiny_model()
    version = save_model(result, name="json_test", registry_path=tmp_path)
    metadata_path = tmp_path / "json_test" / version / "metadata.json"
    payload = json.loads(metadata_path.read_text())
    # Required keys present.
    for key in [
        "name", "version", "trained_at", "data_start", "data_end",
        "feature_columns", "feature_names_used", "model_kind", "params",
        "metrics", "calibration_table", "config_hash", "n_train", "n_val",
    ]:
        assert key in payload, f"missing key: {key}"
