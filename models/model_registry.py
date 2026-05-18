"""Versioned model artifacts.

Each saved model lives at::

    <registry_path>/<name>/<version>/
        model.pkl       joblib-serialized calibrated estimator
        metadata.json   training info + FEATURE_COLUMNS snapshot

The metadata's ``feature_columns`` field is the full ``FEATURE_COLUMNS``
tuple at training time; the predictor compares it against the current
``FEATURE_COLUMNS`` to detect drift.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from app.logging_config import get_logger
from config.settings import get_settings
from features.feature_builder import FEATURE_COLUMNS
from models.trainer import TrainResult
from storage.db import session_scope
from storage.tables import ModelMetadata as ModelMetadataRow


@dataclass
class LoadedModel:
    estimator: Any
    metadata: dict[str, Any]


def _config_hash(params: dict[str, Any], feature_names: list[str], model_kind: str) -> str:
    blob = json.dumps(
        {"params": params, "features": feature_names, "model_kind": model_kind},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _make_version(config_hash: str, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y%m%dT%H%M%SZ')}-{config_hash}"


def save_model(
    train_result: TrainResult,
    *,
    name: str,
    registry_path: Path | None = None,
    extra_metadata: dict | None = None,
    write_db_row: bool = True,
) -> str:
    log = get_logger("models.registry")

    if registry_path is None:
        registry_path = Path(get_settings().MODELS_DIR)
    registry_path = Path(registry_path)

    config_hash = _config_hash(
        train_result.params, train_result.feature_names, train_result.model_kind
    )
    version = _make_version(config_hash)

    model_dir = registry_path / name / version
    model_dir.mkdir(parents=True, exist_ok=True)

    artifact_path = model_dir / "model.pkl"
    joblib.dump(train_result.estimator, artifact_path)

    metadata: dict[str, Any] = {
        "name": name,
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "data_start": train_result.train_start.isoformat(),
        "data_end": train_result.val_end.isoformat(),
        "feature_columns": list(FEATURE_COLUMNS),  # canonical snapshot
        "feature_names_used": list(train_result.feature_names),
        "model_kind": train_result.model_kind,
        "params": train_result.params,
        "metrics": train_result.aggregate_metrics,
        "calibration_table": train_result.calibration_table,
        "slice_metrics": train_result.slice_metrics,
        "config_hash": config_hash,
        "n_train": train_result.n_train,
        "n_val": train_result.n_val,
        "fold_metrics": [asdict(fm) for fm in train_result.fold_metrics],
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    if write_db_row:
        with session_scope() as session:
            row = ModelMetadataRow(
                model_name=name,
                version=version,
                trained_at=datetime.now(timezone.utc),
                data_start=train_result.train_start,
                data_end=train_result.val_end,
                features=list(FEATURE_COLUMNS),
                metrics=train_result.aggregate_metrics,
                config_hash=config_hash,
                artifact_path=str(artifact_path),
            )
            session.add(row)

    log.info(
        "model.saved",
        name=name,
        version=version,
        path=str(model_dir),
        config_hash=config_hash,
        n_train=train_result.n_train,
        n_val=train_result.n_val,
    )
    return version


def load_model(
    name: str,
    version: str = "latest",
    *,
    registry_path: Path | None = None,
) -> LoadedModel:
    if registry_path is None:
        registry_path = Path(get_settings().MODELS_DIR)
    registry_path = Path(registry_path)

    name_dir = registry_path / name
    if not name_dir.exists():
        raise FileNotFoundError(f"No models registered under {name_dir}")

    if version == "latest":
        candidates = sorted(p.name for p in name_dir.iterdir() if p.is_dir())
        if not candidates:
            raise FileNotFoundError(f"No versions found for model '{name}' under {name_dir}")
        version = candidates[-1]

    model_dir = name_dir / version
    if not model_dir.exists():
        raise FileNotFoundError(f"Model version not found: {model_dir}")

    estimator = joblib.load(model_dir / "model.pkl")
    metadata = json.loads((model_dir / "metadata.json").read_text())
    return LoadedModel(estimator=estimator, metadata=metadata)
