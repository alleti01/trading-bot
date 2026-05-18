"""Setup Pydantic contract."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from features.feature_builder import FEATURE_COLUMNS
from strategies.base import Setup


def _valid_features() -> dict[str, float]:
    return {col: 0.0 for col in FEATURE_COLUMNS}


def _valid_kwargs(**overrides) -> dict:
    base = dict(
        instrument="MES",
        timestamp=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        strategy_name="vwap_ema_pullback",
        direction="long",
        entry_price=100.0,
        stop_price=99.0,
        target_price=102.0,
        atr_at_entry=1.0,
        features=_valid_features(),
        bar_index=42,
    )
    base.update(overrides)
    return base


def test_valid_setup_constructs() -> None:
    s = Setup(**_valid_kwargs())
    assert s.id  # default uuid filled in
    assert s.direction == "long"


def test_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Setup(**_valid_kwargs(timestamp=datetime(2024, 1, 15, 14, 30)))


def test_rejects_unknown_direction() -> None:
    with pytest.raises(ValidationError):
        Setup(**_valid_kwargs(direction="sideways"))


def test_rejects_non_canonical_features_missing() -> None:
    feats = _valid_features()
    feats.pop("ema_9")
    with pytest.raises(ValidationError, match="canonical FEATURE_COLUMNS"):
        Setup(**_valid_kwargs(features=feats))


def test_rejects_non_canonical_features_extra() -> None:
    feats = _valid_features()
    feats["unexpected_extra_feature"] = 1.0
    with pytest.raises(ValidationError, match="canonical FEATURE_COLUMNS"):
        Setup(**_valid_kwargs(features=feats))


def test_long_requires_stop_below_entry_below_target() -> None:
    with pytest.raises(ValidationError, match="Long setup requires"):
        Setup(**_valid_kwargs(direction="long", stop_price=101.0, entry_price=100.0, target_price=99.0))


def test_short_requires_target_below_entry_below_stop() -> None:
    with pytest.raises(ValidationError, match="Short setup requires"):
        Setup(
            **_valid_kwargs(
                direction="short",
                stop_price=99.0,
                entry_price=100.0,
                target_price=101.0,
            )
        )


def test_atr_at_entry_must_be_positive() -> None:
    with pytest.raises(ValidationError, match="atr_at_entry"):
        Setup(**_valid_kwargs(atr_at_entry=0.0))


def test_setup_is_frozen() -> None:
    s = Setup(**_valid_kwargs())
    with pytest.raises(ValidationError):
        s.entry_price = 999.0  # type: ignore[misc]
