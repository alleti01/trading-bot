"""Candle Pydantic schema."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from data.candle import Candle


def _kwargs(**overrides) -> dict:
    base = dict(
        instrument="MES",
        timeframe="1m",
        ts=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        volume=1000.0,
    )
    base.update(overrides)
    return base


def test_valid_candle() -> None:
    c = Candle(**_kwargs())
    assert c.high >= c.low


def test_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        Candle(**_kwargs(ts=datetime(2024, 1, 15, 14, 30)))


def test_rejects_high_below_low() -> None:
    with pytest.raises(ValidationError, match=r"high \(\d.*\) < low"):
        Candle(**_kwargs(high=98.0, low=99.0))


def test_rejects_high_below_open() -> None:
    with pytest.raises(ValidationError, match="high"):
        Candle(**_kwargs(open=102.0, high=101.0))


def test_rejects_low_above_close() -> None:
    with pytest.raises(ValidationError, match="low"):
        Candle(**_kwargs(close=98.0, low=99.0))


def test_rejects_negative_volume() -> None:
    with pytest.raises(ValidationError, match="volume"):
        Candle(**_kwargs(volume=-1.0))
