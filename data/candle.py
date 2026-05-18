"""Strict Candle schema.

Candles are the atomic unit everything else consumes, so we validate them
hard at the boundary: tz-aware timestamps only, OHLC ordering enforced.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class Candle(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    instrument: str
    timeframe: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @field_validator("ts")
    @classmethod
    def _ts_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Candle.ts must be timezone-aware (no naive datetimes).")
        return v

    @field_validator("volume")
    @classmethod
    def _non_negative_volume(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Candle.volume must be non-negative.")
        return v

    @model_validator(mode="after")
    def _ohlc_ordering(self) -> "Candle":
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if self.high < max(self.open, self.close):
            raise ValueError(
                f"high ({self.high}) < max(open, close) "
                f"= {max(self.open, self.close)}"
            )
        if self.low > min(self.open, self.close):
            raise ValueError(
                f"low ({self.low}) > min(open, close) "
                f"= {min(self.open, self.close)}"
            )
        return self

    def to_row(self) -> dict[str, Any]:
        """Plain dict for ORM insertion."""
        return self.model_dump()
