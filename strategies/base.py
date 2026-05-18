"""Strategy contract.

Two key types live here:

- ``Setup`` — the universal hand-off between a strategy and everything
  downstream (labeler, model, risk engine, executor, reporter). It carries
  a *frozen* feature snapshot so downstream code never recomputes features
  from raw bars (the most common lookahead bug).
- ``Strategy`` — abstract base. The implementation contract is "produce
  setups from a feature DataFrame; for each row, decide using only that
  row and earlier rows."
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import ClassVar, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from features.feature_builder import FEATURE_COLUMNS

Direction = Literal["long", "short"]


class Setup(BaseModel):
    """Frozen, fully-typed signal produced by a strategy.

    The ``features`` dict is the canonical feature snapshot at signal time
    and MUST contain exactly the columns listed in
    ``features.feature_builder.FEATURE_COLUMNS``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    CANONICAL_FEATURES: ClassVar[frozenset[str]] = frozenset(FEATURE_COLUMNS)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instrument: str
    timestamp: datetime
    strategy_name: str
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    atr_at_entry: float
    features: dict[str, float]
    bar_index: int

    @field_validator("timestamp")
    @classmethod
    def _ts_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("Setup.timestamp must be timezone-aware.")
        return v

    @field_validator("atr_at_entry")
    @classmethod
    def _atr_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Setup.atr_at_entry must be positive.")
        return v

    @field_validator("features")
    @classmethod
    def _features_match_canonical(cls, v: dict[str, float]) -> dict[str, float]:
        keys = set(v.keys())
        if keys != cls.CANONICAL_FEATURES:
            missing = cls.CANONICAL_FEATURES - keys
            extra = keys - cls.CANONICAL_FEATURES
            raise ValueError(
                "Setup.features must equal canonical FEATURE_COLUMNS exactly. "
                f"Missing={sorted(missing)}, Extra={sorted(extra)}"
            )
        return v

    @model_validator(mode="after")
    def _stop_target_consistency(self) -> "Setup":
        if self.direction == "long":
            if not (self.stop_price < self.entry_price < self.target_price):
                raise ValueError(
                    f"Long setup requires stop < entry < target "
                    f"(got stop={self.stop_price}, entry={self.entry_price}, "
                    f"target={self.target_price})"
                )
        else:  # short
            if not (self.target_price < self.entry_price < self.stop_price):
                raise ValueError(
                    f"Short setup requires target < entry < stop "
                    f"(got target={self.target_price}, entry={self.entry_price}, "
                    f"stop={self.stop_price})"
                )
        return self


class StrategyParams(BaseModel):
    """Base class for strategy-specific parameter models."""

    model_config = ConfigDict(extra="forbid")


class Strategy(ABC):
    name: ClassVar[str] = "abstract"

    def __init__(self, params: StrategyParams | None = None) -> None:
        self.params = params or self._default_params()

    @classmethod
    def _default_params(cls) -> StrategyParams:
        raise NotImplementedError

    @abstractmethod
    def detect_setups(self, features_df: pd.DataFrame) -> list[Setup]:
        """Scan features bar-by-bar and emit Setups.

        Implementation contract: for any row at index ``t``, the decision
        must use only that row (and earlier rows visible through the
        already-no-lookahead feature columns). Do not call ``shift(-k)``
        and do not look at ``features_df.iloc[t+1:]``.
        """
