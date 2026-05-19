"""Opening Range Breakout (ORB) strategy.

A trade fires when price breaks the session opening range (the first
``bars_in_or`` bars of the session, captured by the feature builder via
``dist_from_or_high`` and ``dist_from_or_low``).

Conditions (long; short is the mirror):

- ``close > or_high`` (i.e. ``dist_from_or_high > 0``)
- breakout is fresh: ``0 < dist_from_or_high <= confirm_atr_max``
- ``volume_ratio_20 >= min_volume_ratio``
- ``atr_min <= atr_14 <= atr_max``
- ``session_label == 1`` (only inside the configured session)

Setups carry an ATR-multiple stop and target, the standard contract used
everywhere downstream. Like every other strategy, this one consults no
model — model approval is the next layer.
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pydantic import Field

from features.feature_builder import FEATURE_COLUMNS
from strategies.base import Setup, Strategy, StrategyParams


class OpeningRangeBreakoutParams(StrategyParams):
    confirm_atr_max: float = Field(default=1.5, gt=0)
    min_volume_ratio: float = Field(default=0.8, ge=0)
    atr_min: float = Field(default=0.5, gt=0)
    atr_max: float = Field(default=10.0, gt=0)
    stop_atr_mult: float = Field(default=1.0, gt=0)
    target_atr_mult: float = Field(default=2.0, gt=0)


class OpeningRangeBreakout(Strategy):
    name: ClassVar[str] = "opening_range_breakout"

    def __init__(
        self,
        params: OpeningRangeBreakoutParams | None = None,
        *,
        instrument: str = "MES",
    ) -> None:
        super().__init__(params or self._default_params())
        self.instrument = instrument

    @classmethod
    def _default_params(cls) -> OpeningRangeBreakoutParams:
        return OpeningRangeBreakoutParams()

    # ----------------------------------------------------------------------
    # Detection
    # ----------------------------------------------------------------------
    def detect_setups(self, features_df: pd.DataFrame) -> list[Setup]:
        p: OpeningRangeBreakoutParams = self.params  # type: ignore[assignment]

        required = {
            "close",
            "atr_14",
            "volume_ratio_20",
            "dist_from_or_high",
            "dist_from_or_low",
            "session_label",
        }
        missing = required - set(features_df.columns)
        if missing:
            raise ValueError(
                f"features_df missing required columns for {self.name}: "
                f"{sorted(missing)}"
            )

        atr = features_df["atr_14"]
        vol_ratio = features_df["volume_ratio_20"]
        dh = features_df["dist_from_or_high"]
        dl = features_df["dist_from_or_low"]
        sess = features_df["session_label"]

        atr_band_ok = (atr >= p.atr_min) & (atr <= p.atr_max)
        vol_ok = vol_ratio >= p.min_volume_ratio
        in_session = sess.astype(bool)

        # ``dist_from_or_high`` is NaN before the OR is finalized; the
        # arithmetic comparisons evaluate to False which is what we want.
        long_signal = (
            (dh > 0)
            & (dh <= p.confirm_atr_max)
            & vol_ok
            & atr_band_ok
            & in_session
        )
        short_signal = (
            (dl < 0)
            & (-dl <= p.confirm_atr_max)
            & vol_ok
            & atr_band_ok
            & in_session
        )

        setups: list[Setup] = []
        for i, ts in enumerate(features_df.index):
            if long_signal.iloc[i]:
                setups.append(self._build_setup(features_df, i, ts, "long"))
            elif short_signal.iloc[i]:
                setups.append(self._build_setup(features_df, i, ts, "short"))

        return setups

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def _build_setup(
        self,
        features_df: pd.DataFrame,
        i: int,
        ts: pd.Timestamp,
        direction: str,
    ) -> Setup:
        p: OpeningRangeBreakoutParams = self.params  # type: ignore[assignment]
        row = features_df.iloc[i]
        entry = float(row["close"])
        a = float(row["atr_14"])

        if direction == "long":
            stop = entry - p.stop_atr_mult * a
            target = entry + p.target_atr_mult * a
        else:
            stop = entry + p.stop_atr_mult * a
            target = entry - p.target_atr_mult * a

        feat_snapshot = {col: float(row[col]) for col in FEATURE_COLUMNS}

        return Setup(
            instrument=self.instrument,
            timestamp=ts.to_pydatetime(),
            strategy_name=self.name,
            direction=direction,  # type: ignore[arg-type]
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            atr_at_entry=a,
            features=feat_snapshot,
            bar_index=i,
        )
