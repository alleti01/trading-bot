"""VWAP/EMA trend pullback strategy.

This is a *geometric* strategy — it emits Setups when the conditions in
the spec are met, with stops/targets sized in ATR multiples. It does not
score, score-filter, or otherwise consult any model. Model approval is a
later layer that lives between the strategy and the risk engine.

Conditions (long; short is mirrored):
- close > vwap
- ema_9 > ema_21 > ema_50
- pullback within ``pullback_atr_mult * atr_14`` of either ema_21 or vwap
- volume_ratio_20 >= ``min_volume_ratio``
- ``atr_min`` <= atr_14 <= ``atr_max``
"""

from __future__ import annotations

from typing import ClassVar

import pandas as pd
from pydantic import Field

from features.feature_builder import FEATURE_COLUMNS
from strategies.base import Setup, Strategy, StrategyParams


class VWAPEMAPullbackParams(StrategyParams):
    pullback_atr_mult: float = Field(default=0.5, gt=0)
    min_volume_ratio: float = Field(default=0.8, ge=0)
    atr_min: float = Field(default=0.5, gt=0)
    atr_max: float = Field(default=10.0, gt=0)
    stop_atr_mult: float = Field(default=1.0, gt=0)
    target_atr_mult: float = Field(default=2.0, gt=0)


class VWAPEMAPullback(Strategy):
    name: ClassVar[str] = "vwap_ema_pullback"

    def __init__(
        self,
        params: VWAPEMAPullbackParams | None = None,
        *,
        instrument: str = "MES",
    ) -> None:
        super().__init__(params or self._default_params())
        self.instrument = instrument

    @classmethod
    def _default_params(cls) -> VWAPEMAPullbackParams:
        return VWAPEMAPullbackParams()

    # ----------------------------------------------------------------------
    # Detection
    # ----------------------------------------------------------------------
    def detect_setups(self, features_df: pd.DataFrame) -> list[Setup]:
        p: VWAPEMAPullbackParams = self.params  # type: ignore[assignment]

        required = {
            "close",
            "vwap",
            "ema_9",
            "ema_21",
            "ema_50",
            "atr_14",
            "volume_ratio_20",
        }
        missing = required - set(features_df.columns)
        if missing:
            raise ValueError(
                f"features_df missing required columns for {self.name}: {sorted(missing)}"
            )

        close = features_df["close"]
        vwap = features_df["vwap"]
        e9 = features_df["ema_9"]
        e21 = features_df["ema_21"]
        e50 = features_df["ema_50"]
        atr = features_df["atr_14"]
        vol_ratio = features_df["volume_ratio_20"]

        atr_band_ok = (atr >= p.atr_min) & (atr <= p.atr_max)
        vol_ok = vol_ratio >= p.min_volume_ratio

        long_trend = (close > vwap) & (e9 > e21) & (e21 > e50)
        short_trend = (close < vwap) & (e9 < e21) & (e21 < e50)

        pullback_band = p.pullback_atr_mult * atr
        near_e21 = (close - e21).abs() <= pullback_band
        near_vwap = (close - vwap).abs() <= pullback_band
        near_anchor = near_e21 | near_vwap

        long_signal = long_trend & near_anchor & vol_ok & atr_band_ok
        short_signal = short_trend & near_anchor & vol_ok & atr_band_ok

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
        p: VWAPEMAPullbackParams = self.params  # type: ignore[assignment]
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
