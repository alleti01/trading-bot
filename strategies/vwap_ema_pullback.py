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
    # Tighter stop (0.75 ATR) with a 1:2 reward:risk (1.5 ATR target)
    # backtested materially better than 1.0/2.0 on 12mo of equities
    # (profit factor 1.11 vs 1.03, ~6x per-trade expectancy). Tight stops
    # cut losers faster; this is the best-validated config from the sweep.
    stop_atr_mult: float = Field(default=0.75, gt=0)
    target_atr_mult: float = Field(default=1.5, gt=0)
    # Time-of-day filter (minutes from the RTH 09:30–16:00 session edges).
    # 0 = no filtering. Skipping the noisy open/close often helps
    # tight-stop strategies. Applied to setups by their bar timestamp.
    skip_open_minutes: float = Field(default=0.0, ge=0)
    skip_close_minutes: float = Field(default=0.0, ge=0)


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

        # Time-of-day mask: drop setups within skip_open/skip_close minutes
        # of the RTH session edges (09:30 = 570 min, 16:00 = 960 min).
        tod_ok = self._time_of_day_mask(features_df.index, p)

        setups: list[Setup] = []
        for i, ts in enumerate(features_df.index):
            if not tod_ok[i]:
                continue
            if long_signal.iloc[i]:
                setups.append(self._build_setup(features_df, i, ts, "long"))
            elif short_signal.iloc[i]:
                setups.append(self._build_setup(features_df, i, ts, "short"))

        return setups

    @staticmethod
    def _time_of_day_mask(index, p: "VWAPEMAPullbackParams"):  # noqa: ANN001
        """Boolean list: True where the bar is outside the skip windows."""
        n = len(index)
        if p.skip_open_minutes <= 0 and p.skip_close_minutes <= 0:
            return [True] * n
        session_open = 9 * 60 + 30  # 570
        session_close = 16 * 60     # 960
        lo = session_open + p.skip_open_minutes
        hi = session_close - p.skip_close_minutes
        out = []
        for ts in index:
            minute = ts.hour * 60 + ts.minute
            out.append(lo <= minute <= hi)
        return out

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
