"""Generate real trade signals for workflows (VWAP/EMA strategy + model).

This replaces the previous hardcoded ``side="long"`` in market-open. It
builds canonical features from per-symbol OHLCV, runs the enabled
strategy registry on the latest bar, optionally scores the setup with
the configured model, and returns a directional decision — or ``None``
when there is no setup or the model gate rejects it.

Data source: ``data/historical/<SYMBOL>/1m.csv`` (the repo convention).
When no bars are available the engine returns ``None`` (skip) rather
than inventing a direction — safety first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from app.logging_config import get_logger
from config.settings import Settings
from features.feature_builder import build_features
from strategies.base import Setup

_log = get_logger("workflows.signal_engine")


@dataclass(frozen=True)
class WorkflowSignal:
    symbol: str
    direction: str  # "long" | "short"
    entry_price: float
    stop_price: float
    target_price: float
    confidence: Optional[float]
    approved: bool
    model_name: Optional[str]
    reason: str


class SignalEngine:
    """Builds setups + model verdicts for workflow symbols."""

    def __init__(
        self,
        settings: Settings,
        *,
        model_name: Optional[str] = None,
        model_version: str = "latest",
    ) -> None:
        self.settings = settings
        self.model_name = model_name
        self.model_version = model_version
        self.log = _log
        self._predictor = None
        self._strategies = None
        self._load_predictor()

    def _load_predictor(self) -> None:
        if not self.model_name:
            return
        try:
            from models.model_registry import load_model
            from models.predictor import Predictor

            loaded = load_model(self.model_name, version=self.model_version)
            self._predictor = Predictor(loaded)
            self.log.info(
                "signal.model_loaded",
                model=self.model_name,
                version=self.model_version,
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning(
                "signal.model_load_failed",
                model=self.model_name,
                error=str(e),
            )
            self._predictor = None

    def _strategies_for(self, symbol: str):
        from strategies.registry import instantiate_enabled

        return instantiate_enabled(self.settings, instrument=symbol)

    def _load_ohlcv(self, symbol: str) -> Optional[pd.DataFrame]:
        path = Path(self.settings.HISTORICAL_DATA_DIR) / symbol.upper() / "1m.csv"
        if not path.exists():
            return None
        try:
            from data.csv_loader import load_ohlcv_csv

            return load_ohlcv_csv(
                path, symbol.upper(), "1m", self.settings.TIMEZONE
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning("signal.csv_load_failed", symbol=symbol, error=str(e))
            return None

    def generate_signal(self, symbol: str) -> Optional[WorkflowSignal]:
        """Return a real signal for ``symbol`` or ``None`` to skip."""
        df = self._load_ohlcv(symbol)
        if df is None or df.empty:
            self.log.info("signal.no_data", symbol=symbol)
            return None

        # Dynamic-universe symbols (allowlist equities) may not be in the
        # instrument registry yet. Mint an equity spec on demand so the
        # feature builder can look it up — futures/known symbols keep
        # their existing spec.
        from config.instruments import is_supported_symbol, register_equity

        if not is_supported_symbol(symbol):
            register_equity(symbol)

        df = df.copy()
        df["instrument"] = symbol.upper()
        df["timeframe"] = "1m"
        try:
            features = build_features(
                df, instrument=symbol.upper(), tz=self.settings.TIMEZONE
            )
        except Exception as e:  # noqa: BLE001
            self.log.warning("signal.features_failed", symbol=symbol, error=str(e))
            return None
        if features.empty:
            return None

        latest_ts = features.index[-1]
        setups: list[Setup] = []
        for strategy in self._strategies_for(symbol):
            try:
                found = strategy.detect_setups(features)
            except Exception as e:  # noqa: BLE001
                self.log.warning(
                    "signal.strategy_failed",
                    symbol=symbol,
                    strategy=getattr(strategy, "name", "?"),
                    error=str(e),
                )
                continue
            setups.extend(s for s in found if s.timestamp == latest_ts)

        if not setups:
            self.log.info("signal.no_setup", symbol=symbol)
            return None

        # If multiple strategies fire, resolve conflicts (never long+short).
        setup = self._resolve(setups)
        if setup is None:
            return None

        confidence: Optional[float] = None
        approved = True
        if self._predictor is not None:
            try:
                pred = self._predictor.predict_setup(setup)
                confidence = pred.probability
                approved = pred.approved
            except Exception as e:  # noqa: BLE001
                self.log.warning("signal.predict_failed", symbol=symbol, error=str(e))
                # No model verdict → treat as not approved (safe).
                approved = False

        if not approved:
            return WorkflowSignal(
                symbol=symbol.upper(),
                direction=setup.direction,
                entry_price=setup.entry_price,
                stop_price=setup.stop_price,
                target_price=setup.target_price,
                confidence=confidence,
                approved=False,
                model_name=self.model_name,
                reason="model_rejected",
            )

        return WorkflowSignal(
            symbol=symbol.upper(),
            direction=setup.direction,
            entry_price=setup.entry_price,
            stop_price=setup.stop_price,
            target_price=setup.target_price,
            confidence=confidence,
            approved=True,
            model_name=self.model_name,
            reason="strategy_signal" if self._predictor is None else "model_approved",
        )

    def _resolve(self, setups: list[Setup]) -> Optional[Setup]:
        if len(setups) == 1:
            return setups[0]
        from strategies.registry import ScoredSetup, resolve_conflicts

        scored: list[ScoredSetup] = []
        for s in setups:
            conf = None
            appr = None
            if self._predictor is not None:
                try:
                    pred = self._predictor.predict_setup(s)
                    conf = pred.probability
                    appr = pred.approved
                except Exception:  # noqa: BLE001
                    conf, appr = None, None
            scored.append(ScoredSetup(setup=s, confidence=conf, approved=appr))
        resolution = resolve_conflicts(scored)
        survivors = resolution.survivors
        return survivors[0].setup if survivors else None
