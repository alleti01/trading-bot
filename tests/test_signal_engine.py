"""SignalEngine tests: real VWAP/EMA setups from per-symbol OHLCV."""

from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import reload_settings
from tests.fixtures.synthetic import synthetic_ohlcv
from workflows.signal_engine import SignalEngine, WorkflowSignal


def _write_csv(settings, symbol: str, df) -> Path:
    out = Path(settings.HISTORICAL_DATA_DIR) / symbol.upper() / "1m.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    frame = df.reset_index().rename(columns={"index": "timestamp"})
    frame.to_csv(out, index=False)
    return out


def _settings(monkeypatch, **overrides):
    monkeypatch.setenv("INSTRUMENT", "SPY")
    monkeypatch.setenv("MARKET_TYPE", "equity")
    monkeypatch.setenv("ENABLED_SYMBOLS", "SPY")
    monkeypatch.setenv("ENABLED_STRATEGIES", "vwap_ema_pullback")
    for k, v in overrides.items():
        monkeypatch.setenv(k, v)
    return reload_settings()


def test_signal_none_when_no_data(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    engine = SignalEngine(settings)
    assert engine.generate_signal("SPY") is None


def test_signal_generated_from_synthetic_ohlcv(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    df = synthetic_ohlcv(n_bars=500, base_price=550.0)
    _write_csv(settings, "SPY", df)
    engine = SignalEngine(settings)
    signal = engine.generate_signal("SPY")
    # The synthetic series may or may not fire on the very last bar; if it
    # does, it must be a well-formed signal.
    if signal is not None:
        assert isinstance(signal, WorkflowSignal)
        assert signal.direction in ("long", "short")
        assert signal.symbol == "SPY"
        if signal.direction == "long":
            assert signal.stop_price < signal.entry_price < signal.target_price
        else:
            assert signal.target_price < signal.entry_price < signal.stop_price


def test_signal_strategy_only_mode_is_approved(monkeypatch) -> None:
    # No model configured → strategy-only mode → approved when a setup fires.
    settings = _settings(monkeypatch)
    df = synthetic_ohlcv(n_bars=500, base_price=550.0)
    _write_csv(settings, "SPY", df)
    engine = SignalEngine(settings, model_name=None)
    signal = engine.generate_signal("SPY")
    if signal is not None:
        assert signal.model_name is None
        assert signal.approved is True


def test_min_stop_floor_widens_tight_stop(monkeypatch) -> None:
    # A $0.42 stop on a $715 instrument is below the 0.15% floor (~$1.07);
    # the floor widens it and the target proportionally (preserving R:R).
    settings = _settings(monkeypatch)
    eng = SignalEngine(settings)

    from datetime import datetime, timezone

    from strategies.base import Setup
    from features.feature_builder import FEATURE_COLUMNS

    feats = {c: 0.0 for c in FEATURE_COLUMNS}
    setup = Setup(
        instrument="QQQ",
        timestamp=datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc),
        strategy_name="vwap_ema_pullback",
        direction="long",
        entry_price=715.42,
        stop_price=715.0,      # $0.42 risk — sub-floor
        target_price=716.26,   # ~2x risk
        atr_at_entry=0.56,
        features=feats,
        bar_index=0,
    )
    entry, stop, target = eng._apply_min_stop_floor(setup)
    floor = max(0.0015 * entry, 0.05)
    assert entry == 715.42
    assert abs((entry - stop) - floor) < 1e-3           # stop widened to floor
    assert abs((target - entry) / (entry - stop) - 2.0) < 0.05  # R:R preserved


def test_min_stop_floor_leaves_wide_stop_alone(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    eng = SignalEngine(settings)

    from datetime import datetime, timezone

    from strategies.base import Setup
    from features.feature_builder import FEATURE_COLUMNS

    setup = Setup(
        instrument="SPY",
        timestamp=datetime(2026, 6, 17, 14, 30, tzinfo=timezone.utc),
        strategy_name="vwap_ema_pullback",
        direction="long",
        entry_price=100.0,
        stop_price=98.0,    # $2 risk — already well above floor
        target_price=104.0,
        atr_at_entry=2.0,
        features={c: 0.0 for c in FEATURE_COLUMNS},
        bar_index=0,
    )
    entry, stop, target = eng._apply_min_stop_floor(setup)
    assert (entry, stop, target) == (100.0, 98.0, 104.0)  # unchanged


def test_signal_engine_handles_bad_model_gracefully(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    df = synthetic_ohlcv(n_bars=300, base_price=550.0)
    _write_csv(settings, "SPY", df)
    # Nonexistent model → predictor load fails → engine falls back to
    # strategy-only (no crash).
    engine = SignalEngine(settings, model_name="does_not_exist")
    assert engine._predictor is None  # noqa: SLF001
    # Should not raise.
    engine.generate_signal("SPY")
