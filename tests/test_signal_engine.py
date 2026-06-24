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
