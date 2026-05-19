"""Multi-symbol paper mode + multi-symbol backtest.

Coverage:

1. ``MultiSymbolPaperLoop`` builds one :class:`PaperTradingLoop` per
   symbol from per-symbol feeds and forwards bars to each.
2. A missing CSV for one symbol disables only that symbol — the others
   keep running.
3. ``MAX_TRADES_PER_SYMBOL_PER_DAY`` blocks new entries on a symbol
   that already hit its day cap (and the block is recorded as a
   ``risk_block`` for audit, not a silent drop).
4. ``MAX_TOTAL_TRADES_PER_DAY`` blocks new entries across all symbols
   once the global cap is hit.
5. ``MAX_ACTIVE_SYMBOLS`` prevents opening a position on a fresh
   symbol while N other symbols already hold positions.
6. The single-symbol loop's existing safety still holds: long+short on
   the same symbol cannot both fill (the executor enforces "one open
   position per portfolio"; this test verifies the guarantee from the
   orchestrator level).
7. ``run_multi_symbol_backtest`` aggregates per-symbol results +
   reports best/worst symbol.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import pytest

from features.feature_builder import FEATURE_COLUMNS
from paper.loop import (
    MultiSymbolBarResult,
    MultiSymbolPaperLoop,
    PaperTradingLoop,
    build_multi_symbol_paper_loop,
)
from strategies.base import Setup, Strategy, StrategyParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _settings(tmp_path: Path, **overrides):
    from config.settings import reload_settings
    from storage.db import init_db

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "REPORTS_DIR": str(tmp_path / "reports"),
        "MODELS_DIR": str(tmp_path / "models"),
        "DATABASE_URL": "sqlite:///:memory:",
        # Wide trading window so test timestamps reliably land inside it.
        "TRADING_WINDOW_START": "00:00",
        "TRADING_WINDOW_END": "23:55",
        "FORCE_FLAT_TIME": "23:55",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


class _CapturingNotifier:
    """Records every notify() call so tests can assert on the stream."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def notify(self, kind: str, /, **payload):
        self.calls.append((kind, dict(payload)))


def _make_setup(
    *, instrument: str, direction: str, ts: datetime, entry: float = 4500.0,
) -> Setup:
    if direction == "long":
        stop, target = entry - 2.0, entry + 4.0
    else:
        stop, target = entry + 2.0, entry - 4.0
    features = {col: 0.0 for col in FEATURE_COLUMNS}
    return Setup(
        instrument=instrument,
        timestamp=ts,
        strategy_name="test",
        direction=direction,  # type: ignore[arg-type]
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        atr_at_entry=2.5,
        features=features,
        bar_index=0,
    )


class _FixedSetupStrategy(Strategy):
    """Strategy that emits a fixed setup at the latest bar's timestamp."""

    name = "fixed_setup"

    def __init__(self, *, instrument: str, direction: str = "long") -> None:
        super().__init__(StrategyParams())
        self.instrument = instrument
        self.direction = direction

    @classmethod
    def _default_params(cls) -> StrategyParams:
        return StrategyParams()

    def detect_setups(self, features_df: pd.DataFrame) -> list[Setup]:
        if features_df.empty:
            return []
        ts = features_df.index[-1].to_pydatetime()
        return [_make_setup(instrument=self.instrument, direction=self.direction, ts=ts)]


class _StubFeed:
    """Deterministic 1-step-at-a-time incremental feed.

    Doesn't subclass IncrementalFeed (which has abstract methods we'd
    have to satisfy) but quacks like one for the loop's purposes.
    """

    def __init__(self, *, instrument: str, n_bars: int = 80, base_price: float = 4500.0) -> None:
        from tests.fixtures.synthetic import synthetic_ohlcv

        self.instrument = instrument
        self.timeframe = "1m"
        df = synthetic_ohlcv(n_bars=n_bars, tz="UTC", base_price=base_price)
        self._all = df[["open", "high", "low", "close", "volume"]]
        self._cursor = 0
        self.window_bars = n_bars

    def is_exhausted(self) -> bool:
        return self._cursor >= len(self._all)

    def poll_latest(self):
        from data.candle import Candle
        from data.market_data_service import PollResult

        if self.is_exhausted():
            window = self._all.iloc[max(0, len(self._all) - self.window_bars):]
            return PollResult(new_candle=None, rolling_window=window.copy())
        idx = self._cursor
        ts = self._all.index[idx]
        row = self._all.iloc[idx]
        candle = Candle(
            instrument=self.instrument,
            timeframe=self.timeframe,
            ts=ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]),
        )
        self._cursor += 1
        end = self._cursor
        start = max(0, end - self.window_bars)
        return PollResult(new_candle=candle, rolling_window=self._all.iloc[start:end].copy())


def _build_loop_for_symbol(
    *, settings, symbol: str, n_bars: int = 80, direction: str = "long",
) -> PaperTradingLoop:
    notifier = _CapturingNotifier()
    feed = _StubFeed(instrument=symbol, n_bars=n_bars)
    loop = PaperTradingLoop(
        settings=settings,
        feed=feed,
        strategies=[_FixedSetupStrategy(instrument=symbol, direction=direction)],
        notifier=notifier,
        instrument=symbol,
    )
    loop.trading_enabled = True
    return loop


# ---------------------------------------------------------------------------
# 1. Orchestrator forwards to per-symbol loops
# ---------------------------------------------------------------------------
def test_multi_symbol_loop_forwards_to_each_symbol(tmp_path: Path) -> None:
    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES,MNQ",
        MAX_ACTIVE_SYMBOLS="2",
        MAX_TRADES_PER_SYMBOL_PER_DAY="4",
        MAX_TOTAL_TRADES_PER_DAY="8",
    )
    notifier = _CapturingNotifier()
    loops = {
        "MES": _build_loop_for_symbol(settings=s, symbol="MES"),
        "MNQ": _build_loop_for_symbol(settings=s, symbol="MNQ"),
    }
    multi = MultiSymbolPaperLoop(settings=s, loops=loops, notifier=notifier)

    res: MultiSymbolBarResult = multi.on_bar_close(datetime.now(timezone.utc))
    assert set(res.by_symbol) == {"MES", "MNQ"}
    # Both symbols got a new bar.
    assert all(r.new_bar for r in res.by_symbol.values())


# ---------------------------------------------------------------------------
# 2. Disabled symbol does not crash the whole bot
# ---------------------------------------------------------------------------
def test_disabled_symbol_isolated(tmp_path: Path) -> None:
    s = _settings(tmp_path, ENABLED_SYMBOLS="MES,MNQ,MGC")
    notifier = _CapturingNotifier()
    loops = {
        "MES": _build_loop_for_symbol(settings=s, symbol="MES"),
        "MNQ": _build_loop_for_symbol(settings=s, symbol="MNQ"),
    }
    multi = MultiSymbolPaperLoop(
        settings=s,
        loops=loops,
        notifier=notifier,
        disabled_symbols={"MGC": "csv_missing"},
    )
    res = multi.on_bar_close(datetime.now(timezone.utc))
    # MES + MNQ produced bars; MGC is reported as disabled.
    assert res.by_symbol["MES"].new_bar
    assert res.by_symbol["MNQ"].new_bar


def test_one_loop_failure_does_not_break_others(tmp_path: Path) -> None:
    s = _settings(tmp_path, ENABLED_SYMBOLS="MES,MNQ")
    notifier = _CapturingNotifier()

    class _ExplodingLoop(PaperTradingLoop):
        def on_bar_close(self, now):  # type: ignore[override]
            raise RuntimeError("boom")

    bad = _build_loop_for_symbol(settings=s, symbol="MNQ")
    # Swap class so on_bar_close raises but the orchestrator survives.
    bad.__class__ = _ExplodingLoop
    loops = {
        "MES": _build_loop_for_symbol(settings=s, symbol="MES"),
        "MNQ": bad,
    }
    multi = MultiSymbolPaperLoop(settings=s, loops=loops, notifier=notifier)
    res = multi.on_bar_close(datetime.now(timezone.utc))
    assert res.by_symbol["MES"].new_bar
    assert any("loop_failed" in e for e in res.by_symbol["MNQ"].errors)


# ---------------------------------------------------------------------------
# Direct-gate caps tests
#
# The feature builder needs ~200 warmup bars before producing a single
# row of features. Driving full ``on_bar_close`` cycles for cap
# enforcement is therefore slow + brittle (you'd need 500+ synthetic
# bars per symbol). Instead we invoke ``_handle_setup`` directly with
# a synthetic Setup. The gate logic under test is the same code path
# in both cases — it only depends on Portfolio state + the orchestrator's
# counters.
# ---------------------------------------------------------------------------
class _NullExecutor:
    """Drop-in executor that records submits without persisting trades.

    Calling submit() opens the position on the per-symbol portfolio so
    later gate checks see ``open_position`` correctly, but skips the
    fills model and DB writes that the real PaperExecutor performs.
    """

    def __init__(self, portfolio):
        self._portfolio = portfolio
        self.last_closed_trade_id = None

    def submit(self, order):
        # Open the position with zero slippage / commission so the
        # portfolio reflects "we are in the market" — that's all the
        # gate cares about. The portfolio refuses a second open while
        # one already exists; we re-raise as KillSwitchActive (which
        # the loop catches silently) so opposing-direction tests can
        # observe "no fill" without surfacing as a generic error.
        from execution.base import Fill
        from execution.paper_executor import KillSwitchActive

        if self._portfolio.open_position is not None:
            raise KillSwitchActive("already open")

        self._portfolio.open(
            setup_id=order.setup_id,
            instrument=order.instrument,
            direction=order.direction,
            quantity=order.quantity,
            ts=datetime.now(timezone.utc),
            entry_price=order.entry_price,
            stop_price=order.stop_price,
            target_price=order.target_price,
            commission=0.0,
            slippage=0.0,
            bar_index=0,
        )
        return Fill(
            order=order,
            fill_ts=datetime.now(timezone.utc),
            fill_price=order.entry_price,
            commission=0.0,
            slippage=0.0,
        )


def _build_test_loop(*, settings, symbol: str, direction: str = "long") -> PaperTradingLoop:
    """A loop with a stubbed feed + a no-op executor, suitable for
    ``_handle_setup`` direct invocation in cap tests."""
    notifier = _CapturingNotifier()
    feed = _StubFeed(instrument=symbol, n_bars=10)
    loop = PaperTradingLoop(
        settings=settings,
        feed=feed,
        strategies=[_FixedSetupStrategy(instrument=symbol, direction=direction)],
        notifier=notifier,
        instrument=symbol,
    )
    loop.executor = _NullExecutor(loop.portfolio)
    loop.trading_enabled = True
    return loop


def _bar_series(price: float = 4500.0):
    return pd.Series(
        {"open": price, "high": price + 1, "low": price - 1, "close": price, "volume": 100.0},
        name=datetime.now(timezone.utc),
    )


def _notifier_calls(loop: PaperTradingLoop) -> list[tuple[str, dict]]:
    """Read the underlying ``_CapturingNotifier`` calls list, even when
    the orchestrator has wrapped the loop's notifier in a counting
    proxy."""
    notifier = loop.notifier
    while hasattr(notifier, "_inner"):
        notifier = notifier._inner
    return getattr(notifier, "calls", [])


def _drive_setup(loop: PaperTradingLoop, *, instrument: str, direction: str = "long") -> None:
    """Hand-feed a single setup through ``_handle_setup``. Returns the
    BarCycleResult for the caller to inspect cap blocks."""
    from paper.loop import BarCycleResult

    setup = _make_setup(
        instrument=instrument,
        direction=direction,
        ts=datetime.now(timezone.utc),
    )
    result = BarCycleResult(new_bar=True, trading_enabled=True)
    loop._handle_setup(setup, _bar_series(), datetime.now(timezone.utc), result)
    return result


# ---------------------------------------------------------------------------
# 3-5. Caps
# ---------------------------------------------------------------------------
def test_per_symbol_day_cap_enforced(tmp_path: Path) -> None:
    """Once the per-symbol daily fill count hits the cap, the next
    setup on that symbol is rejected with rule=per_symbol_day_cap."""
    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES",
        MAX_TRADES_PER_DAY="100",
        MAX_TRADES_PER_SYMBOL_PER_DAY="2",
        MAX_TOTAL_TRADES_PER_DAY="100",
        MAX_ACTIVE_SYMBOLS="2",
    )
    notifier = _CapturingNotifier()
    loop = _build_test_loop(settings=s, symbol="MES")
    multi = MultiSymbolPaperLoop(
        settings=s, loops={"MES": loop}, notifier=notifier
    )

    # Two fills allowed.
    r1 = _drive_setup(loop, instrument="MES")
    assert r1.setups_filled == 1
    # Close the position so the next setup is allowed by the per-symbol
    # portfolio (``open_position`` would otherwise reject it).
    loop.portfolio.close(
        ts=datetime.now(timezone.utc),
        exit_price=loop.portfolio.open_position.entry_price + 4.0,
        exit_reason="tp",
        commission=0.0,
        slippage=0.0,
        bar_index=0,
    )
    r2 = _drive_setup(loop, instrument="MES")
    assert r2.setups_filled == 1
    loop.portfolio.close(
        ts=datetime.now(timezone.utc),
        exit_price=loop.portfolio.open_position.entry_price + 4.0,
        exit_reason="tp",
        commission=0.0,
        slippage=0.0,
        bar_index=0,
    )

    # Third attempt: cap is 2 per symbol per day → blocked.
    r3 = _drive_setup(loop, instrument="MES")
    assert r3.setups_filled == 0
    assert r3.setups_risk_blocked == 1
    cap_blocks = [
        c for c in _notifier_calls(loop)
        if c[0] == "trade.blocked" and c[1].get("rule") == "per_symbol_day_cap"
    ]
    assert cap_blocks, "expected per_symbol_day_cap block on the 3rd setup"


def test_total_day_cap_blocks_across_symbols(tmp_path: Path) -> None:
    """Once the global daily total fill count hits the cap, no more
    setups on ANY symbol can fill."""
    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES,MNQ",
        MAX_TRADES_PER_DAY="100",
        MAX_TRADES_PER_SYMBOL_PER_DAY="100",
        MAX_TOTAL_TRADES_PER_DAY="1",
        MAX_ACTIVE_SYMBOLS="2",
    )
    notifier = _CapturingNotifier()
    loops = {
        "MES": _build_test_loop(settings=s, symbol="MES"),
        "MNQ": _build_test_loop(settings=s, symbol="MNQ"),
    }
    multi = MultiSymbolPaperLoop(settings=s, loops=loops, notifier=notifier)

    r_mes = _drive_setup(loops["MES"], instrument="MES")
    assert r_mes.setups_filled == 1
    # Close MES position so the next attempt isn't auto-blocked by the
    # per-symbol portfolio's "one open at a time" rule.
    loops["MES"].portfolio.close(
        ts=datetime.now(timezone.utc),
        exit_price=loops["MES"].portfolio.open_position.entry_price + 4.0,
        exit_reason="tp",
        commission=0.0,
        slippage=0.0,
        bar_index=0,
    )

    # Fresh setup on MNQ: the orchestrator's total-day-cap rule kicks in.
    r_mnq = _drive_setup(loops["MNQ"], instrument="MNQ")
    assert r_mnq.setups_filled == 0
    assert r_mnq.setups_risk_blocked == 1
    cap_blocks = [
        c for c in _notifier_calls(loops["MNQ"])
        if c[0] == "trade.blocked" and c[1].get("rule") == "total_day_cap"
    ]
    assert cap_blocks, "expected total_day_cap block once global cap is hit"


def test_max_active_symbols_blocks_new_entries(tmp_path: Path) -> None:
    """While MES has an open position, MNQ trying to open a position
    must be blocked by max_active_symbols=1."""
    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES,MNQ",
        MAX_ACTIVE_SYMBOLS="1",
        MAX_TRADES_PER_DAY="100",
        MAX_TRADES_PER_SYMBOL_PER_DAY="100",
        MAX_TOTAL_TRADES_PER_DAY="100",
    )
    notifier = _CapturingNotifier()
    loops = {
        "MES": _build_test_loop(settings=s, symbol="MES"),
        "MNQ": _build_test_loop(settings=s, symbol="MNQ"),
    }
    multi = MultiSymbolPaperLoop(settings=s, loops=loops, notifier=notifier)

    r_mes = _drive_setup(loops["MES"], instrument="MES")
    assert r_mes.setups_filled == 1
    # MES is now active. MNQ should not be allowed to enter.
    r_mnq = _drive_setup(loops["MNQ"], instrument="MNQ")
    assert r_mnq.setups_filled == 0
    assert r_mnq.setups_risk_blocked == 1
    cap_blocks = [
        c for c in _notifier_calls(loops["MNQ"])
        if c[0] == "trade.blocked" and c[1].get("rule") == "max_active_symbols"
    ]
    assert cap_blocks, "expected max_active_symbols block on MNQ"
    assert multi.open_position_symbols() == ["MES"]


# ---------------------------------------------------------------------------
# 6. Same-symbol opposing positions can never both fill
# ---------------------------------------------------------------------------
def test_same_symbol_opposing_positions_never_both_fill(tmp_path: Path) -> None:
    """The per-symbol Portfolio enforces a single open position at a
    time; the orchestrator inherits that guarantee. We verify it
    holds even when MAX_ACTIVE_SYMBOLS>=2 (so the orchestrator-level
    cap is not what's protecting us)."""
    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES",
        MAX_ACTIVE_SYMBOLS="2",
        MAX_TRADES_PER_SYMBOL_PER_DAY="100",
        MAX_TOTAL_TRADES_PER_DAY="100",
    )
    notifier = _CapturingNotifier()
    loop = _build_test_loop(settings=s, symbol="MES", direction="long")
    multi = MultiSymbolPaperLoop(
        settings=s, loops={"MES": loop}, notifier=notifier
    )

    r_long = _drive_setup(loop, instrument="MES", direction="long")
    assert r_long.setups_filled == 1
    # Try to fire the opposing direction on the same symbol while a
    # position is open. The portfolio refuses (one position per
    # portfolio); the loop surfaces this as a sizing/executor error
    # rather than a fill.
    r_short = _drive_setup(loop, instrument="MES", direction="short")
    assert r_short.setups_filled == 0
    assert loop.portfolio.open_position is not None
    assert loop.portfolio.open_position.direction == "long"


# ---------------------------------------------------------------------------
# 7. Multi-symbol backtest aggregator
# ---------------------------------------------------------------------------
def test_multi_symbol_backtest_aggregates_per_symbol(tmp_path: Path) -> None:
    from backtesting.engine import (
        MultiSymbolBacktestResult,
        run_multi_symbol_backtest,
    )
    from features.feature_builder import build_features
    from strategies.registry import instantiate as instantiate_strategy
    from tests.fixtures.synthetic import synthetic_ohlcv

    s = _settings(
        tmp_path,
        ENABLED_SYMBOLS="MES,MNQ",
        TRADING_WINDOW_START="00:00",
        TRADING_WINDOW_END="23:55",
        FORCE_FLAT_TIME="23:55",
    )

    ohlcv_by_symbol: dict[str, pd.DataFrame] = {}
    setups_by_symbol: dict[str, list] = {}
    for sym, base_price, seed in (("MES", 4500.0, 1), ("MNQ", 18_000.0, 2)):
        df = synthetic_ohlcv(n_bars=2_000, tz=s.TIMEZONE, base_price=base_price, seed=seed)
        feats = build_features(df, instrument=sym, tz=s.TIMEZONE)
        strategy = instantiate_strategy("vwap_ema_pullback", instrument=sym)
        ohlcv_by_symbol[sym] = df
        setups_by_symbol[sym] = strategy.detect_setups(feats)

    result = run_multi_symbol_backtest(
        settings=s,
        ohlcv_by_symbol=ohlcv_by_symbol,
        setups_by_symbol=setups_by_symbol,
    )
    assert isinstance(result, MultiSymbolBacktestResult)
    assert set(result.per_symbol) == {"MES", "MNQ"}
    # Each per-symbol result has its own metrics object (or None if no trades).
    for sym, r in result.per_symbol.items():
        assert r.instrument == sym
    # Aggregate metrics exists when at least one symbol had a trade.
    payload = result.to_dict()
    assert "per_symbol" in payload
    assert "best_symbol" in payload
    assert "worst_symbol" in payload
