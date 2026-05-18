"""Integration: NewsAgent -> orchestrator flag -> paper loop blocks via risk engine."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from agents.llm_client import MockLLMClient
from agents.orchestrator import AgentOrchestrator
from backtesting.fills import make_fills_model
from backtesting.portfolio import Portfolio
from config.instruments import get_instrument
from config.settings import reload_settings
from data.candle import Candle
from data.market_data_service import IncrementalFeed, PollResult
from execution.paper_executor import PaperExecutor
from features.feature_builder import FEATURE_COLUMNS
from notifications.notification_service import NotificationService
from paper.loop import PaperTradingLoop
from storage.db import init_db
from strategies.base import Setup, Strategy, StrategyParams


NY = ZoneInfo("America/New_York")


def _settings(**overrides):
    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "TRADING_WINDOW_START": "09:30",
        "TRADING_WINDOW_END": "15:55",
        "FORCE_FLAT_TIME": "15:55",
        "MAX_TRADES_PER_DAY": "8",
        "MAX_DAILY_LOSS": "10000",
        "MAX_DAILY_PROFIT": "10000",
        "MAX_POSITION_SIZE": "1",
        "RISK_PER_TRADE": "100",
        "MAX_OPEN_POSITIONS": "1",
        "COOLDOWN_AFTER_LOSS_MINUTES": "0",
        "COOLDOWN_AFTER_LARGE_WIN_MINUTES": "0",
        "LARGE_WIN_THRESHOLD": "9999",
        "MAX_HOLD_BARS": "5",
        "SLIPPAGE_TICKS": "0",
        "COMMISSION_PER_CONTRACT": "0",
        "DATABASE_URL": "sqlite:///:memory:",
        "ENABLE_LLM_AGENTS": "true",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


# ---------------------------------------------------------------------------
# Shared scripted feed + scripted strategy (mirrors test_paper_loop helpers).
# ---------------------------------------------------------------------------
class _ScriptedFeed(IncrementalFeed):
    def __init__(self, candle: Candle, window: pd.DataFrame) -> None:
        self.instrument = candle.instrument
        self.timeframe = candle.timeframe
        self._items: list[tuple[Candle, pd.DataFrame]] = [(candle, window)]
        self._idx = 0

    def is_exhausted(self) -> bool:
        return self._idx >= len(self._items)

    def poll_latest(self) -> PollResult:
        if self.is_exhausted():
            return PollResult(new_candle=None, rolling_window=pd.DataFrame())
        candle, window = self._items[self._idx]
        self._idx += 1
        return PollResult(new_candle=candle, rolling_window=window)


class _OneShotStrategy(Strategy):
    name = "one_shot"

    def __init__(self, setup: Setup) -> None:
        super().__init__()
        self._setup = setup

    @classmethod
    def _default_params(cls):
        return StrategyParams()

    def detect_setups(self, features_df: pd.DataFrame) -> list[Setup]:
        return [self._setup] if self._setup.timestamp in features_df.index else []


def _setup(ts: datetime) -> Setup:
    return Setup(
        instrument="MES",
        timestamp=ts,
        strategy_name="one_shot",
        direction="long",
        entry_price=4500.0,
        stop_price=4498.0,
        target_price=4504.0,
        atr_at_entry=1.0,
        features={c: 0.0 for c in FEATURE_COLUMNS},
        bar_index=0,
    )


def _window(ts: datetime) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "open": [4500.0],
            "high": [4501.0],
            "low": [4499.0],
            "close": [4500.0],
            "volume": [1000.0],
        },
        index=pd.DatetimeIndex([ts]),
    )
    df.index.name = "timestamp"
    return df


def _make_loop(
    settings, *, setup: Setup, news_fn
) -> tuple[PaperTradingLoop, list[tuple[str, dict[str, Any]]]]:
    spec = get_instrument("MES")
    portfolio = Portfolio(instrument_spec=spec)
    fills = make_fills_model("MES", slippage_ticks=0.0, commission_per_contract=0.0)
    executor = PaperExecutor(portfolio=portfolio, fills_model=fills)
    notifier = NotificationService(discord=None)
    candle = Candle(
        instrument="MES",
        timeframe="1m",
        ts=setup.timestamp,
        open=4500.0,
        high=4501.0,
        low=4499.0,
        close=4500.0,
        volume=1000.0,
    )
    loop = PaperTradingLoop(
        settings=settings,
        feed=_ScriptedFeed(candle, _window(setup.timestamp)),
        strategy=_OneShotStrategy(setup),
        notifier=notifier,
        portfolio=portfolio,
        executor=executor,
        high_risk_news_fn=news_fn,
    )
    # Skip the heavy feature builder for this test — we already pass a
    # frozen Setup with all feature columns populated, and we only care
    # about the risk-engine path.
    import paper.loop as loop_mod

    def fake_build_features(df, *, instrument, tz):  # noqa: ARG001
        return df

    loop_mod.build_features = fake_build_features  # type: ignore[attr-defined]
    return loop, []


# ---------------------------------------------------------------------------
def test_news_flag_off_lets_setup_trade() -> None:
    s = _settings()
    ts = datetime(2026, 5, 18, 14, 0, tzinfo=NY)
    loop, _ = _make_loop(s, setup=_setup(ts), news_fn=lambda: False)

    res = loop.on_bar_close(ts)
    assert res.setups_seen == 1
    assert res.setups_filled == 1
    assert res.setups_risk_blocked == 0


def test_news_flag_on_blocks_setup_via_risk_engine() -> None:
    s = _settings()
    ts = datetime(2026, 5, 18, 14, 0, tzinfo=NY)
    loop, _ = _make_loop(s, setup=_setup(ts), news_fn=lambda: True)

    res = loop.on_bar_close(ts)
    assert res.setups_seen == 1
    assert res.setups_filled == 0
    assert res.setups_risk_blocked == 1


def test_orchestrator_news_high_risk_payload_flips_callback_used_by_loop() -> None:
    """Full chain: orchestrator pre-session run -> high_risk_news_active() -> loop blocks."""
    s = _settings()
    high_risk = json.dumps(
        {
            "high_risk_window": True,
            "severity": "high",
            "events": ["FOMC"],
            "summary": "FOMC at 14:00",
            "recommendation": "Stand aside.",
        }
    )
    mock = MockLLMClient(responses_by_agent={"news": high_risk})
    orchestrator = AgentOrchestrator(s, llm=mock, notifier=NotificationService(discord=None))
    orchestrator.run_pre_session_news(
        now=datetime(2026, 5, 18, 13, 25, tzinfo=NY)
    )
    assert orchestrator.high_risk_news_active() is True

    ts = datetime(2026, 5, 18, 14, 0, tzinfo=NY)
    loop, _ = _make_loop(
        s, setup=_setup(ts), news_fn=orchestrator.high_risk_news_active
    )
    res = loop.on_bar_close(ts)
    assert res.setups_filled == 0
    assert res.setups_risk_blocked == 1


def test_callback_exception_falls_back_to_no_block() -> None:
    s = _settings()
    ts = datetime(2026, 5, 18, 14, 0, tzinfo=NY)

    def angry() -> bool:
        raise RuntimeError("flag query failed")

    loop, _ = _make_loop(s, setup=_setup(ts), news_fn=angry)
    res = loop.on_bar_close(ts)
    # Loop logs and treats the flag as False — trade goes through.
    assert res.setups_filled == 1
