"""Model load failures must block trading without crashing the bot."""

from __future__ import annotations

import os

from data.market_data_service import SyntheticLiveFeed
from notifications.notification_service import NotificationService
from paper.loop import build_paper_loop


def _settings(**overrides):
    from config.settings import reload_settings
    from storage.db import init_db

    defaults = {
        "MODE": "PAPER",
        "INSTRUMENT": "MES",
        "MARKET_TYPE": "futures",
        "TIMEZONE": "America/New_York",
        "DATABASE_URL": "sqlite:///:memory:",
    }
    defaults.update({k: str(v) for k, v in overrides.items()})
    for k, v in defaults.items():
        os.environ[k] = v
    s = reload_settings()
    init_db()
    return s


def test_missing_model_disables_trading_but_does_not_raise() -> None:
    s = _settings()
    feed = SyntheticLiveFeed(
        instrument="MES", timeframe="1m", tz="America/New_York",
        max_bars=10, window_bars=5,
    )
    notifier = NotificationService(discord=None)

    # Build with a model name that does not exist in the registry.
    loop = build_paper_loop(
        settings=s,
        feed=feed,
        notifier=notifier,
        model_name="nonexistent_model",
        model_version="latest",
    )
    assert loop.trading_enabled is False
    assert loop.predictor is None


def test_disabled_loop_does_not_open_positions() -> None:
    """on_bar_close on a disabled loop must not call the executor."""
    s = _settings()
    feed = SyntheticLiveFeed(
        instrument="MES", timeframe="1m", tz="America/New_York",
        max_bars=10, window_bars=5,
    )
    notifier = NotificationService(discord=None)
    loop = build_paper_loop(
        settings=s,
        feed=feed,
        notifier=notifier,
        model_name="nonexistent_model",
    )

    from datetime import datetime
    from zoneinfo import ZoneInfo

    NY = ZoneInfo("America/New_York")
    res = loop.on_bar_close(datetime(2024, 1, 15, 14, 0, tzinfo=NY))
    # New bar from synthetic feed, but trading disabled → zero entries.
    assert res.new_bar is True
    assert res.setups_filled == 0
    assert loop.portfolio.open_position is None
