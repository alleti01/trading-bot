"""Deterministic risk engine.

INVARIANT (architectural): the risk engine is authoritative. The model
does not override it. Agents do not override it. Strategies do not
override it. If ``evaluate(...)`` returns ``allowed=False``, no trade
gets routed.

The engine is a **pure function** of:

- the proposed ``Setup``,
- a ``Portfolio`` snapshot (open position, day stats, last trade),
- a ``RiskConfig`` (typically lifted from ``Settings``),
- the current time,
- a few external signals (kill switch state, high-risk-news flag).

It does no DB writes and no notifications. The caller persists the
``RiskDecision`` in whatever way makes sense (the backtester collects
them in the ``BacktestResult``; the future paper service will write rows
to ``risk_blocks``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from backtesting.portfolio import Portfolio
from config.settings import Settings
from strategies.base import Setup


# ---------------------------------------------------------------------------
# Decisions and config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    rule: str         # short rule code: "kill_switch", "max_trades_per_day", etc.
    reason: str       # human-readable explanation


@dataclass(frozen=True)
class RiskConfig:
    """Subset of ``Settings`` the risk engine actually needs.

    Frozen + minimal so it's easy to construct in tests without booting
    the full ``Settings`` object.
    """

    timezone: str
    trading_window_start: time
    trading_window_end: time
    force_flat_time: time
    max_trades_per_day: int
    max_daily_loss: float
    max_daily_profit: float
    max_open_positions: int
    cooldown_after_loss_minutes: int
    cooldown_after_large_win_minutes: int
    large_win_threshold: float
    market_type: str  # "futures" | "crypto"

    @classmethod
    def from_settings(cls, settings: Settings) -> "RiskConfig":
        return cls(
            timezone=settings.TIMEZONE,
            trading_window_start=settings.trading_window_start_time(),
            trading_window_end=settings.trading_window_end_time(),
            force_flat_time=settings.force_flat_time(),
            max_trades_per_day=settings.MAX_TRADES_PER_DAY,
            max_daily_loss=settings.MAX_DAILY_LOSS,
            max_daily_profit=settings.MAX_DAILY_PROFIT,
            max_open_positions=settings.MAX_OPEN_POSITIONS,
            cooldown_after_loss_minutes=settings.COOLDOWN_AFTER_LOSS_MINUTES,
            cooldown_after_large_win_minutes=settings.COOLDOWN_AFTER_LARGE_WIN_MINUTES,
            large_win_threshold=settings.LARGE_WIN_THRESHOLD,
            market_type=settings.MARKET_TYPE,
        )


def _allow() -> RiskDecision:
    return RiskDecision(allowed=True, rule="ok", reason="all rules passed")


def _block(rule: str, reason: str) -> RiskDecision:
    return RiskDecision(allowed=False, rule=rule, reason=reason)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    setup: Setup,
    portfolio: Portfolio,
    config: RiskConfig,
    now: datetime,
    *,
    kill_switch_tripped: bool = False,
    high_risk_news_window: bool = False,
) -> RiskDecision:
    """Return a single ``RiskDecision`` for a candidate setup.

    Rules are checked in priority order; the first failing rule wins.
    """
    # 1. Kill switch — absolute.
    if kill_switch_tripped:
        return _block("kill_switch", "Kill switch is tripped — refuse all trading.")

    # 2. High-risk news window — absolute.
    if high_risk_news_window:
        return _block(
            "high_risk_news",
            "Inside a configured high-risk news window — entries blocked.",
        )

    # 3. Trading window. For crypto, treat as 24/7 unless the user shrinks
    #    the window in settings (we still respect explicit windows).
    if not _within_trading_window(now, config):
        return _block(
            "trading_window",
            f"Outside trading window {config.trading_window_start}–{config.trading_window_end}.",
        )

    # 4. Forced-flat: don't open new entries if we're at/past the flat time.
    local_now = now.astimezone(ZoneInfo(config.timezone))
    if local_now.time() >= config.force_flat_time:
        return _block(
            "forced_flat",
            f"At/past FORCE_FLAT_TIME={config.force_flat_time}; no new entries.",
        )

    # 5. Open-positions cap + no hedging.
    if portfolio.open_position is not None:
        if config.max_open_positions <= 1:
            return _block(
                "max_open_positions",
                "Already have an open position; max_open_positions=1.",
            )
        if portfolio.open_position.direction != setup.direction:
            return _block(
                "no_hedging",
                f"Refusing to open {setup.direction} while {portfolio.open_position.direction} is open.",
            )

    # Day stats are read off the portfolio. For crypto + 24h sessions the
    # "day" notion still applies (calendar day in local TZ); the portfolio
    # rolls over via ``maybe_roll_day``.
    portfolio.maybe_roll_day(now)
    day = portfolio.day

    # 6. Max trades per day.
    if day.trades >= config.max_trades_per_day:
        return _block(
            "max_trades_per_day",
            f"Already executed {day.trades} trades today (cap={config.max_trades_per_day}).",
        )

    # 7. Daily loss cap (read as a positive number in settings; net_pnl is signed).
    if day.net_pnl <= -abs(config.max_daily_loss):
        return _block(
            "max_daily_loss",
            f"Day net PnL {day.net_pnl:.2f} <= -{config.max_daily_loss:.2f}.",
        )

    # 8. Daily profit cap.
    if day.net_pnl >= abs(config.max_daily_profit):
        return _block(
            "max_daily_profit",
            f"Day net PnL {day.net_pnl:.2f} >= {config.max_daily_profit:.2f}.",
        )

    # 9. Cooldowns (after losses, after large wins).
    cooldown_block = _cooldown_block(day.last_trade_ts, day.last_trade_net_pnl, now, config)
    if cooldown_block is not None:
        return cooldown_block

    return _allow()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _within_trading_window(now: datetime, config: RiskConfig) -> bool:
    local = now.astimezone(ZoneInfo(config.timezone)).time()
    return config.trading_window_start <= local <= config.trading_window_end


def _cooldown_block(
    last_trade_ts: Optional[datetime],
    last_trade_pnl: Optional[float],
    now: datetime,
    config: RiskConfig,
) -> Optional[RiskDecision]:
    if last_trade_ts is None or last_trade_pnl is None:
        return None

    if last_trade_pnl < 0 and config.cooldown_after_loss_minutes > 0:
        until = last_trade_ts + timedelta(minutes=config.cooldown_after_loss_minutes)
        if now < until:
            return _block(
                "cooldown_after_loss",
                f"In post-loss cooldown until {until.isoformat()}.",
            )

    if (
        last_trade_pnl >= config.large_win_threshold
        and config.cooldown_after_large_win_minutes > 0
    ):
        until = last_trade_ts + timedelta(minutes=config.cooldown_after_large_win_minutes)
        if now < until:
            return _block(
                "cooldown_after_large_win",
                f"In post-large-win cooldown until {until.isoformat()}.",
            )

    return None
