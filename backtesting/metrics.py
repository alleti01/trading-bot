"""Backtest performance metrics.

All metrics consume the trade ledger + equity curve produced by
``backtesting.portfolio.Portfolio``. Nothing in here decides whether a
trade is "good" — that's the strategy/model/risk layer. This module
only summarizes outcomes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import sqrt
from typing import Optional

from backtesting.portfolio import ClosedTradeRecord


@dataclass
class BacktestMetrics:
    n_trades: int
    n_wins: int
    n_losses: int
    n_breakevens: int
    win_rate: float
    gross_pnl: float
    net_pnl: float
    total_commission: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    expectancy_per_trade: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    avg_bars_held: float
    sharpe_per_trade: float
    starting_equity: float
    ending_equity: float
    by_month: dict = field(default_factory=dict)
    by_strategy: dict = field(default_factory=dict)
    by_direction: dict = field(default_factory=dict)
    by_exit_reason: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _max_drawdown(equity_curve: list[tuple[datetime, float]], starting_equity: float) -> tuple[float, float]:
    if not equity_curve:
        return 0.0, 0.0
    peak = max(starting_equity, equity_curve[0][1])
    max_dd_dollars = 0.0
    max_dd_pct = 0.0
    for _, eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd_dollars:
            max_dd_dollars = dd
            max_dd_pct = (dd / peak) if peak > 0 else 0.0
    return max_dd_dollars, max_dd_pct


def _sharpe_per_trade(trade_pnls: list[float]) -> float:
    """Simple per-trade Sharpe: mean / std. No annualization here — the
    timeframe is irrelevant per-trade and varies by strategy."""
    n = len(trade_pnls)
    if n < 2:
        return 0.0
    mean = sum(trade_pnls) / n
    var = sum((p - mean) ** 2 for p in trade_pnls) / (n - 1)
    sd = sqrt(var)
    return float(mean / sd) if sd > 0 else 0.0


def _bucketize(
    trades: list[ClosedTradeRecord],
    key_fn,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for t in trades:
        key = str(key_fn(t))
        b = out.setdefault(key, {"n_trades": 0, "wins": 0, "losses": 0, "net_pnl": 0.0})
        b["n_trades"] += 1
        if t.net_pnl > 0:
            b["wins"] += 1
        elif t.net_pnl < 0:
            b["losses"] += 1
        b["net_pnl"] += t.net_pnl
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_metrics(
    trades: list[ClosedTradeRecord],
    equity_curve: list[tuple[datetime, float]],
    *,
    starting_equity: float,
) -> BacktestMetrics:
    n = len(trades)
    if n == 0:
        return BacktestMetrics(
            n_trades=0,
            n_wins=0,
            n_losses=0,
            n_breakevens=0,
            win_rate=0.0,
            gross_pnl=0.0,
            net_pnl=0.0,
            total_commission=0.0,
            profit_factor=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            expectancy_per_trade=0.0,
            max_drawdown_dollars=0.0,
            max_drawdown_pct=0.0,
            avg_bars_held=0.0,
            sharpe_per_trade=0.0,
            starting_equity=starting_equity,
            ending_equity=starting_equity,
        )

    wins = [t for t in trades if t.net_pnl > 0]
    losses = [t for t in trades if t.net_pnl < 0]
    breakevens = n - len(wins) - len(losses)

    gross_pnl = sum(t.gross_pnl for t in trades)
    net_pnl = sum(t.net_pnl for t in trades)
    total_commission = sum(t.commission for t in trades)
    avg_win = sum(t.net_pnl for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.net_pnl for t in losses) / len(losses) if losses else 0.0

    profit_sum = sum(t.net_pnl for t in wins)
    loss_sum = -sum(t.net_pnl for t in losses)
    profit_factor = float(profit_sum / loss_sum) if loss_sum > 0 else (float("inf") if profit_sum > 0 else 0.0)

    expectancy = net_pnl / n
    max_dd_dollars, max_dd_pct = _max_drawdown(equity_curve, starting_equity)
    avg_bars_held = sum(t.bars_held for t in trades) / n
    sharpe = _sharpe_per_trade([t.net_pnl for t in trades])
    ending_equity = starting_equity + net_pnl

    by_month = _bucketize(trades, lambda t: t.entry_ts.strftime("%Y-%m"))
    by_strategy = _bucketize(trades, lambda t: t.instrument)  # filled later if strategy_name carried
    by_direction = _bucketize(trades, lambda t: t.direction)
    by_exit_reason = _bucketize(trades, lambda t: t.exit_reason)

    return BacktestMetrics(
        n_trades=n,
        n_wins=len(wins),
        n_losses=len(losses),
        n_breakevens=breakevens,
        win_rate=len(wins) / n if n else 0.0,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        total_commission=total_commission,
        profit_factor=profit_factor if profit_factor != float("inf") else 0.0,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_per_trade=expectancy,
        max_drawdown_dollars=max_dd_dollars,
        max_drawdown_pct=max_dd_pct,
        avg_bars_held=avg_bars_held,
        sharpe_per_trade=sharpe,
        starting_equity=starting_equity,
        ending_equity=ending_equity,
        by_month=by_month,
        by_strategy=by_strategy,
        by_direction=by_direction,
        by_exit_reason=by_exit_reason,
    )


def daily_pnl_table(trades: list[ClosedTradeRecord]) -> list[dict]:
    """Return rows of ``{date, trades, wins, losses, gross_pnl, net_pnl}``."""
    by_day: dict[str, dict] = {}
    for t in trades:
        d = t.exit_ts.date().isoformat()
        row = by_day.setdefault(
            d,
            {"date": d, "trades": 0, "wins": 0, "losses": 0, "gross_pnl": 0.0, "net_pnl": 0.0},
        )
        row["trades"] += 1
        if t.net_pnl > 0:
            row["wins"] += 1
        elif t.net_pnl < 0:
            row["losses"] += 1
        row["gross_pnl"] += t.gross_pnl
        row["net_pnl"] += t.net_pnl
    return [by_day[d] for d in sorted(by_day)]
