"""Backtest report writer (Day 4 minimal).

Writes two artifacts to ``settings.REPORTS_DIR`` (or a caller-supplied
path):

- ``backtest_<timestamp>.json`` — machine-readable summary.
- ``backtest_<timestamp>.md``   — short Markdown summary the operator can
  open without tooling.

The full per-trade journal + per-day Markdown report is Day 6's job.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.logging_config import get_logger
from backtesting.engine import BacktestResult
from compliance.rules import ComplianceFlag, general_compliance_flags
from compliance.tradeify_rules import TradeifyFlag, tradeify_compliance_flags
from config.instruments import get_instrument
from config.settings import Settings


def _flag_to_dict(f: ComplianceFlag | TradeifyFlag) -> dict[str, Any]:
    return {
        "rule": f.rule,
        "triggered": bool(f.triggered),
        "detail": f.detail,
        "value": float(f.value),
    }


def build_report_payload(
    result: BacktestResult,
    settings: Settings,
) -> dict[str, Any]:
    spec = get_instrument(settings.INSTRUMENT)
    session_close = settings.trading_window_end_time()
    flat_time = settings.force_flat_time()

    general = general_compliance_flags(result.closed_trades)
    tradeify = tradeify_compliance_flags(
        result.closed_trades,
        market_type=settings.MARKET_TYPE,
        timezone=settings.TIMEZONE,
        session_close=session_close,
        flat_time=flat_time,
        consistency_limit_percent=settings.CONSISTENCY_LIMIT_PERCENT,
    )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "instrument": result.instrument,
        "timeframe": result.timeframe,
        "market_type": spec.market_type,
        "starting_equity": result.starting_equity,
        "n_setups_total": result.n_setups_total,
        "n_setups_filled": result.n_setups_filled,
        "n_setups_risk_blocked": result.n_setups_risk_blocked,
        "n_setups_model_rejected": result.n_setups_model_rejected,
        "metrics": result.metrics.to_dict() if result.metrics else None,
        "daily_pnl": result.daily_pnl,
        "risk_blocks_summary": _summarize_risk_blocks(result),
        "compliance": {
            "general": [_flag_to_dict(f) for f in general],
            "tradeify": [_flag_to_dict(f) for f in tradeify],
        },
        "config": {
            "max_trades_per_day": settings.MAX_TRADES_PER_DAY,
            "max_daily_loss": settings.MAX_DAILY_LOSS,
            "max_daily_profit": settings.MAX_DAILY_PROFIT,
            "max_position_size": settings.MAX_POSITION_SIZE,
            "risk_per_trade": settings.RISK_PER_TRADE,
            "slippage_ticks": settings.SLIPPAGE_TICKS,
            "commission_per_contract": settings.COMMISSION_PER_CONTRACT,
            "force_flat_time": str(flat_time),
            "trading_window_start": str(settings.trading_window_start_time()),
            "trading_window_end": str(session_close),
            "max_hold_bars": settings.MAX_HOLD_BARS,
            "consistency_limit_percent": settings.CONSISTENCY_LIMIT_PERCENT,
        },
    }
    return payload


def _summarize_risk_blocks(result: BacktestResult) -> dict[str, int]:
    by_rule: dict[str, int] = {}
    for rb in result.risk_blocks:
        by_rule[rb.rule] = by_rule.get(rb.rule, 0) + 1
    return by_rule


def render_markdown(payload: dict[str, Any]) -> str:
    m = payload.get("metrics") or {}
    risk_summary = payload.get("risk_blocks_summary") or {}
    general = payload["compliance"]["general"]
    tradeify = payload["compliance"]["tradeify"]

    lines: list[str] = []
    lines.append(f"# Backtest report — {payload['instrument']} ({payload['timeframe']})")
    lines.append("")
    lines.append(f"_Generated: {payload['generated_at']} UTC_")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    if not m:
        lines.append("- No trades placed; no metrics computed.")
    else:
        lines.extend(
            [
                f"- Trades: **{m['n_trades']}** (wins {m['n_wins']} / losses {m['n_losses']} / be {m['n_breakevens']})",
                f"- Win rate: **{m['win_rate']:.2%}**",
                f"- Net PnL: **${m['net_pnl']:.2f}** "
                f"(gross ${m['gross_pnl']:.2f}, commissions ${m['total_commission']:.2f})",
                f"- Expectancy / trade: **${m['expectancy_per_trade']:.2f}**",
                f"- Profit factor: **{m['profit_factor']:.2f}**",
                f"- Max drawdown: **${m['max_drawdown_dollars']:.2f}** "
                f"({m['max_drawdown_pct']:.2%})",
                f"- Sharpe (per-trade): **{m['sharpe_per_trade']:.3f}**",
                f"- Avg bars held: **{m['avg_bars_held']:.2f}**",
                f"- Equity: ${m['starting_equity']:.2f} → ${m['ending_equity']:.2f}",
            ]
        )
    lines.append("")
    lines.append("## Setup pipeline")
    lines.append("")
    lines.append(f"- Detected: {payload['n_setups_total']}")
    lines.append(f"- Model-rejected: {payload['n_setups_model_rejected']}")
    lines.append(f"- Risk-blocked: {payload['n_setups_risk_blocked']}")
    lines.append(f"- Filled: {payload['n_setups_filled']}")
    lines.append("")
    if risk_summary:
        lines.append("### Risk blocks by rule")
        lines.append("")
        for rule, n in sorted(risk_summary.items(), key=lambda kv: -kv[1]):
            lines.append(f"- {rule}: {n}")
        lines.append("")

    lines.append("## Compliance")
    lines.append("")
    for f in general + tradeify:
        marker = "❌" if f["triggered"] else "✅"
        lines.append(f"- {marker} `{f['rule']}` — {f['detail']}")
    lines.append("")

    if payload.get("daily_pnl"):
        lines.append("## Daily PnL")
        lines.append("")
        lines.append("| Date | Trades | W | L | Gross | Net |")
        lines.append("|------|-------:|--:|--:|------:|----:|")
        for row in payload["daily_pnl"]:
            lines.append(
                f"| {row['date']} | {row['trades']} | {row['wins']} | {row['losses']} "
                f"| {row['gross_pnl']:.2f} | {row['net_pnl']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_backtest_report(
    result: BacktestResult,
    settings: Settings,
    *,
    out_dir: Path | None = None,
) -> tuple[Path, Path]:
    out_dir = Path(out_dir) if out_dir is not None else Path(settings.REPORTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_report_payload(result, settings)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"backtest_{ts}.json"
    md_path = out_dir / f"backtest_{ts}.md"

    json_path.write_text(json.dumps(payload, indent=2, default=str))
    md_path.write_text(render_markdown(payload))

    log = get_logger("reports.backtest_report")
    log.info(
        "report.written",
        json_path=str(json_path),
        md_path=str(md_path),
        n_trades=payload["metrics"]["n_trades"] if payload["metrics"] else 0,
    )
    return json_path, md_path
