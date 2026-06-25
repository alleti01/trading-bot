"""Live paper-account evaluation snapshot (broker-sourced).

The intraday loop places orders on Alpaca paper, so the source of truth
for live results is the broker account — not the local ``closed_trades``
table (which only holds local-paper-executor fills). This builds a
snapshot of account equity, day P&L, open positions, and working orders,
writes a dated markdown report, and returns a JSON-safe payload for
Discord.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from config.settings import Settings

_log = get_logger("reports.live_paper")


def build_live_paper_report(
    settings: Settings,
    *,
    broker=None,  # noqa: ANN001 — optional BaseBroker (built if None)
    out_dir: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> dict[str, Any]:
    """Pull broker state and write a live paper evaluation snapshot."""
    now = now or datetime.now(tz=timezone.utc)
    if broker is None:
        from integrations.broker_router import build_broker

        broker = build_broker(settings)

    payload: dict[str, Any] = {
        "as_of": now.isoformat(),
        "provider": getattr(broker, "provider_name", "unknown"),
        "ok": True,
    }
    try:
        acct = broker.get_account()
        payload["account"] = {
            "account_id": acct.account_id,
            "equity": acct.equity,
            "cash": acct.cash_balance,
            "buying_power": acct.buying_power,
            "day_pnl": acct.realized_pnl,
        }
        positions = broker.get_positions()
        payload["positions"] = [
            {
                "symbol": p.symbol,
                "qty": p.quantity,
                "direction": p.direction,
                "avg_price": p.average_price,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ]
        orders = broker.get_open_orders()
        payload["open_orders"] = [
            {"symbol": o.symbol, "side": o.side, "qty": o.quantity, "type": o.order_type}
            for o in orders
        ]
        payload["n_positions"] = len(positions)
        payload["n_open_orders"] = len(orders)
    except Exception as e:  # noqa: BLE001
        payload["ok"] = False
        payload["error"] = str(e)
        _log.error("live_paper_report.broker_failed", error=str(e))

    out_dir = out_dir or (Path(settings.REPORTS_DIR) / "live_paper")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"live_{now.strftime('%Y-%m-%d')}.md"
    path.write_text(_render_markdown(payload), encoding="utf-8")
    payload["report_path"] = str(path)
    _log.info("live_paper_report.written", path=str(path), ok=payload["ok"])
    return payload


def _render_markdown(p: dict[str, Any]) -> str:
    lines = [
        "# Live Paper Evaluation Snapshot",
        "",
        f"**As of:** {p.get('as_of')}",
        f"**Broker:** {p.get('provider')}",
        "",
    ]
    if not p.get("ok"):
        lines += ["## Broker error", f"```\n{p.get('error')}\n```"]
        return "\n".join(lines)

    acct = p.get("account", {})
    lines += [
        "## Account",
        f"- Equity: ${acct.get('equity', 0):,.2f}",
        f"- Day P&L: ${acct.get('day_pnl', 0):,.2f}",
        f"- Cash: ${acct.get('cash', 0):,.2f}",
        f"- Buying power: ${acct.get('buying_power', 0):,.2f}",
        "",
        f"## Open positions ({p.get('n_positions', 0)})",
    ]
    for pos in p.get("positions", []):
        lines.append(
            f"- {pos['symbol']} {pos['direction']} {pos['qty']} @ "
            f"${pos['avg_price']:.2f}  (uPnL ${pos.get('unrealized_pnl', 0):.2f})"
        )
    lines += ["", f"## Working orders ({p.get('n_open_orders', 0)})"]
    for o in p.get("open_orders", []):
        lines.append(f"- {o['symbol']} {o['side']} {o['qty']} ({o['type']})")
    return "\n".join(lines)


def discord_summary_lines(payload: dict[str, Any]) -> list[str]:
    """≤10-line Discord-friendly summary of the live paper snapshot."""
    if not payload.get("ok"):
        return [f"Paper report error: {payload.get('error', 'unknown')}"]
    acct = payload.get("account", {})
    lines = [
        f"Paper account ({payload.get('provider')})",
        f"Equity ${acct.get('equity', 0):,.0f} | Day P&L ${acct.get('day_pnl', 0):,.2f}",
        f"Open positions: {payload.get('n_positions', 0)} | Working orders: {payload.get('n_open_orders', 0)}",
    ]
    for pos in payload.get("positions", [])[:5]:
        lines.append(
            f"  {pos['symbol']} {pos['direction']} {pos['qty']} "
            f"(uPnL ${pos.get('unrealized_pnl', 0):.2f})"
        )
    return lines[:10]
