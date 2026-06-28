"""Live paper-account evaluation snapshot (broker-sourced).

The intraday loop places orders on Alpaca paper, so the source of truth
for live results is the broker account — not the local ``closed_trades``
table (which only holds local-paper-executor fills). This builds a
snapshot of account equity, day P&L, open positions, and working orders,
writes a dated markdown report, and returns a JSON-safe payload for
Discord.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.logging_config import get_logger
from config.settings import Settings

_log = get_logger("reports.live_paper")


def _open_options_summary(settings: Settings, now: datetime) -> dict[str, Any]:
    """Summarize open option positions from the manager's state file.

    Reads the JSON directly (no broker/executor/network) so it is safe to
    call from any report path. Realized option P&L is already reflected in
    the account's day P&L; this surfaces the *open* option book separately.
    """
    summary: dict[str, Any] = {"enabled": True, "n_open": 0, "open": []}
    state_path = Path(getattr(settings, "OPTIONS_STATE_PATH", "data/options/positions.json"))
    if not state_path.exists():
        return summary
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        _log.warning("live_paper_report.options_state_unreadable", error=str(e))
        return summary

    rows: list[dict[str, Any]] = []
    total_upnl = 0.0
    have_upnl = False
    for occ, pos in (data.get("positions") or {}).items():
        entry = float(pos.get("entry_price") or 0.0)
        qty = int(pos.get("qty") or 0)
        current = pos.get("current_price")
        upnl = None
        upnl_pct = None
        if current is not None:
            upnl = round((float(current) - entry) * 100.0 * qty, 2)
            upnl_pct = round(((float(current) - entry) / entry * 100.0), 2) if entry else None
            total_upnl += upnl
            have_upnl = True
        dte = None
        try:
            dte = (date.fromisoformat(str(pos.get("expiry"))) - now.date()).days
        except Exception:  # noqa: BLE001
            pass
        rows.append(
            {
                "occ": occ,
                "underlying": pos.get("underlying", ""),
                "type": pos.get("option_type", ""),
                "qty": qty,
                "entry_price": entry,
                "current_price": current,
                "unrealized_pnl": upnl,
                "unrealized_pnl_pct": upnl_pct,
                "dte": dte,
            }
        )
    summary["open"] = rows
    summary["n_open"] = len(rows)
    summary["unrealized_pnl"] = round(total_upnl, 2) if have_upnl else None
    return summary


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

    # Options book (independent of broker call so it shows even if the
    # broker snapshot failed). Realized option P&L is part of day P&L above.
    if getattr(settings, "OPTIONS_ENABLED", False):
        payload["options"] = _open_options_summary(settings, now)

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

    opts = p.get("options")
    if opts is not None:
        upnl = opts.get("unrealized_pnl")
        upnl_txt = f"  (unrealized ${upnl:,.2f})" if upnl is not None else ""
        lines += ["", f"## Open options ({opts.get('n_open', 0)}){upnl_txt}"]
        for o in opts.get("open", []):
            up = o.get("unrealized_pnl")
            up_txt = f"uPnL ${up:,.2f}" if up is not None else "uPnL n/a"
            dte = o.get("dte")
            dte_txt = f"{dte}DTE" if dte is not None else "?DTE"
            lines.append(
                f"- {o['underlying']} {o['type']} x{o['qty']} "
                f"@ ${o['entry_price']:.2f} ({dte_txt}, {up_txt})"
            )
    return "\n".join(lines)


def _options_line(payload: dict[str, Any]) -> Optional[str]:
    opts = payload.get("options")
    if not opts:
        return None
    upnl = opts.get("unrealized_pnl")
    upnl_txt = f" | uPnL ${upnl:,.2f}" if upnl is not None else ""
    return f"Open options: {opts.get('n_open', 0)}{upnl_txt}"


def discord_summary_lines(payload: dict[str, Any]) -> list[str]:
    """≤10-line Discord-friendly summary of the live paper snapshot."""
    opt_line = _options_line(payload)
    if not payload.get("ok"):
        out = [f"Paper report error: {payload.get('error', 'unknown')}"]
        if opt_line:
            out.append(opt_line)
        return out
    acct = payload.get("account", {})
    lines = [
        f"Paper account ({payload.get('provider')})",
        f"Equity ${acct.get('equity', 0):,.0f} | Day P&L ${acct.get('day_pnl', 0):,.2f}",
        f"Open positions: {payload.get('n_positions', 0)} | Working orders: {payload.get('n_open_orders', 0)}",
    ]
    if opt_line:
        lines.append(opt_line)
    for pos in payload.get("positions", [])[:4]:
        lines.append(
            f"  {pos['symbol']} {pos['direction']} {pos['qty']} "
            f"(uPnL ${pos.get('unrealized_pnl', 0):.2f})"
        )
    return lines[:10]
