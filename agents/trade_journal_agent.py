"""TradeJournalAgent: narrative commentary on today's closed trades.

Reads the ``trades`` array from the daily report payload (which already
matches what ``reports/trade_journal.py`` exports as CSV). Produces
short narrative bullets — *not* a row-by-row table; the deterministic
report and CSV journal already cover that.
"""

from __future__ import annotations

import json
from typing import ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import TradeJournalNarrative


class TradeJournalAgent(BaseAgent):
    name: ClassVar[str] = "trade_journal"
    schema_class = TradeJournalNarrative

    system_prompt: ClassVar[str] = (
        "You are a trade-review journalist. "
        "Given today's closed trades (entry, exit, reason, PnL), "
        "summarize highlights, mistakes, and lessons in concise bullets. "
        "Identify the best and worst trade by setup_id when available. "
        "Do not recommend changing strategy code or thresholds. "
        "Return JSON matching TradeJournalNarrative exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        # Cap the trade list size so we don't blow the prompt budget.
        trades = (context.daily_report or {}).get("trades", []) or []
        trimmed = []
        for t in trades[:50]:
            trimmed.append(
                {
                    "setup_id": t.get("setup_id"),
                    "direction": t.get("direction"),
                    "entry_ts": t.get("entry_ts"),
                    "exit_ts": t.get("exit_ts"),
                    "entry_price": t.get("entry_price"),
                    "exit_price": t.get("exit_price"),
                    "exit_reason": t.get("exit_reason"),
                    "net_pnl": t.get("net_pnl"),
                    "bars_held": t.get("bars_held"),
                }
            )
        metrics = (context.daily_report or {}).get("metrics") or {}
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "session_date": context.session_date,
                    "instrument": context.instrument,
                    "metrics": {
                        k: metrics.get(k)
                        for k in (
                            "n_trades",
                            "n_wins",
                            "n_losses",
                            "net_pnl",
                            "expectancy_per_trade",
                        )
                    },
                    "trades": trimmed,
                },
                indent=2,
            )
            + "\n\nReturn JSON with fields: session_date (string), "
              "highlights (list), mistakes (list), lessons (list), "
              "best_trade_setup_id (string|null), worst_trade_setup_id (string|null)."
        )
