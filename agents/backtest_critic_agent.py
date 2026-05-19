"""BacktestCriticAgent: review backtest reports and surface weak spots.

Reads a backtest summary (metrics + per-bucket breakdowns) plus the
per-symbol numbers from the latest daily report. Produces a structured
critique that operators can act on by **scheduling experiments** —
never by editing strategy code or risk caps directly.

Design properties:

- The schema (:class:`BacktestCritique`) only allows experiment-shaped
  recommendations. It has no field for "change parameter X to Y";
  that's intentional.
- The agent is provider-agnostic — it goes through the
  :class:`~agents.providers.router.ProviderRouter` like every other
  agent. OpenAI is the default for reasoning workloads;
  ``BACKTEST_CRITIC_AGENT_PROVIDER`` can override that.
- Imports nothing from ``execution/`` or ``risk/`` (enforced by
  ``tests/test_agent_isolation.py``).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import BacktestCritique


class BacktestCriticAgent(BaseAgent):
    name: ClassVar[str] = "backtest_critic"
    schema_class = BacktestCritique

    system_prompt: ClassVar[str] = (
        "You are a backtest reviewer for a paper-first algo trading bot. "
        "Given the bot's most recent backtest summary plus a recent live "
        "(paper) snapshot, identify weak spots. Categories you may use:\n"
        "- ``time_window``: hours / parts of the session that produce "
        "  poor expectancy.\n"
        "- ``symbol``: instruments that under-perform the average.\n"
        "- ``confidence_bucket``: ranges of model probability where "
        "  expectancy / win rate falls off.\n"
        "- ``regime``: trend / volatility regimes where the strategy "
        "  struggles.\n"
        "- ``other``: anything else, but use sparingly.\n\n"
        "OUTPUT RULES:\n"
        "- Every ``suggested_experiment`` MUST be an experiment, not an "
        "instruction. Examples of good shape: 'walk-forward with this "
        "filter on/off', 'bucket scores at 0.55/0.65/0.75 and report "
        "expectancy', 'add a session-time filter and compare equity '"
        "'curves'. Do NOT recommend code changes, parameter values, "
        "risk-cap edits, threshold changes, or model promotions.\n"
        "- Cite numbers from the inputs whenever possible. Do not invent "
        "metrics.\n"
        "- Keep ``where`` short and concrete (e.g. '09:30-10:00 ET', "
        "  'MNQ', 'p in [0.50, 0.60)', 'low_vol regime').\n"
        "- Return JSON matching BacktestCritique exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        # ``backtest_summary`` is supplied by the orchestrator caller —
        # typically the JSON written next to the latest backtest
        # Markdown report. ``paper_metrics`` is the recent
        # ``build_daily_report_payload`` output; we trim it to the
        # fields the critic actually needs so the prompt budget is
        # spent on signal, not noise.
        backtest = context.backtest_summary or {}
        paper = context.paper_metrics or (context.daily_report or {}).get("metrics") or {}
        by_symbol = (context.daily_report or {}).get("by_symbol") or []
        per_symbol_paper: list[dict[str, Any]] = []
        for row in by_symbol[:32]:
            per_symbol_paper.append(
                {
                    "symbol": row.get("symbol"),
                    "trades": row.get("trades"),
                    "wins": row.get("wins"),
                    "losses": row.get("losses"),
                    "net_pnl": row.get("net_pnl"),
                    "profit_factor": row.get("profit_factor"),
                    "expectancy": row.get("expectancy"),
                }
            )
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "session_date": context.session_date,
                    "instrument": context.instrument,
                    "backtest_summary": backtest,
                    "paper_metrics": {
                        k: paper.get(k)
                        for k in (
                            "n_trades",
                            "n_wins",
                            "n_losses",
                            "win_rate",
                            "net_pnl",
                            "profit_factor",
                            "expectancy_per_trade",
                            "max_drawdown_dollars",
                        )
                    },
                    "per_symbol_paper": per_symbol_paper,
                },
                indent=2,
                default=str,
            )
            + "\n\nReturn JSON with fields: overall_assessment (string), "
              "weak_spots (list of {category, where, severity, evidence, "
              "suggested_experiment}), bad_time_windows (string list), "
              "weak_symbols (string list), bad_confidence_buckets (string "
              "list), bad_regimes (string list), suggested_experiments "
              "(string list)."
        )
