"""StrategyResearchAgent: web-grounded ideation for new strategies / filters.

Reads the recent backtest summary and the operator's enabled symbols,
then asks a web-grounded provider (Perplexity by default — operators
can re-route via ``STRATEGY_RESEARCH_AGENT_PROVIDER``) to surface
candidate experiment ideas.

Output is a :class:`StrategyResearchReport` whose recommendations are
**experiment-shaped** (hypothesis + plan + risks). The schema does
not allow code or threshold changes; the orchestrator never wires
the agent's output into anything except the report and the persisted
``agent_outputs`` row.

Architectural property: this module imports nothing from
``execution/`` or ``risk/`` (enforced by
``tests/test_agent_isolation.py``).
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import StrategyResearchReport


class StrategyResearchAgent(BaseAgent):
    name: ClassVar[str] = "strategy_research"
    schema_class = StrategyResearchReport

    system_prompt: ClassVar[str] = (
        "You are a research assistant for an algo trading bot. Survey "
        "publicly-known strategy ideas, filters, and indicators that "
        "may complement the bot's current configuration, and propose "
        "small, testable experiments.\n\n"
        "OUTPUT RULES (strict):\n"
        "- Every experiment MUST be advisory and experiment-shaped: "
        "  hypothesis, experiment_plan, and explicit risks. Suggested "
        "  output is the SHAPE of an experiment, not its result.\n"
        "- You MUST NOT write code, propose specific parameter values, "
        "  or instruct the operator to change risk caps, model "
        "  thresholds, or strategy rules. Treat 'try' / 'consider' as "
        "  the strongest verbs you may use.\n"
        "- You MUST NOT promote any model or claim the bot should "
        "  switch strategies live. Promotion is a deterministic, "
        "  walk-forward gated workflow that you cannot trigger.\n"
        "- ``related_filters`` should be short, generic names "
        "  (e.g. 'session_time', 'atr_regime'), not code.\n"
        "- Cite your sources when you used retrieval; the consumer "
        "  treats them as audit metadata only.\n"
        "Return JSON matching StrategyResearchReport exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        backtest = context.backtest_summary or {}
        # Trim to the high-signal fields. The agent does not need the
        # full per-trade list; aggregate metrics + per-symbol numbers
        # are enough to seed an experiment list.
        trimmed_backtest: dict[str, Any] = {
            k: backtest.get(k)
            for k in (
                "strategy",
                "metrics",
                "per_symbol",
                "best_symbol",
                "worst_symbol",
                "regime_breakdown",
                "confidence_buckets",
            )
            if k in (backtest or {})
        }
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "session_date": context.session_date,
                    "primary_instrument": context.instrument,
                    "enabled_symbols": list(context.enabled_symbols)
                    or [context.instrument],
                    "market_type": context.settings_snapshot.get("MARKET_TYPE"),
                    "backtest_summary": trimmed_backtest,
                },
                indent=2,
                default=str,
            )
            + "\n\nReturn JSON with fields: summary (string), "
              "experiments (list of {title, hypothesis, "
              "experiment_plan, risks (list), related_filters (list)}), "
              "sources (list of {url, title?, snippet?})."
        )
