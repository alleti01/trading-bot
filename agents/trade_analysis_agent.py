"""TradeAnalysisAgent: per-closed-trade narrative.

Strictly explanatory. The agent receives a deterministic
:class:`PostTradeAnalysis` plus the rule-based mistake tags and writes
a short JSON summary that fits into both the Discord ``trade.analysis``
notification and the per-trade Markdown report.

The agent **never** assigns mistake tags, modifies thresholds, or
suggests rule changes. The classifier and pattern miner are the only
places where behavior-shaping decisions are made.

Architectural property: this module imports nothing from
``execution/`` or ``risk/`` (enforced by ``tests/test_agent_isolation.py``).
"""

from __future__ import annotations

import json
from typing import ClassVar, Optional

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import TradeAnalysisSummary


class TradeAnalysisAgent(BaseAgent):
    """One-trade narrator. Mock-friendly via BaseAgent + LLMClient ABC."""

    name: ClassVar[str] = "trade_analysis"
    schema_class = TradeAnalysisSummary

    system_prompt: ClassVar[str] = (
        "You are a trade-review assistant for a paper-first algo trading bot. "
        "You receive a structured post-trade analysis (entry, exit, stop, target, "
        "model confidence, regime, mistake tags) plus the deterministic mistake "
        "tags. Write a short JSON summary explaining why the trade was taken and "
        "why it won or lost, then briefly restate the mistake tags in plain English. "
        "Add at most 4 review notes for the operator. "
        "You may NOT propose changes to strategy code, risk limits, model thresholds, "
        "or model promotion. The deterministic system handles all of that. "
        "Return JSON matching TradeAnalysisSummary exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        # The orchestrator passes the analysis dict in `daily_report` for
        # convenience (we keep AgentContext simple). ``mistake_tags`` is a
        # bare list of strings.
        analysis = (context.daily_report or {}).get("post_trade_analysis", {})
        mistake_tags = (context.daily_report or {}).get("mistake_tags", [])
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "post_trade_analysis": analysis,
                    "mistake_tags": list(mistake_tags),
                },
                indent=2,
                default=str,
            )
            + "\n\nReturn JSON with fields: trade_id (string), headline (string), "
              "why_taken (string), why_outcome (string), mistake_summary (string), "
              "review_notes (string list, max 4), "
              "confidence_in_analysis ('low'|'medium'|'high')."
        )
