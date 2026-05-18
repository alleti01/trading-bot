"""ReportAgent: executive-summary commentary for the daily report.

Reads the full ``build_daily_report_payload`` dict and emits a one-line
headline plus short bullets and a "tomorrow focus" string. The
orchestrator can append the result to the daily Markdown as an
``## Agent commentary`` section so a human reader sees both the
deterministic numbers and the LLM gloss in the same artifact.
"""

from __future__ import annotations

import json
from typing import ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import ReportCommentary


class ReportAgent(BaseAgent):
    name: ClassVar[str] = "report"
    schema_class = ReportCommentary

    system_prompt: ClassVar[str] = (
        "You are a trading-desk operator's morning-briefing writer. "
        "Given today's daily report payload (metrics, compliance flags, "
        "risk-block summary, trade list), write a one-line headline and "
        "3-7 short bullets. Mention compliance failures explicitly. "
        "Always include a 'tomorrow_focus' sentence with concrete habits "
        "(e.g. 'review losses before 10am') — never code or threshold changes. "
        "Return JSON matching ReportCommentary exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        report = context.daily_report or {}
        metrics = report.get("metrics") or {}
        compliance = report.get("compliance") or {}
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
                            "win_rate",
                            "net_pnl",
                            "profit_factor",
                            "expectancy_per_trade",
                            "max_drawdown_dollars",
                        )
                    },
                    "compliance_flags": compliance,
                    "risk_blocks_summary": report.get("risk_blocks_summary", {}),
                    "n_trades": metrics.get("n_trades", 0),
                },
                indent=2,
            )
            + "\n\nReturn JSON with fields: session_date (string), "
              "headline (string), bullets (list), compliance_notes (list), "
              "tomorrow_focus (string)."
        )
