"""RiskExplainerAgent: plain-English summary of today's risk blocks.

Reads ``risk_blocks_summary`` (counts by rule) plus configured risk caps
from the daily report payload. Emits operator-facing explanations and
optional review actions.

The agent never recommends raising or removing a risk rule. It only
explains what happened and what an operator might investigate. Real
config changes are out of scope for the agent layer.
"""

from __future__ import annotations

import json
from typing import ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import RiskExplainerOutput


class RiskExplainerAgent(BaseAgent):
    name: ClassVar[str] = "risk_explainer"
    schema_class = RiskExplainerOutput

    system_prompt: ClassVar[str] = (
        "You are a risk-policy explainer. "
        "Given a count of how many trade attempts each risk rule blocked today, "
        "plus the active caps, write a short plain-English explanation per rule "
        "and an overall assessment of the day's risk posture. "
        "Suggest only review actions for the operator (e.g. 'review losses before "
        "10am'); never propose modifying the risk rules themselves. "
        "Return JSON matching RiskExplainerOutput exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        report = context.daily_report or {}
        risk_summary = report.get("risk_blocks_summary", {})
        config = report.get("config", {})
        # Include up to 5 sample reasons per rule from the raw block list to
        # ground the explanations.
        samples_by_rule: dict[str, list[str]] = {}
        for block in report.get("risk_blocks", []) or []:
            rule = block.get("rule")
            if not rule:
                continue
            samples_by_rule.setdefault(rule, [])
            if len(samples_by_rule[rule]) < 5 and block.get("reason"):
                samples_by_rule[rule].append(str(block["reason"]))
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "session_date": context.session_date,
                    "instrument": context.instrument,
                    "risk_blocks_summary": risk_summary,
                    "sample_reasons_by_rule": samples_by_rule,
                    "active_caps": {
                        k: config.get(k)
                        for k in (
                            "max_trades_per_day",
                            "max_daily_loss",
                            "max_daily_profit",
                            "max_position_size",
                            "risk_per_trade",
                            "force_flat_time",
                        )
                    },
                },
                indent=2,
            )
            + "\n\nReturn JSON with fields: session_date (string), "
              "blocks (list of {rule, count, explanation}), "
              "overall_assessment (string), operator_actions (string list)."
        )
