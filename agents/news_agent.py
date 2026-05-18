"""NewsAgent: pre-session and EOD news risk assessment.

The agent reads a (possibly empty) static list of headlines plus
instrument metadata. It returns a :class:`NewsAssessment` whose
``high_risk_window`` flag the orchestrator may surface to the paper loop
— **block-only** via the existing ``risk_engine.evaluate(...,
high_risk_news_window=…)`` path. The agent has no other channel into
trading behavior.
"""

from __future__ import annotations

import json
from typing import ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import NewsAssessment


class NewsAgent(BaseAgent):
    name: ClassVar[str] = "news"
    schema_class = NewsAssessment

    system_prompt: ClassVar[str] = (
        "You are a market-news risk assistant. "
        "Given an instrument and an optional list of curated headlines, "
        "decide whether the upcoming session sits inside a high-risk news window "
        "(major macro release, central-bank decision, earnings, geopolitical event). "
        "Return JSON matching the NewsAssessment schema exactly. "
        "Set high_risk_window=true only when there is a clearly material event. "
        "When unsure, prefer high_risk_window=false and severity='low'. "
        "You are read-only and cannot place trades or change risk limits."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        s = context.settings_snapshot
        headlines = context.news_headlines or []
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "instrument": context.instrument,
                    "market_type": s.get("MARKET_TYPE"),
                    "session_date": context.session_date,
                    "trading_window_start": s.get("trading_window_start"),
                    "trading_window_end": s.get("trading_window_end"),
                    "curated_headlines": headlines,
                },
                indent=2,
            )
            + "\n\nReturn JSON with fields: high_risk_window (bool), "
              "severity ('low'|'medium'|'high'), events (string list), "
              "summary (string), recommendation (string)."
        )
