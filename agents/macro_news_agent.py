"""MacroNewsAgent: web-grounded macro / news risk research.

Defaults to the Perplexity provider (configured via the per-agent
router) so it can pull live event calendars, central-bank statements,
and headlines. The provider is *not* hard-coded — operators can
re-route via ``MACRO_NEWS_AGENT_PROVIDER`` in ``.env``.

The agent is strictly **block-only**: it can flag a window as risky
or recommend reduced size, never approve or place trades. The
orchestrator surfaces ``risk_level == "high"`` (or any
``blocked_windows``) by flipping the existing
``high_risk_news_active`` flag, which feeds into
``risk_engine.evaluate(..., high_risk_news_window=...)`` — that
engine path is the only bridge from the agent layer back into
trading, and it can only refuse trades.

Architectural property: this module imports nothing from
``execution/`` or ``risk/`` (enforced by
``tests/test_agent_isolation.py``).
"""

from __future__ import annotations

import json
from typing import ClassVar

from agents.base_agent import AgentContext, BaseAgent
from agents.schemas import MacroNewsAssessment


class MacroNewsAgent(BaseAgent):
    name: ClassVar[str] = "macro_news"
    schema_class = MacroNewsAssessment

    system_prompt: ClassVar[str] = (
        "You are a macro / news risk researcher for a paper-first algo "
        "trading bot. Use only the inputs and your retrieval to identify "
        "scheduled macro releases, central-bank decisions, geopolitical "
        "events, and earnings prints that may impact the operator's "
        "enabled symbols today.\n\n"
        "Output rules:\n"
        "- You may recommend BLOCKING or REDUCING risk, never approving "
        "trades.\n"
        "- ``affected_symbols`` MUST be a subset of the operator's "
        "enabled_symbols. Do not invent symbols.\n"
        "- ``blocked_windows`` are advisory: each entry should describe "
        "a contiguous window (start/end) when entries should pause and "
        "explain why.\n"
        "- ``sources`` should list any URLs you used; we treat them as "
        "audit metadata, not authoritative.\n"
        "- When unsure, prefer ``risk_level='low'`` and an empty "
        "blocked_windows list. False positives waste a session.\n"
        "Return JSON matching MacroNewsAssessment exactly."
    )

    def build_user_prompt(self, context: AgentContext) -> str:
        s = context.settings_snapshot
        enabled_symbols = list(context.enabled_symbols) or [context.instrument]
        return (
            "Inputs (JSON):\n"
            + json.dumps(
                {
                    "session_date": context.session_date,
                    "primary_instrument": context.instrument,
                    "enabled_symbols": enabled_symbols,
                    "market_type": s.get("MARKET_TYPE"),
                    "timezone": s.get("TIMEZONE"),
                    "trading_window_start": s.get("trading_window_start"),
                    "trading_window_end": s.get("trading_window_end"),
                    "curated_headlines": list(context.news_headlines),
                },
                indent=2,
            )
            + "\n\nReturn JSON with fields: risk_level "
              "('low'|'medium'|'high'), affected_symbols (string list, "
              "subset of enabled_symbols), blocked_windows (list of "
              "{start, end, reason, severity}), key_events (string list), "
              "sources (list of {url, title?, snippet?}), summary (string)."
        )
