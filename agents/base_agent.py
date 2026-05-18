"""Shared agent run loop.

Every concrete agent inherits from :class:`BaseAgent` and implements two
hooks:

- ``build_user_prompt(context) -> str`` — the data-only prompt body.
- ``schema_class`` (class attr)         — the Pydantic schema the LLM
  output must match.

:meth:`BaseAgent.run` does the rest: prepend the ``agent=<name>`` marker
that :class:`MockLLMClient` keys off, send the prompts, strip optional
markdown fences, validate against the schema, and wrap everything in an
:class:`AgentResult`.

Critical safety property: ``run()`` **never raises**. Any error (network,
JSON, schema, attribute) is captured into ``AgentResult(schema_valid=False,
error=…)`` so the orchestrator can persist the failure for audit and
keep going.

Architectural property: this module imports nothing from ``execution`` or
``risk`` — checked by ``tests/test_agent_isolation.py``.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date as _date
from typing import Any, ClassVar, Optional

from pydantic import BaseModel, ValidationError

from app.logging_config import get_logger
from agents.llm_client import LLMClient, LLMClientError
from agents.schemas import AgentResult


# ---------------------------------------------------------------------------
# Agent context — read-only snapshot the orchestrator hands to every agent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentContext:
    """Read-only data the agents reason about.

    Built once per orchestrator call from DB rows + the daily report
    JSON payload. Agents must not mutate this object.

    Fields:
        settings_snapshot:  ``{INSTRUMENT, MARKET_TYPE, TIMEZONE, MODE,
                              risk caps, ...}`` — frozen subset.
        session_date:       Local-tz session date (``YYYY-MM-DD`` string
                            so prompts are stable).
        instrument:         Convenience alias.
        trades:             List of dicts (the ``trades`` array from
                            ``build_daily_report_payload``).
        risk_blocks:        List of dicts (``rule``, ``count``, sample
                            reasons).
        daily_report:       Full ``build_daily_report_payload`` output.
        model_metadata:     ``{"name", "version", "metrics", ...}`` or
                            ``None`` if no model was used.
        news_headlines:     Optional curated headline list (default []).
    """

    settings_snapshot: dict[str, Any]
    session_date: str
    instrument: str
    trades: list[dict[str, Any]] = field(default_factory=list)
    risk_blocks: list[dict[str, Any]] = field(default_factory=list)
    daily_report: dict[str, Any] = field(default_factory=dict)
    model_metadata: Optional[dict[str, Any]] = None
    news_headlines: list[str] = field(default_factory=list)

    def session_date_obj(self) -> _date:
        return _date.fromisoformat(self.session_date)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)


def _strip_markdown_fences(text: str) -> str:
    """Some models wrap JSON in ```json ... ``` even with response_format set."""
    text = text.strip()
    if text.startswith("```"):
        # Remove leading fence (``` or ```json) on its own line.
        text = re.sub(r"^```(?:json)?\s*\n", "", text, count=1, flags=re.IGNORECASE)
    if text.endswith("```"):
        text = re.sub(r"\n```\s*$", "", text, count=1)
    return text.strip()


class BaseAgent(ABC):
    """Abstract base for all advisory agents.

    Concrete subclasses set:
        - ``name``          : str — agent id used in prompts and DB rows.
        - ``schema_class``  : Pydantic model the output must match.
        - ``system_prompt`` : str — short, schema-aware system message.

    They implement ``build_user_prompt(context)`` to produce the data
    portion of the prompt.
    """

    name: ClassVar[str] = "abstract"
    schema_class: ClassVar[type[BaseModel]]
    system_prompt: ClassVar[str] = (
        "You are an advisory trading-research assistant. "
        "Respond with a single JSON object that exactly matches the requested schema. "
        "Do not include prose outside the JSON. Do not include markdown fences. "
        "You are read-only and cannot place trades or change risk limits."
    )

    def __init__(self, llm: LLMClient) -> None:
        if llm is None:  # defensive: orchestrator should skip None
            raise ValueError("BaseAgent requires a non-None LLMClient")
        self.llm = llm
        self.log = get_logger(f"agents.{self.name}")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------
    @abstractmethod
    def build_user_prompt(self, context: AgentContext) -> str:
        """Produce the data-only portion of the prompt."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self, context: AgentContext) -> AgentResult:
        """Run the agent. Always returns; never raises."""
        try:
            user = self._wrap_user_prompt(context)
        except Exception as e:
            self.log.error("agents.prompt_build_failed", error=str(e))
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=None,
                error=f"prompt_build_failed: {e}",
            )

        # Network / API path.
        try:
            raw = self.llm.complete(system=self.system_prompt, user=user)
        except LLMClientError as e:
            self.log.warning("agents.llm_error", error=str(e))
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=None,
                error=f"llm_error: {e}",
            )
        except Exception as e:  # pragma: no cover - belt and braces
            self.log.error("agents.llm_unexpected", error=str(e))
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=None,
                error=f"llm_unexpected: {e}",
            )

        # Parse + validate path.
        cleaned = _strip_markdown_fences(raw)
        try:
            parsed = self.schema_class.model_validate_json(cleaned)
        except ValidationError as e:
            self.log.warning("agents.schema_invalid", error=str(e)[:300])
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=raw,
                error=f"schema_invalid: {e}",
            )
        except (ValueError, json.JSONDecodeError) as e:
            self.log.warning("agents.json_invalid", error=str(e))
            return AgentResult(
                agent_name=self.name,
                schema_valid=False,
                payload=None,
                raw_text=raw,
                error=f"json_invalid: {e}",
            )

        return AgentResult(
            agent_name=self.name,
            schema_valid=True,
            payload=parsed.model_dump(),
            raw_text=raw,
            error=None,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _wrap_user_prompt(self, context: AgentContext) -> str:
        """Prepend ``agent=<name>`` so MockLLMClient can dispatch."""
        body = self.build_user_prompt(context)
        return f"agent={self.name}\nsession_date={context.session_date}\n\n{body}"
