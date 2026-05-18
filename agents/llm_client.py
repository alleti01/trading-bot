"""LLM client adapters used by the advisory agents.

Three implementations:

- :class:`LLMClient`         — abstract; ``complete(system, user) -> str``.
- :class:`OpenAILLMClient`   — minimal ``httpx``-based wrapper around
  the OpenAI chat-completions endpoint. No ``openai`` package needed.
- :class:`MockLLMClient`     — deterministic JSON for tests + ``--smoke-agents``.

``build_llm_client(settings)`` returns ``None`` if agents are disabled or
no key is configured. Every caller (orchestrator) treats ``None`` as
"agents off" — no per-agent special-casing required.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Optional

import httpx
from pydantic import SecretStr

from app.logging_config import get_logger
from config.settings import Settings


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class LLMClientError(RuntimeError):
    """Wraps any backend failure (network, auth, bad response, JSON parse)."""


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------
class LLMClient(ABC):
    """Single primitive: turn (system, user) prompts into a string response.

    The agent layer is responsible for parsing the response into a schema
    and handling validation errors — the client just yields raw text.
    """

    @abstractmethod
    def complete(self, *, system: str, user: str) -> str:  # pragma: no cover
        ...


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAILLMClient(LLMClient):
    """Minimal OpenAI chat-completions adapter.

    Uses the bare HTTPS endpoint via ``httpx`` so we don't pull in the
    full ``openai`` SDK for a single endpoint. ``response_format``
    requests JSON; the agent still validates the parsed object against a
    Pydantic schema so a non-conforming JSON object is caught downstream.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
        temperature: float = 0.2,
    ) -> None:
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        if not api_key:
            raise ValueError("OpenAILLMClient requires a non-empty api_key")
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.base_url = base_url or self.DEFAULT_BASE_URL
        self.temperature = float(temperature)
        self.log = get_logger("agents.llm.openai")

    def complete(self, *, system: str, user: str) -> str:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature,
            # Ask the API for valid JSON. Agents still revalidate.
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            response = httpx.post(
                self.base_url,
                headers=headers,
                json=body,
                timeout=self.timeout_seconds,
            )
        except httpx.HTTPError as e:
            raise LLMClientError(f"openai.http_error: {e}") from e

        if response.status_code >= 400:
            raise LLMClientError(
                f"openai.bad_status: {response.status_code} {response.text[:200]}"
            )

        try:
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError, json.JSONDecodeError) as e:
            raise LLMClientError(f"openai.bad_payload: {e}") from e


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------
class MockLLMClient(LLMClient):
    """Deterministic LLM client for tests and ``--smoke-agents``.

    The orchestrator builds prompts that begin with the agent name (e.g.
    ``"agent=news\\n…"``); :meth:`complete` keys off that to return a
    canned JSON string. Tests can also pass ``responses_by_agent`` to
    override per-agent payloads.
    """

    DEFAULT_RESPONSES: dict[str, str] = {
        "news": json.dumps(
            {
                "high_risk_window": False,
                "severity": "low",
                "events": [],
                "summary": "No high-impact scheduled events identified.",
                "recommendation": "Trade normal session size.",
            }
        ),
        "risk_explainer": json.dumps(
            {
                "session_date": "1970-01-01",
                "blocks": [],
                "overall_assessment": "No risk blocks today.",
                "operator_actions": [],
            }
        ),
        "trade_journal": json.dumps(
            {
                "session_date": "1970-01-01",
                "highlights": [],
                "mistakes": [],
                "lessons": [],
                "best_trade_setup_id": None,
                "worst_trade_setup_id": None,
            }
        ),
        "report": json.dumps(
            {
                "session_date": "1970-01-01",
                "headline": "Quiet session.",
                "bullets": [],
                "compliance_notes": [],
                "tomorrow_focus": "Stick to plan.",
            }
        ),
        "model_review": json.dumps(
            {
                "model_name": "none",
                "model_version": "none",
                "calibration_comment": "No model loaded today.",
                "drift_warnings": [],
                "retrain_recommended": False,
                "reason": "Insufficient signal volume to evaluate drift.",
            }
        ),
    }

    def __init__(
        self,
        *,
        responses_by_agent: Optional[dict[str, str]] = None,
        raise_for_agents: Optional[set[str]] = None,
    ) -> None:
        self.responses = dict(self.DEFAULT_RESPONSES)
        if responses_by_agent:
            self.responses.update(responses_by_agent)
        self.raise_for_agents = raise_for_agents or set()
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str) -> str:
        self.calls.append((system, user))
        agent = _extract_agent_marker(user) or _extract_agent_marker(system) or ""
        if agent in self.raise_for_agents:
            raise LLMClientError(f"mock.simulated_failure agent={agent}")
        if agent in self.responses:
            return self.responses[agent]
        # Fall back to a generic empty JSON object so callers can detect
        # a validation failure rather than a string parse error.
        return "{}"


def _extract_agent_marker(text: str) -> Optional[str]:
    """Find the ``agent=<name>`` marker the orchestrator prepends to prompts."""
    if not text:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("agent="):
            return line.split("=", 1)[1].strip()
    return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_llm_client(settings: Settings) -> Optional[LLMClient]:
    """Pick the right client based on settings, or ``None`` if disabled.

    ``ENABLE_LLM_AGENTS=false`` always wins. With agents enabled but no
    ``OPENAI_API_KEY`` the orchestrator runs as a no-op (also returning
    ``None`` here) so the bot keeps running unchanged.
    """
    log = get_logger("agents.llm.builder")
    if not settings.ENABLE_LLM_AGENTS:
        log.info("agents.disabled", reason="ENABLE_LLM_AGENTS=false")
        return None
    if settings.OPENAI_API_KEY is None:
        log.warning("agents.disabled", reason="OPENAI_API_KEY missing")
        return None
    return OpenAILLMClient(
        settings.OPENAI_API_KEY,
        model=settings.LLM_MODEL,
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
    )
