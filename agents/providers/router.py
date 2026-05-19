"""Per-agent LLM provider routing.

The advisory agent layer used to call OpenAI for every agent. This
module lets each agent pick the right tool for its job:

- web-grounded research (NewsAgent, MacroNewsAgent, future
  StrategyResearchAgent) -> Perplexity by default;
- structured reasoning / summarization (TradeAnalysisAgent,
  ModelReviewAgent, ReportAgent, RiskExplainerAgent,
  TradeJournalAgent) -> OpenAI by default;
- any agent can be retargeted by setting its ``*_AGENT_PROVIDER`` env
  var to ``openai`` / ``perplexity`` / ``anthropic`` / ``gemini`` /
  ``mock`` / ``none``.

Three things this module guarantees:

1. **Graceful disable on missing key.** If an agent is routed to a
   provider whose API key is not configured, the router yields ``None``
   for that agent. The orchestrator already treats ``None`` as "agent
   off" — no crash, no half-working state.
2. **Architectural isolation preserved.** This module imports nothing
   from :mod:`execution` or :mod:`risk` (verified by
   ``tests/test_agent_isolation.py``).
3. **Drop-in compatibility.** A thin :class:`ProviderLLMClient`
   adapter lets existing agents (which expect the legacy
   :class:`agents.llm_client.LLMClient`) consume the new providers
   unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from pydantic import SecretStr

from agents.llm_client import LLMClient, LLMClientError
from agents.providers.anthropic_provider import AnthropicProvider
from agents.providers.base import BaseLLMProvider, ProviderError
from agents.providers.gemini_provider import GeminiProvider
from agents.providers.openai_provider import OpenAIProvider
from agents.providers.perplexity_provider import PerplexityProvider
from app.logging_config import get_logger
from config.settings import Settings


# ---------------------------------------------------------------------------
# Constants — agent name -> settings field for its provider override
# ---------------------------------------------------------------------------
PROVIDER_NAMES: tuple[str, ...] = (
    "openai",
    "perplexity",
    "anthropic",
    "gemini",
)
DISABLED_NAMES: frozenset[str] = frozenset(
    {"none", "off", "disabled", "false", ""}
)

# Maps the canonical agent id (the value of ``BaseAgent.name``) to the
# Settings attribute holding its provider choice. Future agents that
# don't yet exist as classes (``macro_news``, ``strategy_research``)
# are listed too so the router pre-validates their config; the
# orchestrator just doesn't construct an instance for them yet.
AGENT_PROVIDER_FIELDS: dict[str, str] = {
    "news": "NEWS_AGENT_PROVIDER",
    "macro_news": "MACRO_NEWS_AGENT_PROVIDER",
    "strategy_research": "STRATEGY_RESEARCH_AGENT_PROVIDER",
    "trade_analysis": "TRADE_ANALYSIS_AGENT_PROVIDER",
    "model_review": "MODEL_REVIEW_AGENT_PROVIDER",
    "report": "REPORT_AGENT_PROVIDER",
    "risk_explainer": "RISK_EXPLAINER_AGENT_PROVIDER",
    "trade_journal": "TRADE_JOURNAL_AGENT_PROVIDER",
}


# ---------------------------------------------------------------------------
# Adapter so existing BaseAgent code works with the new providers
# ---------------------------------------------------------------------------
class ProviderLLMClient(LLMClient):
    """Bridges :class:`BaseLLMProvider` into the legacy
    :class:`LLMClient` interface used by :class:`agents.base_agent.BaseAgent`.

    Each existing agent calls ``self.llm.complete(system=..., user=...)``
    expecting a JSON string; this adapter funnels that into the
    provider's :meth:`generate_text` (we don't use ``generate_json``
    here because the agent layer parses + validates with its own
    Pydantic schemas right after, and double-parsing wastes CPU on
    larger payloads).

    Provider failures are translated to :class:`LLMClientError` so the
    agent's existing error handler keeps working.
    """

    def __init__(
        self,
        provider: BaseLLMProvider,
        *,
        temperature: float = 0.2,
    ) -> None:
        self.provider = provider
        self.temperature = float(temperature)
        self.last_citations: list[Any] = []

    @property
    def model(self) -> str:
        return self.provider.model_name

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    def complete(self, *, system: str, user: str) -> str:
        try:
            result = self.provider.generate_text(
                user,
                system_prompt=system,
                temperature=self.temperature,
            )
        except ProviderError as e:
            # Re-raise via the agent layer's existing error type so the
            # ``BaseAgent.run`` handler catches it without changes.
            raise LLMClientError(str(e)) from e
        # Citations are kept side-channel so a future agent can read
        # them via ``orchestrator.last_citations_for("news")`` — out of
        # scope for the MVP, but cheap to surface.
        self.last_citations = list(result.citations)
        return result.text


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ProviderSpec:
    """Internal record of how to construct a provider lazily.

    Lazily because we don't want to fail boot when a single key is
    missing — only the agents routed to that provider should disable.
    """

    name: str
    factory: Callable[[], Optional[BaseLLMProvider]]


class ProviderRouter:
    """Selects the right :class:`BaseLLMProvider` for each agent.

    Construct via :meth:`from_settings`. The router caches one
    instance per provider name (so all agents routed to OpenAI share
    a single ``OpenAIProvider`` and httpx connection pool).

    Public API:

    - :meth:`provider_for(agent_name)` returns a ``BaseLLMProvider``
      or ``None``.
    - :meth:`client_for(agent_name)` returns a legacy ``LLMClient``
      adapter or ``None``. The orchestrator uses this so it can keep
      handing each :class:`BaseAgent` an object that looks like the
      old single-client API.
    - :meth:`is_enabled(agent_name)` is a cheap precheck.
    - :meth:`enabled_agents()` reports the set of agent ids that have
      a working provider given the current settings.
    """

    def __init__(
        self,
        *,
        agent_provider_choice: dict[str, str],
        provider_specs: dict[str, _ProviderSpec],
        log_disabled: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> None:
        self._choices = dict(agent_provider_choice)
        self._provider_specs = dict(provider_specs)
        # Memoized provider instances by canonical name.
        self._provider_cache: dict[str, Optional[BaseLLMProvider]] = {}
        # Memoized LLMClient adapters by agent name (different agents
        # routed to the same provider still share the underlying
        # provider instance — they just get separate adapters).
        self._client_cache: dict[str, Optional[ProviderLLMClient]] = {}
        self._log = get_logger("agents.providers.router")
        self._log_disabled = log_disabled

    # ---- factory --------------------------------------------------
    @classmethod
    def from_settings(cls, settings: Settings) -> "ProviderRouter":
        """Build a router from the global :class:`Settings` object.

        Provider keys / models are read once here. The router itself
        is immutable after construction so the operator gets a stable
        dispatch table for the run.
        """

        def _key(name: str) -> Optional[str]:
            value: Optional[SecretStr] = getattr(settings, name, None)
            if value is None:
                return None
            return value.get_secret_value() or None

        openai_key = _key("OPENAI_API_KEY")
        perplexity_key = _key("PERPLEXITY_API_KEY")
        anthropic_key = _key("ANTHROPIC_API_KEY")
        gemini_key = _key("GEMINI_API_KEY")

        timeout = float(settings.LLM_TIMEOUT_SECONDS)
        # OPENAI_MODEL falls back to LLM_MODEL so single-provider users
        # don't need to set both.
        openai_model = settings.OPENAI_MODEL or settings.LLM_MODEL

        provider_specs: dict[str, _ProviderSpec] = {
            "openai": _ProviderSpec(
                name="openai",
                factory=(
                    (lambda: OpenAIProvider(
                        openai_key,
                        model=openai_model,
                        timeout_seconds=timeout,
                    ))
                    if openai_key
                    else (lambda: None)
                ),
            ),
            "perplexity": _ProviderSpec(
                name="perplexity",
                factory=(
                    (lambda: PerplexityProvider(
                        perplexity_key,
                        model=settings.PERPLEXITY_MODEL,
                        timeout_seconds=timeout,
                    ))
                    if perplexity_key
                    else (lambda: None)
                ),
            ),
            "anthropic": _ProviderSpec(
                name="anthropic",
                factory=(
                    (lambda: AnthropicProvider(
                        anthropic_key,
                        model=settings.ANTHROPIC_MODEL,
                        timeout_seconds=timeout,
                    ))
                    if anthropic_key
                    else (lambda: None)
                ),
            ),
            "gemini": _ProviderSpec(
                name="gemini",
                factory=(
                    (lambda: GeminiProvider(
                        gemini_key,
                        model=settings.GEMINI_MODEL,
                        timeout_seconds=timeout,
                    ))
                    if gemini_key
                    else (lambda: None)
                ),
            ),
        }

        choices: dict[str, str] = {}
        for agent_name, field_name in AGENT_PROVIDER_FIELDS.items():
            raw = getattr(settings, field_name, "")
            choices[agent_name] = (raw or "").strip().lower()

        return cls(
            agent_provider_choice=choices,
            provider_specs=provider_specs,
        )

    # ---- internal helpers -----------------------------------------
    def _resolve_provider(self, name: str) -> Optional[BaseLLMProvider]:
        if name in self._provider_cache:
            return self._provider_cache[name]
        spec = self._provider_specs.get(name)
        if spec is None:
            self._provider_cache[name] = None
            return None
        try:
            instance = spec.factory()
        except Exception as e:  # noqa: BLE001 - construction failures are non-fatal
            self._log.warning(
                "providers.construct_failed", provider=name, error=str(e)
            )
            instance = None
        self._provider_cache[name] = instance
        return instance

    def _disable(self, agent_name: str, **details: Any) -> None:
        # Single emit point so tests can patch ``log_disabled`` to
        # observe disable reasons without scraping log output.
        details = {"agent": agent_name, **details}
        self._log.warning("providers.agent_disabled", **details)
        if self._log_disabled is not None:
            self._log_disabled(agent_name, details)

    # ---- public API -----------------------------------------------
    def chosen_provider_name(self, agent_name: str) -> Optional[str]:
        """Return the configured provider id for an agent (lowercased)
        or ``None`` if the agent is not registered or explicitly off.
        """
        choice = self._choices.get(agent_name)
        if choice is None:
            return None
        if choice in DISABLED_NAMES:
            return None
        return choice

    def provider_for(self, agent_name: str) -> Optional[BaseLLMProvider]:
        """Return a ready :class:`BaseLLMProvider` for an agent, or
        ``None`` if the agent is disabled / its key is missing /
        the provider name is unknown.
        """
        choice = self.chosen_provider_name(agent_name)
        if choice is None:
            self._disable(agent_name, reason="explicitly_disabled_or_unknown_agent")
            return None
        if choice not in self._provider_specs:
            self._disable(
                agent_name,
                reason="unknown_provider",
                requested=choice,
                available=list(self._provider_specs),
            )
            return None
        provider = self._resolve_provider(choice)
        if provider is None:
            self._disable(
                agent_name, reason="missing_api_key", provider=choice
            )
            return None
        return provider

    def client_for(self, agent_name: str) -> Optional[ProviderLLMClient]:
        """Return a legacy-shaped :class:`LLMClient` adapter for an agent.

        Adapters are cached per agent so an :class:`AgentOrchestrator`
        instance always hands the same agent the same adapter (and so
        ``last_citations`` stays stable across consecutive calls).
        """
        if agent_name in self._client_cache:
            return self._client_cache[agent_name]
        provider = self.provider_for(agent_name)
        if provider is None:
            self._client_cache[agent_name] = None
            return None
        client = ProviderLLMClient(provider)
        self._client_cache[agent_name] = client
        return client

    def is_enabled(self, agent_name: str) -> bool:
        return self.client_for(agent_name) is not None

    def enabled_agents(self) -> list[str]:
        return sorted(a for a in self._choices if self.is_enabled(a))

    def has_any_enabled(self) -> bool:
        return any(self.is_enabled(a) for a in self._choices)

    def routing_table(self) -> dict[str, dict[str, Any]]:
        """Operator-facing snapshot of the current routing.

        Returns ``{agent_name: {"provider": <name|None>,
        "enabled": bool, "model": <model_name|None>}}`` so the daily
        report or a status endpoint can show what's wired without
        peeking into Settings.
        """
        out: dict[str, dict[str, Any]] = {}
        for agent_name in self._choices:
            choice = self.chosen_provider_name(agent_name)
            client = self.client_for(agent_name)
            out[agent_name] = {
                "provider": choice,
                "enabled": client is not None,
                "model": client.model if client is not None else None,
            }
        return out


__all__ = [
    "AGENT_PROVIDER_FIELDS",
    "PROVIDER_NAMES",
    "ProviderLLMClient",
    "ProviderRouter",
]
