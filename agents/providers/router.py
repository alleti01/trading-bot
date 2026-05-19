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
# Settings attribute holding its provider choice. Listing ``data_quality``
# here makes its routing visible in dashboards / tests even though the
# agent itself runs deterministically and never calls the router.
AGENT_PROVIDER_FIELDS: dict[str, str] = {
    "news": "NEWS_AGENT_PROVIDER",
    "macro_news": "MACRO_NEWS_AGENT_PROVIDER",
    "strategy_research": "STRATEGY_RESEARCH_AGENT_PROVIDER",
    "trade_analysis": "TRADE_ANALYSIS_AGENT_PROVIDER",
    "model_review": "MODEL_REVIEW_AGENT_PROVIDER",
    "report": "REPORT_AGENT_PROVIDER",
    "risk_explainer": "RISK_EXPLAINER_AGENT_PROVIDER",
    "trade_journal": "TRADE_JOURNAL_AGENT_PROVIDER",
    "backtest_critic": "BACKTEST_CRITIC_AGENT_PROVIDER",
    "model_drift": "MODEL_DRIFT_AGENT_PROVIDER",
    "data_quality": "DATA_QUALITY_AGENT_PROVIDER",
}

# Same idea for the per-agent model overrides. The router looks up the
# resolved model via ``Settings.model_for_agent(name)`` so the
# ``${OPENAI_DEFAULT_MODEL}``-style shorthand operators may write in
# ``.env`` collapses to a real model name automatically.
AGENT_MODEL_FIELDS: dict[str, str] = {
    "news": "NEWS_AGENT_MODEL",
    "macro_news": "MACRO_NEWS_AGENT_MODEL",
    "strategy_research": "STRATEGY_RESEARCH_AGENT_MODEL",
    "trade_analysis": "TRADE_ANALYSIS_AGENT_MODEL",
    "model_review": "MODEL_REVIEW_AGENT_MODEL",
    "report": "REPORT_AGENT_MODEL",
    "risk_explainer": "RISK_EXPLAINER_AGENT_MODEL",
    "trade_journal": "TRADE_JOURNAL_AGENT_MODEL",
    "backtest_critic": "BACKTEST_CRITIC_AGENT_MODEL",
    "model_drift": "MODEL_DRIFT_AGENT_MODEL",
    "data_quality": "DATA_QUALITY_AGENT_MODEL",
}

# Providers that need an API key to be usable. ``anthropic`` and
# ``gemini`` are recognized as valid providers but disabled until the
# corresponding API key is set; this matches the spec's "future
# providers" expectation.
PROVIDERS_NEEDING_KEY: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
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

    ``available`` captures whether the API key is set; the factory is
    only called when we actually need an instance. ``default_model``
    is what we hand to a provider routed for an agent that did not
    set a per-agent model override.

    Lazily because we don't want to fail boot when a single key is
    missing — only the agents routed to that provider should disable.
    """

    name: str
    available: bool
    default_model: Optional[str]
    build: Callable[[str], BaseLLMProvider]


class ProviderRouter:
    """Selects the right :class:`BaseLLMProvider` for each agent.

    Construct via :meth:`from_settings`. The router supports two
    layers of caching:

    1. **Provider-by-name** (``_provider_cache``): the legacy hook
       used by tests that monkey-patch a provider for *every* agent
       sharing that provider name (``router._provider_cache["openai"]
       = mock``). When an entry is present here, all agents routed to
       that provider name reuse it as-is — model overrides are
       ignored. This keeps existing tests + simple "share one client
       across agents" deployments working.
    2. **Provider-by-agent** (``_agent_provider_cache``): when no
       legacy injection exists, each agent gets its own provider
       instance built with that agent's resolved model (so
       ``model_review`` can run on ``OPENAI_REVIEW_MODEL`` while
       ``trade_journal`` runs on ``OPENAI_DEFAULT_MODEL``).

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

    The router never logs API keys. Disable / construct logs include
    only ``provider``, ``model``, ``agent``, and a ``reason`` string.
    """

    def __init__(
        self,
        *,
        agent_provider_choice: dict[str, str],
        agent_models: dict[str, Optional[str]],
        provider_specs: dict[str, _ProviderSpec],
        log_disabled: Optional[Callable[[str, dict[str, Any]], None]] = None,
    ) -> None:
        self._choices = dict(agent_provider_choice)
        self._agent_models = dict(agent_models)
        self._provider_specs = dict(provider_specs)
        # Memoized provider instances by canonical provider name —
        # legacy back-compat hook for tests that inject a single
        # provider for every agent in that family.
        self._provider_cache: dict[str, Optional[BaseLLMProvider]] = {}
        # Memoized provider instances by *agent* name. Each entry
        # captures the agent's specific model override.
        self._agent_provider_cache: dict[str, Optional[BaseLLMProvider]] = {}
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
        # ``OPENAI_DEFAULT_MODEL`` is the modern knob; ``OPENAI_MODEL``
        # is the legacy fallback so single-provider deployments don't
        # need to add a new env var.
        openai_default_model = (
            settings.OPENAI_MODEL or settings.OPENAI_DEFAULT_MODEL
        )

        def _openai_build(model: str) -> BaseLLMProvider:
            return OpenAIProvider(
                openai_key, model=model, timeout_seconds=timeout
            )

        def _perplexity_build(model: str) -> BaseLLMProvider:
            return PerplexityProvider(
                perplexity_key, model=model, timeout_seconds=timeout
            )

        def _anthropic_build(model: str) -> BaseLLMProvider:
            return AnthropicProvider(
                anthropic_key, model=model, timeout_seconds=timeout
            )

        def _gemini_build(model: str) -> BaseLLMProvider:
            return GeminiProvider(
                gemini_key, model=model, timeout_seconds=timeout
            )

        provider_specs: dict[str, _ProviderSpec] = {
            "openai": _ProviderSpec(
                name="openai",
                available=bool(openai_key),
                default_model=openai_default_model,
                build=_openai_build,
            ),
            "perplexity": _ProviderSpec(
                name="perplexity",
                available=bool(perplexity_key),
                default_model=(
                    settings.PERPLEXITY_DEFAULT_MODEL
                    or settings.PERPLEXITY_MODEL
                ),
                build=_perplexity_build,
            ),
            "anthropic": _ProviderSpec(
                name="anthropic",
                available=bool(anthropic_key),
                default_model=settings.ANTHROPIC_MODEL,
                build=_anthropic_build,
            ),
            "gemini": _ProviderSpec(
                name="gemini",
                available=bool(gemini_key),
                default_model=settings.GEMINI_MODEL,
                build=_gemini_build,
            ),
        }

        choices: dict[str, str] = {}
        agent_models: dict[str, Optional[str]] = {}
        for agent_name, field_name in AGENT_PROVIDER_FIELDS.items():
            raw = getattr(settings, field_name, "")
            choices[agent_name] = (raw or "").strip().lower()
            try:
                agent_models[agent_name] = settings.model_for_agent(agent_name)
            except Exception:  # pragma: no cover - settings should not raise here
                agent_models[agent_name] = None

        return cls(
            agent_provider_choice=choices,
            agent_models=agent_models,
            provider_specs=provider_specs,
        )

    # ---- internal helpers -----------------------------------------
    def _disable(self, agent_name: str, **details: Any) -> None:
        # Single emit point so tests can patch ``log_disabled`` to
        # observe disable reasons without scraping log output. We
        # never log API keys — only ``provider``, ``model``, ``agent``,
        # and the disable reason.
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

    def model_for(self, agent_name: str) -> Optional[str]:
        """Resolved model name for an agent, or ``None`` if disabled."""
        return self._agent_models.get(agent_name)

    def provider_for(self, agent_name: str) -> Optional[BaseLLMProvider]:
        """Return a ready :class:`BaseLLMProvider` for an agent, or
        ``None`` if the agent is disabled / its key is missing /
        the provider name is unknown.

        Resolution order:

        1. **Per-agent cache** — already-built instance, possibly
           overridden by a test.
        2. **Provider-by-name cache** (``_provider_cache``) — legacy
           back-compat hook: if a test set
           ``router._provider_cache["openai"] = mock``, every agent
           routed to OpenAI returns ``mock`` regardless of model
           override. Tests rely on this.
        3. **Lazy construct** — build a fresh provider with the
           agent's resolved model, cache by agent.
        """
        if agent_name in self._agent_provider_cache:
            return self._agent_provider_cache[agent_name]

        choice = self.chosen_provider_name(agent_name)
        if choice is None:
            self._disable(
                agent_name, reason="explicitly_disabled_or_unknown_agent"
            )
            self._agent_provider_cache[agent_name] = None
            return None
        if choice not in self._provider_specs:
            self._disable(
                agent_name,
                reason="unknown_provider",
                requested=choice,
                available=list(self._provider_specs),
            )
            self._agent_provider_cache[agent_name] = None
            return None

        # Legacy back-compat: tests set ``_provider_cache[choice] = mock``
        # to override every agent on that provider. Honor that without
        # building a new instance.
        legacy = self._provider_cache.get(choice)
        if legacy is not None:
            self._agent_provider_cache[agent_name] = legacy
            return legacy

        spec = self._provider_specs[choice]
        if not spec.available:
            self._disable(
                agent_name, reason="missing_api_key", provider=choice
            )
            self._agent_provider_cache[agent_name] = None
            # Tag the by-name cache so subsequent lookups for sibling
            # agents on the same provider take the fast path.
            self._provider_cache.setdefault(choice, None)
            return None

        model = self._agent_models.get(agent_name) or spec.default_model
        if not model:
            # No model resolved (e.g. agent intentionally deterministic
            # but routed by mistake). Disable rather than guess.
            self._disable(
                agent_name, reason="no_model_resolved", provider=choice
            )
            self._agent_provider_cache[agent_name] = None
            return None

        try:
            instance: Optional[BaseLLMProvider] = spec.build(model)
        except Exception as e:  # noqa: BLE001 - construction failures are non-fatal
            # ``str(e)`` from a provider constructor never includes the
            # API key (we control all four constructors), so this is
            # safe to log.
            self._log.warning(
                "providers.construct_failed",
                provider=choice,
                model=model,
                error=str(e),
            )
            instance = None

        self._agent_provider_cache[agent_name] = instance
        if instance is None:
            self._disable(
                agent_name,
                reason="construct_failed",
                provider=choice,
                model=model,
            )
            return None

        self._log.info(
            "providers.agent_ready",
            agent=agent_name,
            provider=choice,
            model=model,
        )
        return instance

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
        peeking into Settings. Never includes API keys — only
        provider names, model names, and a boolean enabled flag.
        """
        out: dict[str, dict[str, Any]] = {}
        for agent_name in self._choices:
            choice = self.chosen_provider_name(agent_name)
            resolved_model = self._agent_models.get(agent_name)
            client = self.client_for(agent_name)
            out[agent_name] = {
                "provider": choice,
                "enabled": client is not None,
                "model": (
                    client.model
                    if client is not None
                    else (resolved_model if choice is not None else None)
                ),
            }
        return out


__all__ = [
    "AGENT_MODEL_FIELDS",
    "AGENT_PROVIDER_FIELDS",
    "DISABLED_NAMES",
    "PROVIDER_NAMES",
    "PROVIDERS_NEEDING_KEY",
    "ProviderLLMClient",
    "ProviderRouter",
]
