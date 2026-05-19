"""Multi-provider LLM routing for advisory agents.

The advisory agent layer used to talk to a single OpenAI client. This
package generalizes that to a per-agent multi-provider model: research
agents (NewsAgent, MacroNewsAgent, future StrategyResearchAgent) can
target Perplexity for web grounding while reasoning agents
(TradeAnalysisAgent, ModelReviewAgent, ReportAgent, etc.) keep using
OpenAI / Anthropic / Gemini.

Architectural invariants enforced by tests:

- Modules under ``agents/providers/`` MUST NOT import :mod:`execution`
  or :mod:`risk`. Agents and their providers are advisory only.
- Providers raise :class:`ProviderError` (a ``RuntimeError`` subclass)
  on any backend failure — the existing
  :class:`agents.llm_client.LLMClientError` handler in
  :class:`agents.base_agent.BaseAgent` catches these via
  :class:`ProviderLLMClient`.
- The router yields ``None`` for agents whose required API key is
  missing instead of raising; the orchestrator already treats ``None``
  as "agent off", so a missing key disables one agent without
  affecting any other.
"""

from agents.providers.anthropic_provider import AnthropicProvider
from agents.providers.base import (
    BaseLLMProvider,
    Citation,
    ProviderError,
    ProviderJSONResult,
    ProviderTextResult,
    parse_json_with_optional_schema,
)
from agents.providers.gemini_provider import GeminiProvider
from agents.providers.openai_provider import OpenAIProvider
from agents.providers.perplexity_provider import PerplexityProvider
from agents.providers.router import (
    AGENT_PROVIDER_FIELDS,
    PROVIDER_NAMES,
    ProviderLLMClient,
    ProviderRouter,
)

__all__ = [
    "AGENT_PROVIDER_FIELDS",
    "AnthropicProvider",
    "BaseLLMProvider",
    "Citation",
    "GeminiProvider",
    "OpenAIProvider",
    "PerplexityProvider",
    "PROVIDER_NAMES",
    "ProviderError",
    "ProviderJSONResult",
    "ProviderLLMClient",
    "ProviderRouter",
    "ProviderTextResult",
    "parse_json_with_optional_schema",
]
