"""Provider-neutral interface for LLM agents.

The advisory agent layer historically targeted a single
:class:`~agents.llm_client.LLMClient` (OpenAI). This module generalizes
that to a per-agent multi-provider model: each advisory agent picks the
right tool for its job (web-grounded research vs. structured
reasoning/summarization) and the rest of the pipeline stays unchanged.

Design rules
------------
- Providers expose ``generate_text`` + ``generate_json`` as the only
  primitives. Anything fancier (tool use, streaming) is intentionally
  out of scope for the MVP.
- Providers MUST NOT raise on transient errors — they raise a single
  :class:`ProviderError` so the agent layer's existing
  ``LLMClientError`` handling continues to apply unchanged.
- Providers MUST NOT import :mod:`execution` or :mod:`risk`. The
  advisory layer is read-only by construction; the architectural
  isolation test (`tests/test_agent_isolation.py`) enforces this.
- Citations are optional. Reasoning providers (OpenAI, Anthropic,
  Gemini) typically return ``citations=[]``; web-grounded providers
  (Perplexity) populate them. Callers must treat the citation list as
  best-effort metadata, never as authoritative URLs.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class ProviderError(RuntimeError):
    """Wraps any backend failure (network, auth, bad shape, JSON parse).

    Agents catch this in their existing
    :class:`~agents.llm_client.LLMClientError` block (we wrap into that
    via :class:`agents.providers.router.ProviderLLMClient`) so failure
    handling does not change between the legacy single-client mode and
    the new per-agent mode.
    """


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Citation:
    """One supporting source returned alongside a generation.

    ``url`` is the only required field; ``title`` and ``snippet`` are
    populated when the upstream provider exposes them. Treat citations
    as advisory metadata only — they are not authenticated and may
    change between requests.
    """

    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None


@dataclass(frozen=True)
class ProviderTextResult:
    """Plain-text generation result.

    ``provider``/``model`` are propagated from the underlying provider
    instance so downstream logging always knows where text came from
    (especially useful when more than one provider is in use within the
    same orchestrator run).
    """

    text: str
    provider: str
    model: str
    citations: list[Citation] = field(default_factory=list)
    raw: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class ProviderJSONResult:
    """Structured JSON generation result.

    ``data`` is the parsed JSON object. When a Pydantic ``schema`` is
    provided to :meth:`BaseLLMProvider.generate_json`, ``data`` is the
    validated model instance dumped via ``model_dump()`` so callers can
    treat it as a plain dict. ``text`` is always the raw JSON string
    the provider emitted, kept for the audit trail.
    """

    data: dict[str, Any]
    text: str
    provider: str
    model: str
    citations: list[Citation] = field(default_factory=list)
    raw: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers shared by every provider
# ---------------------------------------------------------------------------
_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)


def _strip_markdown_fences(text: str) -> str:
    r"""Remove ``\`\`\`json`` / ``\`\`\`\`` fences without altering content.

    Several providers emit fenced JSON even when explicitly asked for a
    bare object. Stripping fences before ``json.loads`` keeps the
    schema-validation path noise-free.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n", "", s, count=1, flags=re.IGNORECASE)
    if s.endswith("```"):
        s = re.sub(r"\n```\s*$", "", s, count=1)
    return s.strip()


def parse_json_with_optional_schema(
    raw_text: str,
    *,
    schema: Optional[Type[BaseModel]] = None,
) -> dict[str, Any]:
    """Parse ``raw_text`` as JSON; optionally validate against a schema.

    Centralized here so every provider's ``generate_json`` method
    behaves the same way. Failures raise :class:`ProviderError` (never
    a bare ``ValueError`` / ``ValidationError``) so the agent layer's
    existing error handling kicks in uniformly.
    """
    cleaned = _strip_markdown_fences(raw_text)
    if not cleaned:
        raise ProviderError("provider.empty_response: no JSON content")
    try:
        loaded = json.loads(cleaned)
    except (ValueError, json.JSONDecodeError) as e:
        raise ProviderError(f"provider.json_parse: {e}") from e

    if not isinstance(loaded, dict):
        raise ProviderError(
            "provider.json_shape: expected a JSON object, got "
            f"{type(loaded).__name__}"
        )

    if schema is not None:
        try:
            instance = schema.model_validate(loaded)
        except ValidationError as e:
            raise ProviderError(f"provider.schema_invalid: {e}") from e
        return instance.model_dump()

    return loaded


# ---------------------------------------------------------------------------
# Base provider interface
# ---------------------------------------------------------------------------
class BaseLLMProvider(ABC):
    """Provider-neutral primitives for the advisory agent layer.

    Concrete subclasses set :attr:`provider_name` (e.g. ``"openai"``)
    and implement :meth:`generate_text`. :meth:`generate_json` has a
    default implementation that wraps ``generate_text`` with a JSON
    instruction; providers that natively support structured outputs
    (OpenAI's ``response_format=json_object``, Gemini's response MIME
    type, etc.) should override it for a tighter contract.

    Both methods MUST NOT raise on bad shapes — they raise
    :class:`ProviderError`. Network errors are also funneled through
    :class:`ProviderError` so the agent layer's existing
    ``LLMClientError`` handling continues to apply.
    """

    provider_name: str = "abstract"

    def __init__(self, *, model: str, timeout_seconds: float = 30.0) -> None:
        if not model:
            raise ValueError("BaseLLMProvider requires a non-empty model")
        self._model = str(model)
        self._timeout = float(timeout_seconds)

    # ---- public properties ---------------------------------------
    @property
    def model_name(self) -> str:
        return self._model

    @property
    def timeout_seconds(self) -> float:
        return self._timeout

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"<{self.provider_name} model={self._model}>"

    # ---- text + json --------------------------------------------
    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderTextResult:
        """Free-form text generation. Concrete providers must implement."""

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> ProviderJSONResult:
        """Structured-JSON generation.

        Default implementation: call :meth:`generate_text` with a JSON
        instruction baked into the system prompt and parse the result
        through :func:`parse_json_with_optional_schema`.

        Providers that support a native structured-output mode should
        override this for stronger guarantees (and lower error rates).
        """
        system = system_prompt or ""
        json_instruction = (
            "Respond ONLY with a single JSON object. No prose, no "
            "markdown fences. Match the requested schema exactly."
        )
        composed_system = (
            (system + "\n\n" + json_instruction).strip()
            if system
            else json_instruction
        )
        text_result = self.generate_text(
            prompt,
            system_prompt=composed_system,
            temperature=temperature,
        )
        data = parse_json_with_optional_schema(text_result.text, schema=schema)
        return ProviderJSONResult(
            data=data,
            text=text_result.text,
            provider=text_result.provider,
            model=text_result.model,
            citations=list(text_result.citations),
            raw=text_result.raw,
        )


__all__ = [
    "BaseLLMProvider",
    "Citation",
    "ProviderError",
    "ProviderJSONResult",
    "ProviderTextResult",
    "parse_json_with_optional_schema",
]
