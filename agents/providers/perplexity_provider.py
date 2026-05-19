"""Perplexity provider for web-grounded research agents.

Perplexity exposes an OpenAI-compatible chat-completions API at
``https://api.perplexity.ai/chat/completions`` plus a top-level
``citations`` (and the newer ``search_results``) field on the response
that lists the sources the model retrieved. We surface those as
:class:`agents.providers.base.Citation` objects so downstream code can
display "the LLM said this — and here is what it read".

This provider is the right pick for:

- ``news_agent`` and ``macro_news_agent`` — pre-session and EOD news
  risk evaluation. Web grounding is exactly the value-add here.
- ``strategy_research_agent`` (future) — survey of public market
  research before any candidate-strategy work.

Anything that doesn't need web access (trade narratives, model review,
report writing) should route to OpenAI / Anthropic instead so the
extra retrieval cost isn't paid for nothing.

Safety properties enforced here:

- Missing ``PERPLEXITY_API_KEY`` -> ``ValueError`` at construct time.
  The router catches that and disables every routed agent.
- Network errors -> :class:`ProviderError`. The agent layer's existing
  ``LLMClientError`` handler treats this exactly like an OpenAI error
  so paper mode never crashes on a flaky search call.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Type

import httpx
from pydantic import BaseModel, SecretStr

from agents.providers.base import (
    BaseLLMProvider,
    Citation,
    ProviderError,
    ProviderJSONResult,
    ProviderTextResult,
    parse_json_with_optional_schema,
)
from app.logging_config import get_logger


class PerplexityProvider(BaseLLMProvider):
    """Web-grounded chat completions with citations.

    The wire format is OpenAI-compatible enough that the request body
    is essentially identical to :class:`OpenAIProvider` — what changes
    is the response shape (Perplexity adds ``citations`` and
    ``search_results``) and the model catalog (``sonar``, ``sonar-pro``,
    ``sonar-reasoning``, etc.).
    """

    provider_name = "perplexity"
    DEFAULT_BASE_URL = "https://api.perplexity.ai/chat/completions"
    DEFAULT_MODEL = "sonar"

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        if not api_key:
            raise ValueError("PerplexityProvider requires a non-empty api_key")
        self._api_key: str = api_key
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._http = http_client
        self._owns_http = http_client is None
        self._log = get_logger("agents.providers.perplexity")

    # ---- HTTP helpers ----------------------------------------------
    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        try:
            if self._http is not None:
                response = self._http.post(
                    self._base_url,
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
            else:
                response = httpx.post(
                    self._base_url,
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
        except httpx.HTTPError as e:
            raise ProviderError(f"perplexity.http_error: {e}") from e

        if response.status_code >= 400:
            raise ProviderError(
                f"perplexity.bad_status: {response.status_code} "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise ProviderError(f"perplexity.bad_json: {e}") from e

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"perplexity.bad_payload: {e}") from e

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _extract_citations(payload: dict[str, Any]) -> list[Citation]:
        """Read citations / search results regardless of API shape.

        Perplexity has been migrating from a flat ``citations: [str]``
        list to a richer ``search_results: [{url,title,...}]`` shape.
        We accept either (or both) and dedupe by URL so the consumer
        never has to know which API version it talked to.
        """
        seen: set[str] = set()
        out: list[Citation] = []

        def _push(url: Any, title: Any = None, snippet: Any = None) -> None:
            if not url:
                return
            url_s = str(url).strip()
            if not url_s or url_s in seen:
                return
            seen.add(url_s)
            out.append(
                Citation(
                    url=url_s,
                    title=str(title).strip() if title else None,
                    snippet=str(snippet).strip() if snippet else None,
                )
            )

        # Newer shape.
        for entry in _ensure_list(payload.get("search_results")):
            if isinstance(entry, dict):
                _push(
                    entry.get("url"),
                    entry.get("title"),
                    entry.get("snippet") or entry.get("content"),
                )

        # Older shape.
        for entry in _ensure_list(payload.get("citations")):
            if isinstance(entry, str):
                _push(entry)
            elif isinstance(entry, dict):
                _push(
                    entry.get("url"),
                    entry.get("title"),
                    entry.get("snippet"),
                )
        return out

    # ---- public API -----------------------------------------------
    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderTextResult:
        body = {
            "model": self._model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": float(temperature),
            "return_citations": True,
        }
        payload = self._request(body)
        text = self._extract_text(payload)
        return ProviderTextResult(
            text=text,
            provider=self.provider_name,
            model=self._model,
            citations=self._extract_citations(payload),
            raw=payload,
        )

    def generate_json(
        self,
        prompt: str,
        *,
        schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
    ) -> ProviderJSONResult:
        # Perplexity supports a JSON-mode hint; even without it we can
        # bake the instruction into the system prompt and rely on
        # local validation. We do both for robustness.
        json_instruction = (
            "Respond ONLY with a single JSON object. No prose, no "
            "markdown fences. Match the requested schema exactly."
        )
        composed_system = (
            (system_prompt + "\n\n" + json_instruction).strip()
            if system_prompt
            else json_instruction
        )
        body: dict[str, Any] = {
            "model": self._model,
            "messages": self._build_messages(prompt, composed_system),
            "temperature": float(temperature),
            "return_citations": True,
            "response_format": {"type": "json_object"},
        }
        payload = self._request(body)
        text = self._extract_text(payload)
        data = parse_json_with_optional_schema(text, schema=schema)
        return ProviderJSONResult(
            data=data,
            text=text,
            provider=self.provider_name,
            model=self._model,
            citations=self._extract_citations(payload),
            raw=payload,
        )

    # ---- lifecycle ------------------------------------------------
    def close(self) -> None:  # pragma: no cover - trivial
        if self._owns_http and self._http is not None:
            self._http.close()


def _ensure_list(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, list):
        return value
    return (value,)


__all__ = ["PerplexityProvider"]
