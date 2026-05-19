"""OpenAI provider for the advisory agent layer.

Targets the chat-completions endpoint directly via ``httpx`` to keep
the dependency footprint minimal — the project already uses ``httpx``
for Discord notifications. This is the same approach the legacy
:class:`agents.llm_client.OpenAILLMClient` took; the provider here
exposes the new :class:`BaseLLMProvider` API on top of the same wire
format so per-agent routing works without a second SDK.

This provider is the right pick for:

- ``trade_analysis_agent`` — explanatory narrative, no web grounding.
- ``model_review_agent`` — calibration / drift commentary.
- ``report_agent`` — daily-report headline + bullets.
- Any structured summarization where Perplexity's web tools would only
  add noise.

If ``OPENAI_API_KEY`` is missing the constructor refuses; the router
catches that and disables every agent that's routed to ``openai``.
"""

from __future__ import annotations

from typing import Any, Optional, Type

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


class OpenAIProvider(BaseLLMProvider):
    """Minimal chat-completions wrapper.

    Always sends ``response_format={"type": "json_object"}`` for
    :meth:`generate_json` so the API itself returns a well-formed JSON
    object whenever it can; we still validate locally because
    ``response_format`` is a request, not a guarantee.
    """

    provider_name = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
    DEFAULT_MODEL = "gpt-4o-mini"

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
            raise ValueError("OpenAIProvider requires a non-empty api_key")
        self._api_key: str = api_key
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._http = http_client
        self._owns_http = http_client is None
        self._log = get_logger("agents.providers.openai")

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
            raise ProviderError(f"openai.http_error: {e}") from e

        if response.status_code >= 400:
            raise ProviderError(
                f"openai.bad_status: {response.status_code} {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise ProviderError(f"openai.bad_json: {e}") from e

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"openai.bad_payload: {e}") from e

    @staticmethod
    def _build_messages(
        prompt: str, system_prompt: Optional[str]
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

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
        }
        payload = self._request(body)
        text = self._extract_text(payload)
        return ProviderTextResult(
            text=text,
            provider=self.provider_name,
            model=self._model,
            citations=[],
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
        # Use OpenAI's native JSON-mode for tighter contracts. We still
        # parse + validate locally because ``response_format`` is a
        # best-effort hint — the agent layer must not trust the wire
        # format to be valid.
        body = {
            "model": self._model,
            "messages": self._build_messages(prompt, system_prompt),
            "temperature": float(temperature),
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
            citations=[],
            raw=payload,
        )

    # ---- lifecycle ------------------------------------------------
    def close(self) -> None:  # pragma: no cover - trivial
        if self._owns_http and self._http is not None:
            self._http.close()


__all__ = ["OpenAIProvider"]
