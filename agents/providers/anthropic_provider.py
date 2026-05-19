"""Anthropic (Claude) provider for the advisory agent layer.

Targets ``POST https://api.anthropic.com/v1/messages``. Anthropic's
messages API differs from OpenAI's in two ways that matter here:

1. The system prompt is its own top-level field (``system``), not a
   message with ``role=system``.
2. The response shape is ``content: [{"type": "text", "text": "..."}]``
   rather than ``choices[0].message.content``.

Anthropic does not natively expose a JSON-mode flag, so
:meth:`generate_json` relies on prompt instructions plus local
validation — same robustness story as the legacy OpenAI client when it
ran without ``response_format``.

This provider is a drop-in alternative for any reasoning agent
(``trade_analysis_agent``, ``model_review_agent``,
``report_agent``). Set ``TRADE_ANALYSIS_AGENT_PROVIDER=anthropic`` in
``.env`` to route a single agent without touching code.
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


class AnthropicProvider(BaseLLMProvider):
    """Claude messages-API wrapper."""

    provider_name = "anthropic"
    DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"
    DEFAULT_MODEL = "claude-3-5-sonnet-latest"
    API_VERSION = "2023-06-01"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        api_key: str | SecretStr,
        *,
        model: Optional[str] = None,
        timeout_seconds: float = 30.0,
        base_url: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        super().__init__(
            model=model or self.DEFAULT_MODEL,
            timeout_seconds=timeout_seconds,
        )
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()
        if not api_key:
            raise ValueError("AnthropicProvider requires a non-empty api_key")
        self._api_key: str = api_key
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._http = http_client
        self._owns_http = http_client is None
        self._max_tokens = int(max_tokens)
        self._log = get_logger("agents.providers.anthropic")

    # ---- HTTP -------------------------------------------------------
    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": self.API_VERSION,
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
            raise ProviderError(f"anthropic.http_error: {e}") from e

        if response.status_code >= 400:
            raise ProviderError(
                f"anthropic.bad_status: {response.status_code} "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise ProviderError(f"anthropic.bad_json: {e}") from e

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        # ``content`` is a list of typed blocks; we concatenate every
        # ``text`` block in order. Tool-use blocks are ignored — the
        # advisory agent layer never asks for them.
        try:
            blocks = payload.get("content") or []
            chunks: list[str] = []
            for block in blocks:
                if isinstance(block, dict) and block.get("type") == "text":
                    chunks.append(str(block.get("text", "")))
            joined = "".join(chunks).strip()
            if not joined:
                raise ProviderError("anthropic.bad_payload: empty content")
            return joined
        except (KeyError, TypeError) as e:
            raise ProviderError(f"anthropic.bad_payload: {e}") from e

    # ---- public API -----------------------------------------------
    def _build_body(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str],
        temperature: float,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "temperature": float(temperature),
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            body["system"] = system_prompt
        return body

    def generate_text(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
    ) -> ProviderTextResult:
        body = self._build_body(
            prompt, system_prompt=system_prompt, temperature=temperature
        )
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
        # Anthropic does not yet expose a JSON-mode flag, so we lean on
        # an explicit instruction baked into the system prompt + local
        # validation. ``parse_json_with_optional_schema`` strips any
        # markdown fences a model might emit.
        json_instruction = (
            "Respond ONLY with a single JSON object. No prose, no "
            "markdown fences. Match the requested schema exactly."
        )
        composed_system = (
            (system_prompt + "\n\n" + json_instruction).strip()
            if system_prompt
            else json_instruction
        )
        body = self._build_body(
            prompt, system_prompt=composed_system, temperature=temperature
        )
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

    def close(self) -> None:  # pragma: no cover - trivial
        if self._owns_http and self._http is not None:
            self._http.close()


__all__ = ["AnthropicProvider"]
