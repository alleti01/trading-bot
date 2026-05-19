"""Gemini (Google Generative Language) provider for the agent layer.

Targets the v1beta REST endpoint:

    POST https://generativelanguage.googleapis.com/v1beta/models/<MODEL>:generateContent?key=<KEY>

Two API quirks worth pinning down:

1. Authentication is via ``?key=`` query param, not a bearer header.
   We urlencode the key once at construction time and refuse if it's
   empty (the router catches that and disables every routed agent).
2. The system prompt is its own top-level field
   (``systemInstruction``), and content blocks are typed
   ``parts: [{"text": "..."}]`` instead of OpenAI-style messages.

For JSON-mode, Gemini accepts ``generationConfig.responseMimeType =
"application/json"`` plus ``responseSchema`` (the latter is optional;
when a Pydantic schema is provided we still validate locally because
Gemini's schema enforcement is best-effort).

Pick this provider for any reasoning agent; it's an OK alternative to
OpenAI when latency and cost matter more than peak quality.
"""

from __future__ import annotations

import urllib.parse
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


class GeminiProvider(BaseLLMProvider):
    """Gemini ``generateContent`` wrapper."""

    provider_name = "gemini"
    DEFAULT_BASE_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
    )
    DEFAULT_MODEL = "gemini-1.5-flash"

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
            raise ValueError("GeminiProvider requires a non-empty api_key")
        self._api_key: str = api_key
        self._base_url = base_url or self.DEFAULT_BASE_URL
        self._http = http_client
        self._owns_http = http_client is None
        self._log = get_logger("agents.providers.gemini")

    # ---- HTTP -------------------------------------------------------
    def _endpoint(self) -> str:
        # ``models/<MODEL>:generateContent?key=<KEY>``. The model id
        # may itself contain a slash (e.g. ``gemini-2.0-flash-exp``) so
        # we rstrip + join carefully.
        base = self._base_url.rstrip("/") + "/"
        return (
            f"{base}{self._model}:generateContent?key="
            f"{urllib.parse.quote(self._api_key, safe='')}"
        )

    def _request(self, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        try:
            if self._http is not None:
                response = self._http.post(
                    self._endpoint(),
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
            else:
                response = httpx.post(
                    self._endpoint(),
                    headers=headers,
                    json=body,
                    timeout=self._timeout,
                )
        except httpx.HTTPError as e:
            raise ProviderError(f"gemini.http_error: {e}") from e

        if response.status_code >= 400:
            raise ProviderError(
                f"gemini.bad_status: {response.status_code} "
                f"{response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as e:
            raise ProviderError(f"gemini.bad_json: {e}") from e

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        try:
            candidates = payload.get("candidates") or []
            if not candidates:
                raise ProviderError("gemini.bad_payload: no candidates")
            parts = candidates[0].get("content", {}).get("parts") or []
            chunks = [str(p.get("text", "")) for p in parts if isinstance(p, dict)]
            joined = "".join(chunks).strip()
            if not joined:
                raise ProviderError("gemini.bad_payload: empty text parts")
            return joined
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"gemini.bad_payload: {e}") from e

    # ---- public API -----------------------------------------------
    def _build_body(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str],
        temperature: float,
        json_mode: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "contents": [
                {"role": "user", "parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "temperature": float(temperature),
            },
        }
        if system_prompt:
            body["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": system_prompt}],
            }
        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"
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
            prompt,
            system_prompt=composed_system,
            temperature=temperature,
            json_mode=True,
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


__all__ = ["GeminiProvider"]
