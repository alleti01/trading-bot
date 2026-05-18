"""Discord notifier reliability — Retry-After + 5xx retry behavior.

The notifier must:

- Retry on transient 5xx errors with exponential backoff (up to
  ``max_retries`` total).
- Respect Discord's ``Retry-After`` header on 429 for high-priority
  kinds within ``max_wait_seconds``.
- Never raise — return a ``SendResult`` with the failure reason instead.
"""

from __future__ import annotations

from typing import Optional

import httpx

from notifications.discord import DiscordNotifier


class _MockResponse:
    def __init__(
        self,
        *,
        status_code: int,
        text: str = "",
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self.status_code = int(status_code)
        self.text = str(text)
        self.headers = dict(headers or {})


def _patch_post(monkeypatch, responses: list[_MockResponse]) -> list[dict]:
    """Patch ``httpx.post`` to consume ``responses`` in order. Returns the
    list of recorded calls (one dict per attempt)."""
    calls: list[dict] = []
    iterator = iter(responses)

    def fake_post(url, *, json, timeout):  # noqa: ARG001
        calls.append({"url": url, "json": json})
        try:
            return next(iterator)
        except StopIteration:
            return _MockResponse(status_code=204)

    monkeypatch.setattr(httpx, "post", fake_post)
    return calls


def test_retries_on_5xx_then_succeeds(monkeypatch) -> None:
    notifier = DiscordNotifier(
        "https://example.com/webhook",
        max_retries=2,
        retry_backoff_seconds=0.0,
    )
    calls = _patch_post(
        monkeypatch,
        [
            _MockResponse(status_code=502, text="bad gateway"),
            _MockResponse(status_code=204),
        ],
    )
    res = notifier.send("trade.opened", {"instrument": "MES"})
    assert res.delivered is True
    assert res.status_code == 204
    assert len(calls) == 2


def test_gives_up_after_max_retries(monkeypatch) -> None:
    notifier = DiscordNotifier(
        "https://example.com/webhook",
        max_retries=1,  # 2 attempts total
        retry_backoff_seconds=0.0,
    )
    _patch_post(
        monkeypatch,
        [
            _MockResponse(status_code=503, text="service down"),
            _MockResponse(status_code=503, text="still down"),
        ],
    )
    res = notifier.send("trade.opened", {"instrument": "MES"})
    assert res.delivered is False
    assert res.status_code == 503


def test_respects_retry_after_for_high_priority(monkeypatch) -> None:
    notifier = DiscordNotifier(
        "https://example.com/webhook",
        max_retries=1,
        retry_backoff_seconds=0.0,
        max_wait_seconds=5.0,
    )
    sleep_calls: list[float] = []

    def fake_sleep(secs: float) -> None:
        sleep_calls.append(secs)

    monkeypatch.setattr("notifications.discord.time.sleep", fake_sleep)
    _patch_post(
        monkeypatch,
        [
            _MockResponse(status_code=429, headers={"Retry-After": "0.25"}),
            _MockResponse(status_code=204),
        ],
    )
    res = notifier.send("trade.opened", {"instrument": "MES"})
    assert res.delivered is True
    # We slept the value of Retry-After before retrying.
    assert any(abs(s - 0.25) < 1e-6 for s in sleep_calls)


def test_does_not_wait_too_long_for_low_priority(monkeypatch) -> None:
    notifier = DiscordNotifier(
        "https://example.com/webhook",
        max_retries=1,
        retry_backoff_seconds=0.0,
        max_wait_seconds=1.0,
    )
    _patch_post(
        monkeypatch,
        [
            _MockResponse(status_code=429, headers={"Retry-After": "10.0"}),
        ],
    )
    res = notifier.send("signal.generated", {"sym": "MES"})
    # 10s is past max_wait — caller must NOT block; reports rate-limited.
    assert res.delivered is False
    assert res.dropped_reason == "rate_limited_remote"


def test_4xx_other_than_429_does_not_retry(monkeypatch) -> None:
    notifier = DiscordNotifier(
        "https://example.com/webhook",
        max_retries=3,
        retry_backoff_seconds=0.0,
    )
    calls = _patch_post(
        monkeypatch,
        [_MockResponse(status_code=400, text="bad request")],
    )
    res = notifier.send("trade.opened", {})
    assert res.delivered is False
    # Only one call — 4xx is non-retryable.
    assert len(calls) == 1


def test_does_not_raise_on_network_failure(monkeypatch) -> None:
    notifier = DiscordNotifier(
        "https://example.com/webhook",
        max_retries=0,
    )

    def boom(*a, **k):
        raise httpx.ConnectError("dns fail")

    monkeypatch.setattr(httpx, "post", boom)
    res = notifier.send("trade.opened", {})
    assert res.delivered is False
    assert res.error and "dns fail" in res.error
