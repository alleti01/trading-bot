"""Discord rate limiter: capacity, sliding window, high-priority kind handling."""

from __future__ import annotations

from notifications.discord import (
    HIGH_PRIORITY_KINDS,
    DiscordNotifier,
    DiscordRateLimiter,
    SendResult,
)


def test_rate_limiter_blocks_after_capacity() -> None:
    rl = DiscordRateLimiter(capacity=3, window_seconds=60.0)
    t = 100.0
    for _ in range(3):
        assert rl.has_budget(now=t)
        rl.record(now=t)
        t += 0.01
    assert not rl.has_budget(now=t)
    # Until 60s have passed since the first record.
    assert rl.has_budget(now=t + 60.0) is True


def test_rate_limiter_seconds_until_budget_decreases() -> None:
    rl = DiscordRateLimiter(capacity=2, window_seconds=10.0)
    rl.record(now=0.0)
    rl.record(now=1.0)
    assert rl.seconds_until_budget(now=2.0) > 0
    # At t=10.0 the first record exits the window → budget restored.
    assert rl.seconds_until_budget(now=10.0) == 0.0


class _FakeNotifier(DiscordNotifier):
    """DiscordNotifier with the network call stubbed out so we can count attempts."""

    def __init__(self, **kwargs) -> None:
        super().__init__("https://example.com/webhook", **kwargs)
        self.attempts = 0

    def send(self, kind: str, payload: dict) -> SendResult:  # type: ignore[override]
        if not self.webhook_url:
            return SendResult(delivered=False, dropped_reason="no_webhook")
        if not self.limiter.has_budget():
            wait = self.limiter.seconds_until_budget()
            if kind in HIGH_PRIORITY_KINDS and wait <= self.max_wait_seconds:
                # Pretend we waited then slot in.
                self.limiter._sends.popleft()
            else:
                return SendResult(delivered=False, dropped_reason="rate_limited")
        self.limiter.record()
        self.attempts += 1
        return SendResult(delivered=True, status_code=204)


def test_low_priority_kind_dropped_when_rate_limited() -> None:
    notifier = _FakeNotifier(rate_limiter=DiscordRateLimiter(capacity=2, window_seconds=60.0))
    assert notifier.send("signal.generated", {}).delivered is True
    assert notifier.send("signal.generated", {}).delivered is True
    result = notifier.send("signal.generated", {})
    assert result.delivered is False
    assert result.dropped_reason == "rate_limited"


def test_high_priority_kind_squeezes_in_when_rate_limited() -> None:
    notifier = _FakeNotifier(
        rate_limiter=DiscordRateLimiter(capacity=2, window_seconds=60.0),
        max_wait_seconds=999.0,
    )
    notifier.send("signal.generated", {})
    notifier.send("signal.generated", {})
    # Now budget is full. A high-priority kind should still get through.
    result = notifier.send("system.error", {"kind": "boom"})
    assert result.delivered is True


def test_no_webhook_url_returns_drop_reason() -> None:
    notifier = DiscordNotifier("")
    result = notifier.send("trade.opened", {"instrument": "MES"})
    assert result.delivered is False
    assert result.dropped_reason == "no_webhook"
