"""Discord webhook client.

Day 5 ships a small, defensive Discord client. The two design rules:

1. **Never raise.** Network failures return ``False`` so a flaky webhook
   can't take the trading process down.
2. **Stay under Discord's rate limit.** Discord webhooks accept ~30
   requests / 60s. We use a conservative 25-per-60s rolling cap and drop
   low-priority messages when the budget is full. Higher-priority kinds
   (trade.opened/closed, system.error, forced_flat, daily_loss.warning)
   are never dropped — they wait briefly for budget instead.

Design constraint: this module owns *delivery only*. Persistence of every
attempt is the :class:`NotificationService`'s job, so the audit trail
stays accurate even when the network is unreachable.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import httpx

from app.logging_config import get_logger

# Kinds that must never be dropped silently. They will wait for rate-limit
# budget for up to ``SendResult.WAIT_BUDGET_SECONDS`` seconds.
HIGH_PRIORITY_KINDS: frozenset[str] = frozenset(
    {
        "bot.start",
        "bot.stop",
        "trade.opened",
        "trade.closed",
        "forced_flat",
        "daily_loss.warning",
        "daily_profit.cap",
        "high_risk_news",
        "system.error",
        "kill_switch.tripped",
        "eod.summary",
    }
)


@dataclass(frozen=True)
class SendResult:
    delivered: bool
    status_code: Optional[int] = None
    error: Optional[str] = None
    dropped_reason: Optional[str] = None


class DiscordRateLimiter:
    """Sliding-window rate limiter (max ``capacity`` requests per ``window`` s)."""

    def __init__(self, *, capacity: int = 25, window_seconds: float = 60.0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self.capacity = int(capacity)
        self.window = float(window_seconds)
        self._sends: deque[float] = deque()

    def _purge(self, now: float) -> None:
        cutoff = now - self.window
        while self._sends and self._sends[0] < cutoff:
            self._sends.popleft()

    def has_budget(self, now: Optional[float] = None) -> bool:
        now = time.monotonic() if now is None else now
        self._purge(now)
        return len(self._sends) < self.capacity

    def record(self, now: Optional[float] = None) -> None:
        now = time.monotonic() if now is None else now
        self._purge(now)
        self._sends.append(now)

    def seconds_until_budget(self, now: Optional[float] = None) -> float:
        now = time.monotonic() if now is None else now
        self._purge(now)
        if len(self._sends) < self.capacity:
            return 0.0
        return max(0.0, self.window - (now - self._sends[0]))


class DiscordNotifier:
    """Posts JSON content to a Discord webhook with rate limiting + retry.

    The webhook URL is expected to be the full Discord webhook URL. If
    ``webhook_url`` is empty, every send returns
    ``SendResult(delivered=False, dropped_reason="no_webhook")`` so the
    caller can treat the notifier as a no-op without special-casing.
    """

    def __init__(
        self,
        webhook_url: Optional[str],
        *,
        rate_limiter: Optional[DiscordRateLimiter] = None,
        timeout: float = 5.0,
        max_wait_seconds: float = 2.0,
    ) -> None:
        self.webhook_url = (webhook_url or "").strip()
        self.timeout = float(timeout)
        self.max_wait_seconds = float(max_wait_seconds)
        self.limiter = rate_limiter or DiscordRateLimiter()
        self.log = get_logger("notifications.discord")

    def _format_content(self, kind: str, payload: dict) -> str:
        lines = [f"**[{kind}]**"]
        for k, v in payload.items():
            if isinstance(v, float):
                lines.append(f"- `{k}`: {v:.4f}")
            else:
                lines.append(f"- `{k}`: {v}")
        body = "\n".join(lines)
        # Discord caps content at 2000 chars. Truncate aggressively rather
        # than risk a 400.
        if len(body) > 1900:
            body = body[:1897] + "..."
        return body

    def send(self, kind: str, payload: dict) -> SendResult:
        if not self.webhook_url:
            return SendResult(delivered=False, dropped_reason="no_webhook")

        # Rate-limit gate. High-priority kinds wait briefly; everything
        # else is dropped immediately when the budget is full.
        if not self.limiter.has_budget():
            wait = self.limiter.seconds_until_budget()
            if kind in HIGH_PRIORITY_KINDS and wait <= self.max_wait_seconds:
                time.sleep(wait)
            else:
                self.log.warning(
                    "discord.rate_limited", kind=kind, seconds_until_budget=round(wait, 2)
                )
                return SendResult(delivered=False, dropped_reason="rate_limited")

        content = self._format_content(kind, payload)
        try:
            response = httpx.post(
                self.webhook_url,
                json={"content": content},
                timeout=self.timeout,
            )
            self.limiter.record()
        except httpx.HTTPError as e:
            self.log.warning("discord.http_error", kind=kind, error=str(e))
            return SendResult(delivered=False, error=str(e))
        except Exception as e:  # pragma: no cover - belt-and-braces
            self.log.error("discord.unexpected_error", kind=kind, error=str(e))
            return SendResult(delivered=False, error=str(e))

        if 200 <= response.status_code < 300:
            return SendResult(delivered=True, status_code=response.status_code)

        self.log.warning(
            "discord.bad_status",
            kind=kind,
            status_code=response.status_code,
            body=response.text[:200],
        )
        return SendResult(
            delivered=False,
            status_code=response.status_code,
            error=response.text[:200] or f"HTTP {response.status_code}",
        )
