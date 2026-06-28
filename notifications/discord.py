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
        "options.closed",
    }
)

# Kinds that describe a *closed* trade and get the rich, human-readable
# "Trade Closed" layout (gain/loss headline) instead of the generic
# key/value dump.
CLOSE_KINDS: frozenset[str] = frozenset(
    {
        "trade.closed",
        "forced_flat",
        "webhook.closed",
        "options.closed",
    }
)


def _fmt_signed_money(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def _fmt_signed_pct(value: float) -> str:
    sign = "+" if value > 0 else "-" if value < 0 else ""
    return f"{sign}{abs(value):.2f}%"


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
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.webhook_url = (webhook_url or "").strip()
        self.timeout = float(timeout)
        self.max_wait_seconds = float(max_wait_seconds)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.limiter = rate_limiter or DiscordRateLimiter()
        self.log = get_logger("notifications.discord")

    def _format_content(self, kind: str, payload: dict) -> str:
        if kind in CLOSE_KINDS:
            body = self._format_trade_closed(kind, payload)
        else:
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

    def _format_trade_closed(self, kind: str, payload: dict) -> str:
        """Render a closed trade as a clear gain/loss message.

        Falls back gracefully: any field that is missing is simply omitted,
        so this stays safe even for sparse payloads.
        """
        result = str(payload.get("result", "")).upper()
        symbol = payload.get("instrument") or payload.get("symbol") or "?"
        direction = str(payload.get("direction", "")).lower()
        net = payload.get("net_pnl")
        return_pct = payload.get("return_pct")
        entry = payload.get("entry_price")
        exit_price = payload.get("exit_price")
        qty = payload.get("quantity")
        bars_held = payload.get("bars_held")
        costs = payload.get("costs")
        exit_reason = payload.get("exit_reason")

        if result == "WIN":
            emoji, verb = "🟢", "gained"
        elif result == "LOSS":
            emoji, verb = "🔴", "lost"
        else:
            emoji, verb = "⚪", "broke even"

        header = f"{emoji} **Trade Closed — {result or 'DONE'}**"
        if kind == "forced_flat":
            header += " (end-of-day flatten)"
        lines = [header]

        # Headline the user actually cares about: how much did I make/lose?
        if isinstance(net, (int, float)):
            money = _fmt_signed_money(float(net))
            pct = ""
            if isinstance(return_pct, (int, float)):
                pct = f" ({_fmt_signed_pct(float(return_pct))})"
            if result == "BREAKEVEN":
                lines.append(f"Flat — **{money}**{pct}")
            else:
                lines.append(f"You {verb} **{money}**{pct}")

        # Instrument / direction / exit reason context.
        ctx = [f"`{symbol}`"]
        if direction:
            ctx.append(direction)
        if exit_reason:
            ctx.append(f"exit: {exit_reason}")
        lines.append(" · ".join(ctx))

        # Trade mechanics: entry → exit, size, hold time, fees.
        detail: list[str] = []
        if isinstance(entry, (int, float)) and isinstance(exit_price, (int, float)):
            detail.append(f"{entry:g} → {exit_price:g}")
        if isinstance(qty, (int, float)):
            detail.append(f"qty {qty:g}")
        elif qty is not None:
            detail.append(f"qty {qty}")
        if bars_held is not None:
            detail.append(f"{bars_held} bars")
        if isinstance(costs, (int, float)) and costs:
            detail.append(f"fees ${abs(float(costs)):,.2f}")
        if detail:
            lines.append(" · ".join(detail))

        return "\n".join(lines)

    @staticmethod
    def _parse_retry_after(response: "httpx.Response") -> float:
        """Best-effort parse of Discord's Retry-After header (seconds)."""
        retry_after = response.headers.get("Retry-After") or response.headers.get(
            "retry-after"
        )
        if retry_after is None:
            return 0.0
        try:
            return max(0.0, float(retry_after))
        except (ValueError, TypeError):
            return 0.0

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

        last_error: Optional[str] = None
        last_status: Optional[int] = None
        attempts = self.max_retries + 1
        for attempt in range(attempts):
            try:
                response = httpx.post(
                    self.webhook_url,
                    json={"content": content},
                    timeout=self.timeout,
                )
                self.limiter.record()
            except httpx.HTTPError as e:
                last_error = str(e)
                self.log.warning(
                    "discord.http_error",
                    kind=kind,
                    error=last_error,
                    attempt=attempt + 1,
                )
                if attempt < attempts - 1:
                    time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                    continue
                return SendResult(delivered=False, error=last_error)
            except Exception as e:  # pragma: no cover - belt-and-braces
                self.log.error("discord.unexpected_error", kind=kind, error=str(e))
                return SendResult(delivered=False, error=str(e))

            last_status = response.status_code

            if 200 <= response.status_code < 300:
                return SendResult(delivered=True, status_code=response.status_code)

            # 429: respect server-side Retry-After.
            if response.status_code == 429:
                wait_s = self._parse_retry_after(response)
                self.log.warning(
                    "discord.rate_limited_remote",
                    kind=kind,
                    retry_after=round(wait_s, 2),
                    attempt=attempt + 1,
                )
                if (
                    attempt < attempts - 1
                    and kind in HIGH_PRIORITY_KINDS
                    and wait_s <= self.max_wait_seconds
                ):
                    time.sleep(wait_s)
                    continue
                return SendResult(
                    delivered=False,
                    status_code=response.status_code,
                    dropped_reason="rate_limited_remote",
                    error=f"HTTP 429 retry_after={wait_s:.2f}s",
                )

            # 5xx: transient, retry with exponential backoff.
            if 500 <= response.status_code < 600 and attempt < attempts - 1:
                self.log.warning(
                    "discord.transient_error",
                    kind=kind,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )
                time.sleep(self.retry_backoff_seconds * (2 ** attempt))
                continue

            # 4xx (other than 429) — no point retrying.
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

        return SendResult(
            delivered=False,
            status_code=last_status,
            error=last_error or "all retries exhausted",
        )
