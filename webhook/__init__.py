"""External webhook signal inputs (optional).

Webhooks are an *additional* signal source for the bot — never the only
one. The bot's primary scanning system is :class:`MultiSymbolPaperLoop`,
which polls per-symbol feeds on the configured cadence regardless of
whether any webhook is wired up.

Currently shipped: :func:`validate_webhook_signal` for TradingView-shaped
JSON payloads. The validator's job is to (a) filter by
``ENABLED_SYMBOLS`` and (b) refuse malformed inputs without opinions on
which executor consumes the signal afterwards.
"""

from webhook.tradingview import (
    InvalidWebhookSignal,
    WebhookSignal,
    validate_webhook_signal,
)

__all__ = [
    "InvalidWebhookSignal",
    "WebhookSignal",
    "validate_webhook_signal",
]
