"""High-level notification dispatcher.

The service is the single entry point used by the scheduler and the paper
loop. It:

- persists every notification attempt (success or failure) to the
  ``notifications`` table for the audit trail,
- catches every exception that could come from the underlying channel —
  notification failures must never crash the bot,
- when no Discord webhook is configured, falls back to a "log-only"
  channel so the audit trail still gets a row.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from app.logging_config import get_logger
from notifications.discord import DiscordNotifier, SendResult
from storage.db import session_scope
from storage.tables import Notification


def _coerce_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Ensure the payload is JSON-serializable. Anything weird becomes ``str``."""
    safe: dict[str, Any] = {}
    for k, v in payload.items():
        try:
            json.dumps(v)
            safe[k] = v
        except TypeError:
            safe[k] = str(v)
    return safe


class NotificationService:
    """Fan-out to channels + persistence + safety wrapping.

    Today there is exactly one channel (Discord). The interface is built
    so adding email/SMS/Slack later is straight-forward.
    """

    def __init__(
        self,
        *,
        discord: Optional[DiscordNotifier] = None,
    ) -> None:
        self.discord = discord
        self.log = get_logger("notifications.service")

    @classmethod
    def from_settings(cls, settings) -> "NotificationService":
        url = (
            settings.DISCORD_WEBHOOK_URL.get_secret_value()
            if settings.DISCORD_WEBHOOK_URL is not None
            else ""
        )
        return cls(discord=DiscordNotifier(url))

    def notify(self, kind: str, /, **payload: Any) -> None:
        """Dispatch to channel and persist the outcome.

        This method MUST NOT raise. Any unexpected exception from a
        channel is caught and recorded as an error notification.
        """
        safe_payload = _coerce_payload(payload)

        if self.discord is None or not self.discord.webhook_url:
            self._persist(channel="log", kind=kind, payload=safe_payload, delivered=True)
            # Avoid kwarg collision when the payload itself contains "kind".
            log_kwargs = {k: v for k, v in safe_payload.items() if k != "kind"}
            self.log.info("notify.log_only", kind=kind, **log_kwargs)
            return

        try:
            result: SendResult = self.discord.send(kind, safe_payload)
        except Exception as e:  # pragma: no cover - belt-and-braces
            self.log.error("notify.unexpected", kind=kind, error=str(e))
            self._persist(
                channel="discord",
                kind=kind,
                payload=safe_payload,
                delivered=False,
                error=str(e),
            )
            return

        self._persist(
            channel="discord",
            kind=kind,
            payload=safe_payload,
            delivered=result.delivered,
            error=result.error or result.dropped_reason,
        )

        if result.delivered:
            self.log.info("notify.sent", kind=kind, status=result.status_code)
        else:
            self.log.warning(
                "notify.failed",
                kind=kind,
                error=result.error,
                dropped_reason=result.dropped_reason,
            )

    def _persist(
        self,
        *,
        channel: str,
        kind: str,
        payload: dict[str, Any],
        delivered: bool,
        error: Optional[str] = None,
    ) -> None:
        try:
            with session_scope() as session:
                row = Notification(
                    channel=channel,
                    kind=kind,
                    payload=payload,
                    delivered=bool(delivered),
                    error=error,
                )
                session.add(row)
        except Exception as e:
            # If even the DB write fails, log it but do not raise — that
            # would defeat the "never crash" guarantee.
            self.log.error("notify.persist_failed", kind=kind, error=str(e))
