"""NotificationService: persists every attempt; never raises; log-only fallback."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select

from notifications.discord import DiscordNotifier, SendResult
from notifications.notification_service import NotificationService
from storage.db import init_db, session_scope
from storage.tables import Notification


class _FakeDiscord(DiscordNotifier):
    def __init__(self, result: SendResult, raises: Optional[Exception] = None) -> None:
        # Skip parent init - we only need send/webhook_url.
        self.webhook_url = "https://example.com/webhook/x"
        self._result = result
        self._raises = raises
        self.calls: list[tuple[str, dict]] = []

    def send(self, kind: str, payload: dict) -> SendResult:  # type: ignore[override]
        self.calls.append((kind, payload))
        if self._raises is not None:
            raise self._raises
        return self._result


def test_logs_only_when_no_webhook() -> None:
    init_db()
    service = NotificationService(discord=DiscordNotifier(""))
    service.notify("trade.opened", instrument="MES", direction="long")

    with session_scope() as session:
        rows = session.execute(select(Notification)).scalars().all()
    assert len(rows) == 1
    assert rows[0].channel == "log"
    assert rows[0].delivered is True
    assert rows[0].kind == "trade.opened"


def test_persists_success_when_discord_delivers() -> None:
    init_db()
    fake = _FakeDiscord(SendResult(delivered=True, status_code=204))
    service = NotificationService(discord=fake)
    service.notify("signal.generated", instrument="MES", direction="short")

    with session_scope() as session:
        rows = session.execute(select(Notification)).scalars().all()
    assert len(rows) == 1
    assert rows[0].channel == "discord"
    assert rows[0].delivered is True
    assert rows[0].kind == "signal.generated"
    assert fake.calls == [("signal.generated", {"instrument": "MES", "direction": "short"})]


def test_persists_failure_when_discord_returns_error() -> None:
    init_db()
    fake = _FakeDiscord(SendResult(delivered=False, error="HTTP 500"))
    service = NotificationService(discord=fake)
    service.notify("trade.blocked", rule="max_daily_loss")

    with session_scope() as session:
        rows = session.execute(select(Notification)).scalars().all()
    assert rows[0].delivered is False
    assert rows[0].error == "HTTP 500"


def test_does_not_raise_when_discord_throws() -> None:
    init_db()
    fake = _FakeDiscord(SendResult(delivered=True), raises=RuntimeError("boom"))
    service = NotificationService(discord=fake)
    # Must not raise.
    service.notify("system.error", kind="boom")

    with session_scope() as session:
        rows = session.execute(select(Notification)).scalars().all()
    assert rows[0].delivered is False
    assert "boom" in (rows[0].error or "")


def test_payload_is_coerced_to_serializable() -> None:
    init_db()

    class _NotJSON:
        def __repr__(self) -> str:
            return "_NotJSON()"

    fake = _FakeDiscord(SendResult(delivered=True))
    service = NotificationService(discord=fake)
    service.notify("trade.opened", instrument="MES", obj=_NotJSON())

    with session_scope() as session:
        rows = session.execute(select(Notification)).scalars().all()
    payload = rows[0].payload
    # The non-serializable object should have been str()'d.
    assert payload["obj"] == "_NotJSON()"
