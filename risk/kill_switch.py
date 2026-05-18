"""DB-persisted kill switch.

Once tripped, stays tripped across process restarts. Manual reset only —
a crash loop must not be able to silently re-enable trading.

Usage::

    if KillSwitch().is_tripped():
        return RiskDecision(allowed=False, rule="kill_switch", reason=...)

For tests, ``KillSwitch.reset_manual()`` clears it; production code never
auto-resets.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select

from app.logging_config import get_logger
from storage.db import session_scope
from storage.tables import KillSwitchState


@dataclass(frozen=True)
class KillSwitchSnapshot:
    tripped: bool
    reason: Optional[str]
    tripped_at: Optional[datetime]


class KillSwitch:
    """Thin wrapper around the ``kill_switch_state`` row (id=1)."""

    def __init__(self, *, row_id: int = 1) -> None:
        self.row_id = int(row_id)

    # ----------------------------------------------------------------------
    def _read(self) -> KillSwitchSnapshot:
        with session_scope() as session:
            row = session.execute(
                select(KillSwitchState).where(KillSwitchState.id == self.row_id)
            ).scalar_one_or_none()
            if row is None:
                return KillSwitchSnapshot(tripped=False, reason=None, tripped_at=None)
            return KillSwitchSnapshot(
                tripped=bool(row.tripped),
                reason=row.reason,
                tripped_at=row.tripped_at,
            )

    def snapshot(self) -> KillSwitchSnapshot:
        return self._read()

    def is_tripped(self) -> bool:
        return self._read().tripped

    # ----------------------------------------------------------------------
    def trip(self, reason: str) -> KillSwitchSnapshot:
        log = get_logger("risk.kill_switch")
        now = datetime.now(timezone.utc)
        with session_scope() as session:
            row = session.execute(
                select(KillSwitchState).where(KillSwitchState.id == self.row_id)
            ).scalar_one_or_none()
            if row is None:
                row = KillSwitchState(
                    id=self.row_id, tripped=True, reason=reason, tripped_at=now
                )
                session.add(row)
            else:
                row.tripped = True
                row.reason = reason
                row.tripped_at = now
        log.error("kill_switch.tripped", reason=reason)
        return KillSwitchSnapshot(tripped=True, reason=reason, tripped_at=now)

    def reset_manual(self) -> None:
        """Explicit reset. Intentionally not auto-called anywhere."""
        log = get_logger("risk.kill_switch")
        with session_scope() as session:
            row = session.execute(
                select(KillSwitchState).where(KillSwitchState.id == self.row_id)
            ).scalar_one_or_none()
            if row is not None:
                row.tripped = False
                row.reason = None
                row.tripped_at = None
        log.warning("kill_switch.manual_reset")
