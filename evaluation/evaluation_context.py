"""Isolated evaluation context per broker/provider track."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvaluationContext:
    """Per-broker evaluation state — completely isolated between tracks.

    Each track has its own symbol universe, state file, report directory,
    and evaluation id. The parallel runner creates one of these per
    configured broker provider so they can never bleed into each other.
    """

    evaluation_id: str
    broker_provider: str
    enabled_symbols: list[str]
    state_path: Path
    report_path: Path
    started_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    # Mutable runtime state
    trades: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    _blocked: bool = False

    def __post_init__(self) -> None:
        self.state_path = Path(self.state_path)
        self.report_path = Path(self.report_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.mkdir(parents=True, exist_ok=True)

    @property
    def blocked(self) -> bool:
        return self._blocked

    def block(self, reason: str) -> None:
        self._blocked = True
        self.errors.append(reason)

    def record_trade(self, trade: dict[str, Any]) -> None:
        trade.setdefault("broker_provider", self.broker_provider)
        trade.setdefault("evaluation_id", self.evaluation_id)
        self.trades.append(trade)

    def save_state(self) -> None:
        payload = {
            "evaluation_id": self.evaluation_id,
            "broker_provider": self.broker_provider,
            "enabled_symbols": self.enabled_symbols,
            "started_at": self.started_at.isoformat(),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "trade_count": len(self.trades),
            "errors": self.errors[-20:],
            "blocked": self._blocked,
        }
        self.state_path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

    def load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def discord_tags(self) -> dict[str, str]:
        return {
            "broker_provider": self.broker_provider,
            "evaluation_id": self.evaluation_id,
        }
